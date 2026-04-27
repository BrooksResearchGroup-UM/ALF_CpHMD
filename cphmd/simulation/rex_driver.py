from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.utils.seeds import derive_seed

__all__ = [
    "ExchangeStats",
    "REXConfig",
    "REXDriver",
    "REXStepResult",
]


@dataclass(frozen=True)
class REXConfig:
    enabled: bool = False
    exchange_every_segments: int = 1


@dataclass(frozen=True)
class ExchangeStats:
    attempted: tuple[int, ...]
    accepted: tuple[int, ...]

    @classmethod
    def empty(cls, n_pairs: int) -> "ExchangeStats":
        return cls(tuple(0 for _ in range(n_pairs)), tuple(0 for _ in range(n_pairs)))

    def record_decisions(self, decisions: tuple[dict, ...]) -> "ExchangeStats":
        attempted = list(self.attempted)
        accepted = list(self.accepted)
        for decision in decisions:
            pair_idx = min(decision["ranks"])
            attempted[pair_idx] += 1
            if decision.get("accepted"):
                accepted[pair_idx] += 1
        return ExchangeStats(tuple(attempted), tuple(accepted))


@dataclass(frozen=True)
class REXStepResult:
    replica_label: int
    attempted: tuple[int, ...]
    accepted: tuple[int, ...]
    exchange_result: Any


class REXDriver:
    """Thermodynamic-neighbor label diffusion using pyCHARMM direct state exchange."""

    def __init__(self, ctx: RunContext, *, native_rex: Any | None = None):
        self.ctx = ctx
        self.config = REXConfig(
            enabled=ctx.rex_enabled,
            exchange_every_segments=ctx.rex_exchange_every_segments,
        )
        self.native_rex = native_rex if native_rex is not None else self._load_native_rex()
        self.exchanger = self.native_rex.build_exchanger(
            comm=ctx.comm,
            seed=derive_seed(ctx.master_seed, "rex", ctx.rank),
        )

    def should_attempt(self, state: LoopState) -> bool:
        return (
            self.ctx.rex_enabled
            and state.segment_idx > 0
            and state.segment_idx % self.ctx.rex_exchange_every_segments == 0
        )

    def attempt(self, state: LoopState) -> REXStepResult:
        current_label = self._current_label(state)
        current_state = state.with_initial_label(current_label)
        local_ph = self.ctx.replica_ph_values[current_label]
        local_state = self.native_rex.snapshot_state(
            self.exchanger,
            label=current_label,
            ldin_blocks=self._direct_ldin_blocks(),
            msld_theta_blocks=self._msld_theta_blocks(),
            include_lambdas=True,
            include_ph=True,
            include_ffix=True,
            metadata=self._state_metadata(current_label, local_ph),
        )
        local_state = _with_canonical_ph(local_state, local_ph)
        self._debug_state("pre", state, local_state)
        signs = self._sign_vector_for_state(local_state)

        def acceptance_fn(local, partner) -> float:
            return self.native_rex.ph_lambda_probability(
                np.asarray(local.lambdas, dtype=np.float64),
                np.asarray(partner.lambdas, dtype=np.float64),
                signs,
                self._state_ph(local),
                self._state_ph(partner),
            )

        exchange_result = self.native_rex.attempt_neighbor_swap(
            self.exchanger,
            state.rex_attempt_idx,
            local_state,
            acceptance_fn,
        )
        next_label = self._next_label(current_label, exchange_result)
        post_state = self.native_rex.snapshot_state(
            self.exchanger,
            label=next_label,
            ldin_blocks=self._direct_ldin_blocks(),
            msld_theta_blocks=self._msld_theta_blocks(),
            include_lambdas=True,
            include_ph=True,
            include_ffix=True,
            metadata=self._state_metadata(
                next_label,
                float(self.ctx.replica_ph_values[next_label]),
            ),
        )
        self._debug_state("post", state, post_state, exchange_result=exchange_result)
        decisions = tuple(getattr(exchange_result, "decisions", ()) or ())
        self._record_first_decisions(state, decisions)
        stats = ExchangeStats(
            attempted=self._stats_or_empty(current_state.rex_attempted),
            accepted=self._stats_or_empty(current_state.rex_accepted),
        ).record_decisions(decisions)
        return REXStepResult(
            replica_label=next_label,
            attempted=stats.attempted,
            accepted=stats.accepted,
            exchange_result=exchange_result,
        )

    def _sign_vector_for_state(self, state) -> np.ndarray:
        signs = np.asarray(self.ctx.rex_signs, dtype=np.float64)
        lambdas = np.asarray(state.lambdas, dtype=np.float64)
        if lambdas.ndim != 1:
            raise ValueError(f"direct exchange lambdas must be 1D, got shape {lambdas.shape}")
        if signs.shape == lambdas.shape:
            return signs
        if signs.shape[0] == lambdas.shape[0] - 1:
            return np.concatenate((np.array([0.0], dtype=np.float64), signs))
        raise ValueError(
            "rex_signs must match the direct lambda vector or exclude only the "
            f"environment block ({signs.shape} vs {lambdas.shape})"
        )

    def _state_ph(self, state) -> float:
        metadata = getattr(state, "metadata", None) or {}
        ph = metadata.get("ph")
        if ph is not None:
            return float(ph)
        ph = getattr(state, "ph", None)
        if ph is not None:
            return float(ph)
        label = getattr(state, "label", None)
        if label is not None and 0 <= int(label) < len(self.ctx.replica_ph_values):
            return float(self.ctx.replica_ph_values[int(label)])
        raise ValueError("replica exchange state is missing pH metadata")

    def _load_native_rex(self) -> Any:
        return importlib.import_module("cphmd.native.rex")

    def _current_label(self, state: LoopState) -> int:
        if state.replica_label is None:
            return self.ctx.replica_label
        return state.replica_label

    def _direct_ldin_blocks(self) -> tuple[int, ...]:
        blocks = tuple(self.ctx.ldin_blocks or ())
        if not blocks or blocks[0] == 1:
            return blocks
        return (1, *blocks)

    def _msld_theta_blocks(self) -> tuple[int, ...]:
        if not self.ctx.dynamics_backend.uses_blade:
            return ()
        return tuple(self.ctx.ldin_blocks or ())

    def _stats_or_empty(self, values: tuple[int, ...]) -> tuple[int, ...]:
        n_pairs = len(self.ctx.replica_ph_values) - 1
        if not values:
            return tuple(0 for _ in range(n_pairs))
        return tuple(values)

    def _state_metadata(self, label: int, ph: float) -> dict[str, float | int]:
        return {
            "replica_label": int(label),
            "ph": float(ph),
            "temperature": float(self.ctx.temperature),
        }

    def _next_label(self, current_label: int, exchange_result: Any) -> int:
        next_label = getattr(exchange_result, "replica_label", None)
        if next_label is not None:
            return int(next_label)

        partner_state = getattr(exchange_result, "partner_state", None)
        if partner_state is not None and getattr(exchange_result, "accepted", False):
            partner_label = getattr(partner_state, "label", None)
            if partner_label is not None:
                return int(partner_label)

        return current_label

    def _record_first_decisions(self, state: LoopState, decisions: tuple[dict, ...]) -> None:
        if self.ctx.rank != 0 or not decisions:
            return

        path = self.ctx.run_dir / "rex_first_five_swaps.json"
        try:
            recorded = json.loads(path.read_text()) if path.exists() else []
        except (OSError, json.JSONDecodeError):
            recorded = []

        if len(recorded) >= 5:
            return

        for decision in decisions:
            if len(recorded) >= 5:
                break
            ranks = decision.get("ranks") or decision.get("pair") or ()
            recorded.append(
                {
                    "segment": int(state.segment_idx),
                    "attempt": int(state.rex_attempt_idx),
                    "pair": [int(rank) for rank in ranks],
                    "accepted": bool(decision.get("accepted", decision.get("accept", False))),
                    "score": _optional_float(
                        decision.get(
                            "delta",
                            decision.get(
                                "log_probability",
                                decision.get("probability"),
                            ),
                        )
                    ),
                }
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)

    def _debug_state(
        self,
        phase: str,
        state: LoopState,
        rex_state: Any,
        *,
        exchange_result=None,
    ) -> None:
        if os.environ.get("CPHMD_DEBUG_REX_STATE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        payload = {
            "phase": phase,
            "rank": self.ctx.rank,
            "attempt": state.rex_attempt_idx,
            "segment": state.segment_idx,
            "state": _state_payload(rex_state),
        }
        if exchange_result is not None:
            payload["exchange"] = {
                "accepted": bool(getattr(exchange_result, "accepted", False)),
                "partner_rank": _optional_int(getattr(exchange_result, "partner_rank", None)),
                "replica_label": _optional_int(getattr(exchange_result, "replica_label", None)),
                "decisions": list(getattr(exchange_result, "decisions", ()) or ()),
                "partner_state": _state_payload(getattr(exchange_result, "partner_state", None)),
            }
        path = self.ctx.rank_dir / f"rex_debug_attempt_{state.rex_attempt_idx:06d}_{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _with_canonical_ph(state: Any, ph: float) -> Any:
    if is_dataclass(state):
        return replace(state, ph=float(ph))
    try:
        state.ph = float(ph)
    except Exception:
        pass
    return state


def _state_payload(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    values = asdict(state) if is_dataclass(state) else state if isinstance(state, dict) else None
    return {
        "type": f"{type(state).__module__}.{type(state).__qualname__}",
        "dataclass": bool(is_dataclass(state)),
        "label": _optional_int(_value(state, values, "label")),
        "ph": _optional_float(_value(state, values, "ph")),
        "temperature": _optional_float(_value(state, values, "temperature")),
        "metadata": dict(_value(state, values, "metadata") or {}),
        "lambdas": _float_list(_value(state, values, "lambdas")),
        "ldin": [_object_payload(item) for item in (_value(state, values, "ldin") or ())],
        "msld_theta": [
            _object_payload(item) for item in (_value(state, values, "msld_theta") or ())
        ],
    }


def _object_payload(item: Any) -> dict[str, Any]:
    values = asdict(item) if is_dataclass(item) else item if isinstance(item, dict) else None
    payload = {
        "type": f"{type(item).__module__}.{type(item).__qualname__}",
        "dataclass": bool(is_dataclass(item)),
    }
    for name in ("block_id", "lambda_sq", "velocity", "mass", "bias", "friction", "theta"):
        value = _value(item, values, name)
        if value is not None:
            payload[name] = _optional_int(value) if name == "block_id" else _optional_float(value)
    return payload


def _value(obj: Any, values: Any, name: str) -> Any:
    if isinstance(values, dict):
        return values.get(name)
    return getattr(obj, name, None)


def _float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]
