from __future__ import annotations

import importlib
from dataclasses import dataclass
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
            ldin_blocks=self._ldin_blocks(),
            include_lambdas=True,
            include_ph=True,
            metadata={"ph": local_ph},
        )
        signs = np.asarray(self.ctx.rex_signs, dtype=np.float64)

        def acceptance_fn(local, partner) -> float:
            return self.native_rex.ph_lambda_probability(
                np.asarray(local.lambdas, dtype=np.float64),
                np.asarray(partner.lambdas, dtype=np.float64),
                signs,
                float(local.ph),
                float(partner.ph),
            )

        exchange_result = self.native_rex.attempt_neighbor_swap(
            self.exchanger,
            state.rex_attempt_idx,
            local_state,
            acceptance_fn,
            apply_state=True,
        )
        decisions = tuple(getattr(exchange_result, "decisions", ()) or ())
        stats = ExchangeStats(
            attempted=self._stats_or_empty(current_state.rex_attempted),
            accepted=self._stats_or_empty(current_state.rex_accepted),
        ).record_decisions(decisions)
        next_label = self._next_label(current_label, exchange_result)
        return REXStepResult(
            replica_label=next_label,
            attempted=stats.attempted,
            accepted=stats.accepted,
            exchange_result=exchange_result,
        )

    def _load_native_rex(self) -> Any:
        return importlib.import_module("cphmd.native.rex")

    def _current_label(self, state: LoopState) -> int:
        if state.replica_label is None:
            return self.ctx.replica_label
        return state.replica_label

    def _ldin_blocks(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.ctx.lambda_headers) + 1))

    def _stats_or_empty(self, values: tuple[int, ...]) -> tuple[int, ...]:
        n_pairs = len(self.ctx.replica_ph_values) - 1
        if not values:
            return tuple(0 for _ in range(n_pairs))
        return tuple(values)

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
