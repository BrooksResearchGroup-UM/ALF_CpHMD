"""Deprecated pH replica-exchange compatibility shim."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

__all__: tuple[str, ...] = ()


def _warn(name: str, replacement: str) -> None:
    warnings.warn(
        f"cphmd.core.replica_exchange.{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass
class _ReplicaExchangeConfig:
    """Compatibility configuration for legacy callers."""

    enabled: bool = False
    exchange_freq: int = 1000
    backend: str = "native"

    def __post_init__(self) -> None:
        self.exchange_freq = int(self.exchange_freq)
        if self.exchange_freq <= 0:
            raise ValueError("exchange_freq must be positive")
        self.backend = str(self.backend)

    @property
    def exchange_every_segments(self) -> int:
        return self.exchange_freq

    def to_native(self) -> Any:
        from cphmd.simulation.rex_driver import REXConfig

        return REXConfig(
            enabled=self.enabled,
            exchange_every_segments=self.exchange_freq,
        )


@dataclass
class _ExchangeState:
    """Legacy exchange bookkeeping state."""

    attempted: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    total_attempted: int = 0
    total_accepted: int = 0
    permutation: list[int] = field(default_factory=list)

    def ensure_size(self, npairs: int) -> None:
        while len(self.attempted) < npairs:
            self.attempted.append(0)
        while len(self.accepted) < npairs:
            self.accepted.append(0)
        if not self.permutation:
            self.permutation = list(range(npairs + 1))

    def record(self, pair_idx: int, accepted: bool) -> None:
        self.attempted[pair_idx] += 1
        self.total_attempted += 1
        if accepted:
            self.accepted[pair_idx] += 1
            self.total_accepted += 1

    @property
    def acceptance_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_accepted / self.total_attempted

    def pair_acceptance_rates(self) -> list[float]:
        return [a / t if t > 0 else 0.0 for a, t in zip(self.accepted, self.attempted)]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "attempted": self.attempted,
                    "accepted": self.accepted,
                    "total_attempted": self.total_attempted,
                    "total_accepted": self.total_accepted,
                    "permutation": self.permutation,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "_ExchangeState":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            attempted=list(data.get("attempted", [])),
            accepted=list(data.get("accepted", [])),
            total_attempted=int(data.get("total_attempted", 0)),
            total_accepted=int(data.get("total_accepted", 0)),
            permutation=list(data.get("permutation", [])),
        )


class _ReplicaExchanger:
    """Deprecated adapter with import-safe construction.

    The legacy pH-replica-exchange orchestration has been replaced by
    ``cphmd.simulation.rex_driver.REXDriver``. This class keeps the old
    object shape alive long enough for deprecation callers to fail at a
    controlled point instead of import time.
    """

    def __init__(self, config: _ReplicaExchangeConfig, comm, rank: int, size: int):
        if isinstance(config, _ReplicaExchangeConfig):
            self.config = config
        else:
            self.config = _ReplicaExchangeConfig(
                enabled=bool(getattr(config, "enabled", False)),
                exchange_freq=int(
                    getattr(
                        config,
                        "exchange_freq",
                        getattr(config, "exchange_every_segments", 1000),
                    )
                ),
            )
        self.comm = comm
        self.rank = rank
        self.size = size
        self.state = _ExchangeState()

    def compute_n_segments(self, nsteps_prod: int) -> int:
        return nsteps_prod // self.config.exchange_freq

    def attempt_exchange(self, *args, **kwargs):
        raise RuntimeError(
            "ReplicaExchanger is deprecated; use cphmd.simulation.rex_driver.REXDriver "
            "with cphmd.simulation.loop.SimulationLoop."
        )

    def write_exchange_log(self, path: Path, run_idx: int) -> None:
        rates = self.state.pair_acceptance_rates()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"# Replica Exchange Statistics — Run {run_idx}\n")
            handle.write(
                f"# Overall: {self.state.total_accepted}/{self.state.total_attempted} "
                f"({self.state.acceptance_rate:.1%})\n"
            )
            handle.write("#\n# Pair  Attempted  Accepted  Rate\n")
            for i, (att, acc, rate) in enumerate(
                zip(self.state.attempted, self.state.accepted, rates)
            ):
                handle.write(f"  {i}↔{i+1}  {att:9d}  {acc:8d}  {rate:.3f}\n")
            if self.state.permutation:
                handle.write(f"# Current permutation (config→rank): {self.state.permutation}\n")

    def load_state(self, path: Path) -> None:
        if path.exists():
            self.state = _ExchangeState.load(path)

    def save_state(self, path: Path) -> None:
        self.state.save(path)


def _neighbor_partner(rank: int, size: int, attempt_idx: int) -> int | None:
    from cphmd.native.rex import neighbor_partner as native_neighbor_partner

    return native_neighbor_partner(rank, size, attempt_idx)


_EXPORTS: dict[str, tuple[str, str, str]] = {
    "ReplicaExchangeConfig": (
        "cphmd.core.replica_exchange",
        "_ReplicaExchangeConfig",
        "cphmd.simulation.rex_driver.REXConfig",
    ),
    "ExchangeState": (
        "cphmd.core.replica_exchange",
        "_ExchangeState",
        "cphmd.simulation.rex_driver.ExchangeStats",
    ),
    "ReplicaExchanger": (
        "cphmd.core.replica_exchange",
        "_ReplicaExchanger",
        "cphmd.simulation.rex_driver.REXDriver",
    ),
    "neighbor_partner": (
        "cphmd.core.replica_exchange",
        "_neighbor_partner",
        "cphmd.native.rex.neighbor_partner",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name, replacement = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    _warn(name, replacement)
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
