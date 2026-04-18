from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cphmd.simulation.shrinker import LambdaPrecision


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    rank: int
    replica_label: int
    ph: float
    gpu_id: int
    lambda_headers: list[str]
    nsubsites: tuple[int, ...]
    nsites: int
    nsteps_per_segment: int
    nsavl: int
    nsavc: int
    time_step_ps: float
    temperature: float
    checkpoint_every_segments: int
    lambda_precision: LambdaPrecision
    master_seed: int
    config_hash: str
    rex_enabled: bool = False
    replica_ph_values: tuple[float, ...] = ()
    rex_signs: tuple[float, ...] = ()
    rex_exchange_every_segments: int = 1
    comm: Any | None = None
    simulation_name: str = "cphmd-native"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "nsubsites", tuple(self.nsubsites))
        object.__setattr__(
            self,
            "replica_ph_values",
            tuple(self.replica_ph_values),
        )
        object.__setattr__(self, "rex_signs", tuple(self.rex_signs))
        if self.checkpoint_every_segments <= 0:
            raise ValueError("checkpoint_every_segments must be positive")
        if len(self.nsubsites) != len(self.lambda_headers) + 1:
            raise ValueError("nsubsites must include environment plus lambda columns")
        if self.nsubsites[0] != 0:
            raise ValueError("nsubsites must start with the environment marker 0")
        if self.rex_enabled:
            if len(self.replica_ph_values) < 2:
                raise ValueError("replica_ph_values must include at least two pH values")
            if len(self.rex_signs) != len(self.lambda_headers):
                raise ValueError("rex_signs must match lambda_headers")
            if self.rex_exchange_every_segments <= 0:
                raise ValueError("rex_exchange_every_segments must be positive")
            if not 0 <= self.replica_label < len(self.replica_ph_values):
                raise ValueError("replica_label must be within replica_ph_values")

    @property
    def rank_dir(self) -> Path:
        return self.run_dir / "res" / f"rep{self.rank:02d}"

    @property
    def checkpoint_path(self) -> Path:
        return self.rank_dir / "checkpoint.json"

    def segment_path(self, segment_idx: int) -> Path:
        return self.rank_dir / f"segment_{segment_idx:06d}.parquet"


@dataclass(frozen=True)
class LoopState:
    segment_idx: int = 0
    chunk_idx: int = 0
    run_idx: int = 0
    stop_requested: bool = False
    replica_label: int | None = None
    rex_attempt_idx: int = 0
    rex_attempted: tuple[int, ...] = ()
    rex_accepted: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rex_attempted", tuple(self.rex_attempted))
        object.__setattr__(self, "rex_accepted", tuple(self.rex_accepted))

    def advance_segment(self) -> "LoopState":
        return replace(
            self,
            segment_idx=self.segment_idx + 1,
            run_idx=self.run_idx + 1,
        )

    def with_initial_label(self, replica_label: int) -> "LoopState":
        return replace(self, replica_label=replica_label)

    def with_rex_result(
        self,
        replica_label: int,
        attempted: tuple[int, ...],
        accepted: tuple[int, ...],
    ) -> "LoopState":
        return replace(
            self,
            replica_label=replica_label,
            rex_attempt_idx=self.rex_attempt_idx + 1,
            rex_attempted=tuple(attempted),
            rex_accepted=tuple(accepted),
        )


class LoopHooks(Protocol):
    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> None: ...

    def is_done(self, state: LoopState) -> bool: ...
