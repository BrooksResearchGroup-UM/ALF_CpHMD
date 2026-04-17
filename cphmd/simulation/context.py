from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

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
    simulation_name: str = "cphmd-native"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "nsubsites", tuple(self.nsubsites))
        if self.rex_enabled:
            raise ValueError("REX is Phase 2 scope; Phase 1 supports single-replica only")
        if self.checkpoint_every_segments <= 0:
            raise ValueError("checkpoint_every_segments must be positive")
        if len(self.nsubsites) != len(self.lambda_headers) + 1:
            raise ValueError("nsubsites must include environment plus lambda columns")
        if self.nsubsites[0] != 0:
            raise ValueError("nsubsites must start with the environment marker 0")

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

    def advance_segment(self) -> "LoopState":
        return replace(
            self,
            segment_idx=self.segment_idx + 1,
            run_idx=self.run_idx + 1,
        )


class LoopHooks(Protocol):
    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> None: ...

    def is_done(self, state: LoopState) -> bool: ...
