from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cphmd.simulation.backends import (
    AnalysisBackend,
    DomdecConfig,
    DynamicsBackend,
    parse_analysis_backend,
    parse_dynamics_backend,
)
from cphmd.simulation.shrinker import LambdaPrecision


@dataclass(frozen=True)
class TitratableBlock:
    block_id: int
    segid: str
    resid: str
    resname: str
    site: int

    @property
    def lambda_header(self) -> str:
        return f"{self.segid} {self.resid} {self.resname}"


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    rank: int
    replica_label: int
    ph: float
    gpu_id: int | None
    lambda_headers: tuple[str, ...]
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
    ph_enabled: bool = False
    ldin_blocks: tuple[int, ...] | None = None
    rex_enabled: bool = False
    replica_ph_values: tuple[float, ...] = ()
    rex_signs: tuple[float, ...] = ()
    rex_exchange_every_segments: int = 1
    comm: Any | None = None
    simulation_name: str = "cphmd-native"
    dynamics_backend: DynamicsBackend | str = DynamicsBackend.BLADE
    analysis_backend: AnalysisBackend | str = AnalysisBackend.CUDA_WHAM
    domdec: DomdecConfig = field(default_factory=DomdecConfig)
    use_blade: bool | None = None
    walltime_end_epoch: float | None = None
    walltime_safety_factor: float = 2.0
    titratable_blocks: tuple[TitratableBlock, ...] = ()
    startup_minimization_segments: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        backend = (
            DynamicsBackend.BLADE
            if self.use_blade is True
            else DynamicsBackend.DOMDEC_CPU
            if self.use_blade is False
            else parse_dynamics_backend(self.dynamics_backend)
        )
        object.__setattr__(self, "dynamics_backend", backend)
        object.__setattr__(self, "analysis_backend", parse_analysis_backend(self.analysis_backend))
        object.__setattr__(self, "use_blade", backend.uses_blade)
        if not isinstance(self.domdec, DomdecConfig):
            object.__setattr__(self, "domdec", DomdecConfig.from_mapping(dict(self.domdec)))
        object.__setattr__(self, "lambda_headers", tuple(self.lambda_headers))
        if self.ldin_blocks is None:
            object.__setattr__(
                self,
                "ldin_blocks",
                tuple(range(2, len(self.lambda_headers) + 2)),
            )
        else:
            object.__setattr__(
                self,
                "ldin_blocks",
                tuple(int(value) for value in self.ldin_blocks),
            )
        object.__setattr__(self, "nsubsites", tuple(self.nsubsites))
        object.__setattr__(
            self,
            "replica_ph_values",
            tuple(self.replica_ph_values),
        )
        object.__setattr__(self, "rex_signs", tuple(self.rex_signs))
        object.__setattr__(self, "titratable_blocks", tuple(self.titratable_blocks))
        if self.checkpoint_every_segments <= 0:
            raise ValueError("checkpoint_every_segments must be positive")
        if self.walltime_safety_factor <= 0:
            raise ValueError("walltime_safety_factor must be positive")
        if self.startup_minimization_segments < 0:
            raise ValueError("startup_minimization_segments must be non-negative")
        if self.dynamics_backend.requires_gpu and self.gpu_id is None:
            raise ValueError(f"{self.dynamics_backend.value} requires gpu_id")
        if len(self.nsubsites) != len(self.lambda_headers) + 1:
            raise ValueError("nsubsites must include environment plus lambda columns")
        if self.nsubsites[0] != 0:
            raise ValueError("nsubsites must start with the environment marker 0")
        if len(self.ldin_blocks) != len(self.lambda_headers):
            raise ValueError("ldin_blocks must match lambda_headers")
        if any(block_id <= 0 for block_id in self.ldin_blocks):
            raise ValueError("ldin_blocks must contain positive block IDs")
        if self.rex_enabled:
            if not self.ph_enabled:
                raise ValueError("replica exchange requires pH to be enabled")
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
    cycle_idx: int = 0
    phase: int = 1
    stop_requested: bool = False
    replica_label: int | None = None
    rex_attempt_idx: int = 0
    rex_attempted: tuple[int, ...] = ()
    rex_accepted: tuple[int, ...] = ()
    integrator_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rex_attempted", tuple(self.rex_attempted))
        object.__setattr__(self, "rex_accepted", tuple(self.rex_accepted))

    def advance_segment(self) -> "LoopState":
        return replace(
            self,
            segment_idx=self.segment_idx + 1,
            run_idx=self.run_idx + 1,
        )

    def advance_chunk(self) -> "LoopState":
        return replace(self, chunk_idx=self.chunk_idx + 1, integrator_seed=None)

    def with_integrator_seed(self, integrator_seed: int | None) -> "LoopState":
        return replace(self, integrator_seed=integrator_seed)

    def with_initial_label(self, replica_label: int) -> "LoopState":
        if self.replica_label is None:
            return replace(self, replica_label=replica_label)
        if self.replica_label == replica_label:
            return self
        raise ValueError(
            f"replica_label mismatch: expected {self.replica_label}, got {replica_label}"
        )

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

    def with_cycle_result(self, *, phase: int | None = None) -> "LoopState":
        return replace(
            self,
            cycle_idx=self.cycle_idx + 1,
            phase=self.phase if phase is None else phase,
        )

    def with_stop_requested(self) -> "LoopState":
        return replace(self, stop_requested=True)


class LoopHooks(Protocol):
    def on_system_loaded(self, ctx: RunContext, state: LoopState | None = None) -> None: ...

    def before_segment(self, state: LoopState) -> LoopState | None: ...

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> LoopState | None: ...

    def after_rex_swap(
        self,
        state: LoopState,
        *,
        partner_rank: int | None,
        accepted: bool,
    ) -> None: ...

    def is_done(self, state: LoopState) -> bool: ...


class CycleLoopHooks(LoopHooks, Protocol):
    def should_trigger_cycle(self, state: LoopState) -> bool: ...

    def run_cycle(self, state: LoopState): ...
