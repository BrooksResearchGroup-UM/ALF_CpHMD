from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.simulation.shrinker import LambdaPrecision
from cphmd.training.bias_snapshot import BiasSnapshot, LDBVTerm
from cphmd.utils.seeds import derive_seed

logger = logging.getLogger(__name__)


class ProductionConfigError(RuntimeError):
    """Raised when fixed-bias production inputs cannot be applied safely."""


@dataclass(frozen=True)
class IntrinsicBias:
    block_id: int
    bias: float


@dataclass(frozen=True)
class ProductionBiasSpec:
    schema_version: int
    nsubs: tuple[int, ...]
    lambda_headers: tuple[str, ...] | None
    fnex: float | None
    metadata_present: bool
    intrinsic: tuple[IntrinsicBias, ...]
    ldbv_terms: tuple[LDBVTerm, ...]

    @property
    def intrinsic_biases(self) -> tuple[tuple[int, float], ...]:
        return tuple((item.block_id, item.bias) for item in self.intrinsic)


@dataclass(frozen=True)
class ProductionConfig:
    bias_file: Path
    n_chunks: int
    topology_files: list[Path] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)
    vdw_type: str = "CHARMM"
    segments_per_chunk: int = 10
    ldin_mass: float = 10.0
    ldin_friction: float = 10.0
    fnex: float = 5.5
    temperature: float = 298.15
    lambda_precision: LambdaPrecision | str = LambdaPrecision.SHRINKER

    def __post_init__(self) -> None:
        bias_file = Path(self.bias_file)
        if not bias_file.is_absolute():
            raise ValueError(f"ProductionConfig.bias_file must be absolute: {bias_file}")
        if self.n_chunks <= 0:
            raise ValueError("n_chunks must be positive")
        if self.segments_per_chunk <= 0:
            raise ValueError("segments_per_chunk must be positive")
        precision = (
            self.lambda_precision
            if isinstance(self.lambda_precision, LambdaPrecision)
            else LambdaPrecision(str(self.lambda_precision).lower())
        )
        if precision is LambdaPrecision.FULL:
            logger.warning(
                "ProductionConfig.lambda_precision=FULL selected; production archives "
                "default to SHRINKER to reduce storage."
            )
        vdw_type = str(self.vdw_type).upper()
        if vdw_type not in {"CHARMM", "OPLS"}:
            raise ValueError(f"ProductionConfig.vdw_type must be 'CHARMM' or 'OPLS': {vdw_type!r}")
        object.__setattr__(self, "bias_file", bias_file)
        object.__setattr__(self, "lambda_precision", precision)
        object.__setattr__(self, "vdw_type", vdw_type)
        object.__setattr__(self, "topology_files", [Path(path) for path in self.topology_files])
        object.__setattr__(self, "extra_files", [Path(path) for path in self.extra_files])


def load_production_biases(
    cfg: ProductionConfig,
    *,
    nsubs: tuple[int, ...] | list[int] | None = None,
) -> ProductionBiasSpec:
    if not cfg.bias_file.exists():
        raise FileNotFoundError(f"ProductionConfig.bias_file does not exist: {cfg.bias_file}")

    with np.load(cfg.bias_file, allow_pickle=False) as data:
        required = {"schema_version", "b", "c", "x", "s"}
        missing = required - set(data.files)
        if missing:
            raise ProductionConfigError(
                f"{cfg.bias_file} missing required keys {sorted(missing)}; "
                f"found {sorted(data.files)}"
            )
        schema_version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        if schema_version != 1:
            raise ProductionConfigError(f"unsupported checkpoint_bias.npz schema {schema_version}")

        b = np.asarray(data["b"], dtype=np.float64)
        nblocks = int(b.reshape(-1).shape[0])
        metadata_present = "bias_nsubs" in data.files and "lambda_headers" in data.files
        if "bias_nsubs" in data.files:
            nsubs_tuple = tuple(int(value) for value in np.asarray(data["bias_nsubs"]).reshape(-1))
        else:
            nsubs_tuple = tuple(int(value) for value in (nsubs or (nblocks,)))
        if "lambda_headers" in data.files:
            lambda_headers = tuple(str(value) for value in np.asarray(data["lambda_headers"]))
        else:
            lambda_headers = None
        if "fnex" in data.files:
            fnex = float(np.asarray(data["fnex"]).reshape(-1)[0])
        else:
            fnex = None
        snapshot = BiasSnapshot.from_arrays(
            b=b,
            c=np.asarray(data["c"], dtype=np.float64),
            x=np.asarray(data["x"], dtype=np.float64),
            s=np.asarray(data["s"], dtype=np.float64),
            nsubs=nsubs_tuple,
        )

    return ProductionBiasSpec(
        schema_version=schema_version,
        nsubs=nsubs_tuple,
        lambda_headers=lambda_headers,
        fnex=fnex,
        metadata_present=metadata_present,
        intrinsic=tuple(
            IntrinsicBias(block_id, bias) for block_id, bias in snapshot.intrinsic_biases
        ),
        ldbv_terms=snapshot.ldbv_terms,
    )


class ProductionHooks:
    def __init__(
        self,
        ctx: RunContext,
        cfg: ProductionConfig,
        *,
        native_block: Any | None = None,
    ):
        self.ctx = ctx
        self.cfg = cfg
        self.native_block = native_block
        self.bias_spec: ProductionBiasSpec | None = None

    def on_system_loaded(self, ctx: RunContext, state: LoopState | None = None) -> None:
        self._bootstrap_native_state(ctx, state=state, first_load=True)

    def before_segment(self, state: LoopState) -> LoopState:
        if state.segment_idx % self.cfg.segments_per_chunk != 0:
            return state
        return state.with_integrator_seed(self.chunk_seed(state.chunk_idx))

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> LoopState:
        completed_segments = state.segment_idx + 1
        if completed_segments % self.cfg.segments_per_chunk == 0:
            return state.advance_chunk()
        return state

    def should_trigger_cycle(self, state: LoopState) -> bool:
        return False

    def run_cycle(self, state: LoopState):
        raise NotImplementedError("ProductionHooks do not run ALF cycles")

    def after_rex_swap(
        self,
        state: LoopState,
        *,
        partner_rank: int | None,
        accepted: bool,
    ) -> None:
        if not accepted:
            return
        if not self.ctx.ph_enabled:
            return
        native = self._native_block()
        native.set_ph(self._ph_for_state(state))
        native.sync_state()

    def is_done(self, state: LoopState) -> bool:
        return state.chunk_idx >= self.cfg.n_chunks

    def chunk_seed(self, chunk_idx: int) -> int:
        return derive_seed(self.ctx.master_seed, "production_chunk", self.ctx.rank, chunk_idx)

    def _bootstrap_native_state(
        self,
        ctx: RunContext,
        *,
        state: LoopState | None,
        first_load: bool,
    ) -> None:
        if not first_load:
            native = self._native_block()
            if ctx.ph_enabled:
                current_state = state or LoopState(replica_label=ctx.replica_label)
                native.set_ph(self._ph_for_state(current_state))
            native.sync_state()
            return

        spec = load_production_biases(self.cfg, nsubs=_nsubs_from_context(ctx))
        self._validate_bias_matches_topology(ctx, spec)
        native = self._native_block()
        live_fnex = native.get_fnex()
        if not np.isclose(live_fnex, self.cfg.fnex):
            raise ProductionConfigError(
                f"Live FNEX={live_fnex} does not match configured FNEX={self.cfg.fnex}"
            )
        if spec.fnex is not None and not np.isclose(spec.fnex, self.cfg.fnex):
            raise ProductionConfigError(
                f"Bias snapshot FNEX={spec.fnex} does not match configured FNEX={self.cfg.fnex}"
            )

        if ctx.ph_enabled:
            native.set_ph(self._ph_for_state(state or LoopState(replica_label=ctx.replica_label)))
        native.sync_state()
        modify_biases = getattr(native, "modify_biases", None)
        with (modify_biases() if modify_biases is not None else nullcontext()):
            native.clear_biases()
            for intrinsic in spec.intrinsic:
                native.set_intrinsic_bias(intrinsic.block_id, intrinsic.bias)
            set_bias_count = getattr(native, "set_bias_count", None)
            if set_bias_count is not None:
                set_bias_count(len(spec.ldbv_terms))
            for index, term in enumerate(spec.ldbv_terms, start=1):
                native.add_bias(*term.as_tuple(), index=index)
        native.sync_state()
        self.bias_spec = spec

    def _validate_bias_matches_topology(
        self,
        ctx: RunContext,
        spec: ProductionBiasSpec,
    ) -> None:
        if not ctx.titratable_blocks:
            raise ProductionConfigError("titratable_blocks is empty for production run")
        if not spec.metadata_present or spec.lambda_headers is None:
            raise ProductionConfigError(
                "Bias snapshot metadata is required for production; regenerate checkpoint_bias.npz "
                "with bias_nsubs and lambda_headers"
            )
        expected_nsubs = _nsubs_from_context(ctx)
        if spec.nsubs != expected_nsubs:
            raise ProductionConfigError(
                f"Bias snapshot nsubs mismatch: expected {expected_nsubs}, found {spec.nsubs}"
            )
        if spec.lambda_headers != ctx.lambda_headers:
            raise ProductionConfigError(
                "Bias snapshot lambda_headers mismatch: "
                f"expected {ctx.lambda_headers}, found {spec.lambda_headers}"
            )
        bias_ids = {item.block_id for item in spec.intrinsic}
        topology_ids = {item.block_id for item in ctx.titratable_blocks}
        if bias_ids != topology_ids:
            raise ProductionConfigError(
                "Bias snapshot / topology mismatch: "
                f"bias block ids={sorted(bias_ids)}, topology block ids={sorted(topology_ids)}"
            )

    def _ph_for_state(self, state: LoopState) -> float:
        label = self.ctx.replica_label if state.replica_label is None else state.replica_label
        if self.ctx.replica_ph_values and 0 <= label < len(self.ctx.replica_ph_values):
            return float(self.ctx.replica_ph_values[label])
        return float(self.ctx.ph)

    def _native_block(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block


def _nsubs_from_context(ctx: RunContext) -> tuple[int, ...]:
    """Return per-site lambda block counts from a validated run context."""
    if ctx.nsites <= 0:
        raise ProductionConfigError("production context has no titratable sites")
    if len(ctx.nsubsites) != len(ctx.lambda_headers) + 1:
        raise ProductionConfigError("nsubsites must include one value per lambda header plus env")
    if ctx.nsubsites[0] != 0:
        raise ProductionConfigError("nsubsites must start with environment marker 0")
    counts: list[int] = []
    for site in range(1, ctx.nsites + 1):
        counts.append(sum(1 for value in ctx.nsubsites[1:] if value == site))
    if any(count <= 0 for count in counts):
        raise ProductionConfigError(f"nsubsites do not cover all sites 1..{ctx.nsites}")
    return tuple(counts)


FixedBiasHooks = ProductionHooks


__all__ = [
    "FixedBiasHooks",
    "IntrinsicBias",
    "LDBVTerm",
    "ProductionBiasSpec",
    "ProductionConfig",
    "ProductionConfigError",
    "ProductionHooks",
    "load_production_biases",
]
