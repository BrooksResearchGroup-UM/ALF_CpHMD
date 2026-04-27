from __future__ import annotations

import csv
import json
import logging
import math
import os
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
_MIN_PLAUSIBLE_FNEX = 0.0
_MAX_PLAUSIBLE_FNEX = 100.0


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
class _PHBiasTag:
    tag_type: str
    pka: float | None


@dataclass(frozen=True)
class ProductionConfig:
    n_chunks: int
    bias_file: Path | None = None
    use_presets: bool = True
    preset_config: str | None = None
    topology_files: list[Path] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)
    segments_per_chunk: int = 10
    ldin_mass: float = 10.0
    ldin_friction: float = 10.0
    fnex: float = 5.5
    temperature: float = 298.15
    lambda_precision: LambdaPrecision | str = LambdaPrecision.SHRINKER

    def __post_init__(self) -> None:
        bias_file = None if self.bias_file is None else Path(self.bias_file)
        if bias_file is not None and not bias_file.is_absolute():
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
        object.__setattr__(self, "bias_file", bias_file)
        object.__setattr__(self, "use_presets", bool(self.use_presets))
        object.__setattr__(
            self,
            "preset_config",
            None if self.preset_config is None else str(self.preset_config),
        )
        object.__setattr__(self, "lambda_precision", precision)
        object.__setattr__(self, "topology_files", [Path(path) for path in self.topology_files])
        object.__setattr__(self, "extra_files", [Path(path) for path in self.extra_files])


def load_production_biases(
    cfg: ProductionConfig,
    *,
    nsubs: tuple[int, ...] | list[int] | None = None,
) -> ProductionBiasSpec:
    if cfg.bias_file is None:
        raise ProductionConfigError("production.bias_file is required for fixed-bias production")
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


def load_production_preset_biases(
    cfg: ProductionConfig,
    ctx: RunContext,
) -> ProductionBiasSpec:
    """Build a production bias spec from bundled residue presets."""
    from cphmd.core.alf_utils import ALFInfo, _load_preset_biases
    from cphmd.presets import list_presets

    nsubs = _nsubs_from_context(ctx)
    site_residue_types = _site_residue_types_from_context(ctx)
    available = set(list_presets(cfg.preset_config))
    missing = sorted({residue for residue in site_residue_types if residue not in available})
    if missing:
        preset_name = cfg.preset_config or "default"
        raise ProductionConfigError(
            f"Preset config {preset_name!r} has no biases for residues {missing}"
        )

    alf_info = ALFInfo(
        name=site_residue_types[0] if len(set(site_residue_types)) == 1 else ctx.simulation_name,
        nblocks=sum(nsubs),
        nsubs=np.asarray(nsubs, dtype=np.int32),
        nreps=max(1, len(ctx.replica_ph_values)),
        ncentral=max(0, max(1, len(ctx.replica_ph_values)) // 2),
        temp=cfg.temperature,
        fnex=cfg.fnex,
    )
    b, c, x, s = _load_preset_biases(
        alf_info,
        preset_config=cfg.preset_config,
        site_residue_types=list(site_residue_types),
    )
    snapshot = BiasSnapshot.from_arrays(b=b, c=c, x=x, s=s, nsubs=nsubs)
    return ProductionBiasSpec(
        schema_version=1,
        nsubs=nsubs,
        lambda_headers=ctx.lambda_headers,
        fnex=cfg.fnex,
        metadata_present=True,
        intrinsic=tuple(
            IntrinsicBias(block_id, bias) for block_id, bias in snapshot.intrinsic_biases
        ),
        ldbv_terms=snapshot.ldbv_terms,
    )


def write_production_bias_file(
    path: str | Path,
    snapshot: BiasSnapshot,
    *,
    lambda_headers: tuple[str, ...] | list[str],
    fnex: float,
) -> Path:
    """Write a fixed-bias checkpoint consumable by production runs."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.array([1], dtype=np.int32),
        bias_nsubs=np.asarray(snapshot.nsubs, dtype=np.int32),
        lambda_headers=np.asarray(tuple(lambda_headers)),
        fnex=np.array([float(fnex)], dtype=np.float64),
        b=np.asarray(snapshot.b, dtype=np.float64),
        c=np.asarray(snapshot.c, dtype=np.float64),
        x=np.asarray(snapshot.x, dtype=np.float64),
        s=np.asarray(snapshot.s, dtype=np.float64),
    )
    return out_path


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
        self._ph_bias_tags: dict[int, _PHBiasTag] | None = None
        self._validated_runtime = False

    def on_system_loaded(self, ctx: RunContext, state: LoopState | None = None) -> None:
        self._bootstrap_native_state(ctx, state=state, phase="system_loaded")

    def on_native_ready(self, ctx: RunContext, state: LoopState | None = None) -> None:
        self._bootstrap_native_state(ctx, state=state, phase="native_ready")

    def before_segment(self, state: LoopState) -> LoopState:
        if state.segment_idx != 0:
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
        # The pyCHARMM exchanger already applies accepted pH/LDIN/FFIX state.
        if not accepted or not _production_bias_debug_enabled():
            return None
        spec = self._load_and_validate_biases(self.ctx)
        self._debug_bias_state(
            self.ctx,
            state,
            self._native_block(),
            spec,
            phase="rex_after_exchange",
        )
        return None

    def is_done(self, state: LoopState) -> bool:
        return state.chunk_idx >= self.cfg.n_chunks

    def chunk_seed(self, chunk_idx: int) -> int:
        return derive_seed(self.ctx.master_seed, "production_chunk", self.ctx.rank, chunk_idx)

    def _bootstrap_native_state(
        self,
        ctx: RunContext,
        *,
        state: LoopState | None,
        phase: str,
    ) -> None:
        spec = self._load_and_validate_biases(ctx)
        native = self._native_block()

        current_state = state or LoopState(replica_label=ctx.replica_label)
        if spec is None:
            native.sync_state()
            if ctx.ph_enabled:
                self._apply_ph_state(native, ctx, current_state, None)
            self._debug_bias_state(ctx, state, native, None, phase=f"{phase}_simple")
            return

        native.sync_state()
        self._debug_bias_state(ctx, state, native, spec, phase=f"{phase}_before_apply")
        if ctx.ph_enabled:
            native.set_ph(self._ph_for_state(current_state))
        modify_biases = getattr(native, "modify_biases", None)
        with (modify_biases() if modify_biases is not None else nullcontext()):
            native.clear_biases()
            for intrinsic in self._intrinsic_biases_for_state(ctx, current_state, spec):
                native.set_intrinsic_bias(intrinsic.block_id, intrinsic.bias)
            set_bias_count = getattr(native, "set_bias_count", None)
            if set_bias_count is not None:
                set_bias_count(len(spec.ldbv_terms))
            for index, term in enumerate(spec.ldbv_terms, start=1):
                native.add_bias(*term.as_tuple(), index=index)
        native.sync_state()
        self._debug_bias_state(ctx, state, native, spec, phase=f"{phase}_after_apply")

    def _apply_ph_state(
        self,
        native,
        ctx: RunContext,
        state: LoopState,
        spec: ProductionBiasSpec | None,
    ) -> None:
        native.set_ph(self._ph_for_state(state))
        if spec is not None:
            for intrinsic in self._intrinsic_biases_for_state(ctx, state, spec):
                native.set_intrinsic_bias(intrinsic.block_id, intrinsic.bias)
        native.sync_state()

    def _intrinsic_biases_for_state(
        self,
        ctx: RunContext,
        state: LoopState,
        spec: ProductionBiasSpec,
    ) -> tuple[IntrinsicBias, ...]:
        if not ctx.ph_enabled:
            return spec.intrinsic
        ph = self._ph_for_state(state)
        tags = self._ph_bias_tags_for_context(ctx)
        if not tags:
            return spec.intrinsic
        return tuple(
            IntrinsicBias(
                intrinsic.block_id,
                _ldin_bias_at_ph(
                    intrinsic.bias,
                    ph,
                    tags.get(intrinsic.block_id),
                    self.cfg.temperature,
                ),
            )
            for intrinsic in spec.intrinsic
        )

    def _ph_bias_tags_for_context(self, ctx: RunContext) -> dict[int, _PHBiasTag]:
        if self._ph_bias_tags is None:
            self._ph_bias_tags = _read_ph_bias_tags(ctx)
        return self._ph_bias_tags

    def _load_and_validate_biases(self, ctx: RunContext) -> ProductionBiasSpec | None:
        if self.cfg.bias_file is None:
            if not self.cfg.use_presets:
                return None
            if self.bias_spec is None:
                spec = load_production_preset_biases(self.cfg, ctx)
                self._validate_bias_matches_topology(ctx, spec)
                self.bias_spec = spec
            spec = self.bias_spec
        else:
            if self.bias_spec is None:
                spec = load_production_biases(self.cfg, nsubs=_nsubs_from_context(ctx))
                self._validate_bias_matches_topology(ctx, spec)
                self.bias_spec = spec
            spec = self.bias_spec

        if not self._validated_runtime:
            native = self._native_block()
            live_fnex = native.get_fnex()
            if _is_plausible_fnex(live_fnex) and not np.isclose(live_fnex, self.cfg.fnex):
                raise ProductionConfigError(
                    f"Live FNEX={live_fnex} does not match configured FNEX={self.cfg.fnex}"
                )
            if not _is_plausible_fnex(live_fnex):
                logger.warning(
                    "Ignoring invalid live FNEX read %s during production bootstrap; "
                    "using configured FNEX=%s",
                    live_fnex,
                    self.cfg.fnex,
                )
            if spec.fnex is not None and not np.isclose(spec.fnex, self.cfg.fnex):
                raise ProductionConfigError(
                    f"Bias snapshot FNEX={spec.fnex} does not match configured FNEX={self.cfg.fnex}"
                )
            self._validated_runtime = True
        return spec

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

    def _debug_bias_state(
        self,
        ctx: RunContext,
        state: LoopState | None,
        native,
        spec: ProductionBiasSpec | None,
        *,
        phase: str,
    ) -> None:
        if not _production_bias_debug_enabled():
            return
        payload = {
            "schema_version": 1,
            "phase": phase,
            "rank": ctx.rank,
            "segment_idx": None if state is None else state.segment_idx,
            "chunk_idx": None if state is None else state.chunk_idx,
            "replica_label": None if state is None else state.replica_label,
            "expected": _expected_bias_payload(spec),
            "live": _read_live_bias_state(native, ctx.ldin_blocks or ()),
        }
        path = ctx.rank_dir / f"production_bias_debug_{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)


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


_PATCH_RESIDUE_PREFIXES = {
    "ARG": "ARG",
    "ARU": "ARG",
    "ASH": "ASP",
    "ASP": "ASP",
    "CYS": "CYS",
    "GLH": "GLU",
    "GLU": "GLU",
    "HSD": "HSP",
    "HSE": "HSP",
    "HSP": "HSP",
    "LYS": "LYS",
    "TYR": "TYR",
}


def _site_residue_types_from_context(ctx: RunContext) -> tuple[str, ...]:
    if not ctx.titratable_blocks:
        raise ProductionConfigError("titratable_blocks is empty for preset production")
    residues_by_site: dict[int, str] = {}
    for block in ctx.titratable_blocks:
        residue = _preset_residue_from_patch(block.resname)
        previous = residues_by_site.setdefault(int(block.site), residue)
        if previous != residue:
            raise ProductionConfigError(
                f"site {block.site} mixes preset residues {previous!r} and {residue!r}"
            )
    missing_sites = [site for site in range(1, ctx.nsites + 1) if site not in residues_by_site]
    if missing_sites:
        raise ProductionConfigError(f"missing titratable blocks for sites {missing_sites}")
    return tuple(residues_by_site[site] for site in range(1, ctx.nsites + 1))


def _preset_residue_from_patch(resname: str) -> str:
    token = str(resname).strip().upper()
    if len(token) < 3:
        raise ProductionConfigError(f"cannot derive preset residue from patch name {resname!r}")
    prefix = token[:3]
    return _PATCH_RESIDUE_PREFIXES.get(prefix, prefix)


def _read_ph_bias_tags(ctx: RunContext) -> dict[int, _PHBiasTag]:
    path = ctx.run_dir / "prep" / "patches.dat"
    if not path.exists():
        return {}
    tags: dict[int, _PHBiasTag] = {}
    with path.open(newline="") as handle:
        for block_id, row in enumerate(csv.DictReader(handle), start=2):
            raw = str(row.get("TAG", "NONE")).strip()
            parts = raw.split()
            tag_type = parts[0].upper() if parts else "NONE"
            pka = float(parts[1]) if len(parts) > 1 else None
            tags[block_id] = _PHBiasTag(tag_type=tag_type, pka=pka)
    return tags


def _ldin_bias_at_ph(
    reference_bias: float,
    ph: float,
    tag: _PHBiasTag | None,
    temperature: float,
) -> float:
    if tag is None or tag.pka is None:
        return float(reference_bias)
    factor = math.log(10.0) * 0.0019872041 * float(temperature)
    if tag.tag_type == "UPOS":
        return float(reference_bias) + factor * (float(ph) - tag.pka)
    if tag.tag_type == "UNEG":
        return float(reference_bias) - factor * (float(ph) - tag.pka)
    return float(reference_bias)


def _is_plausible_fnex(value: float | None) -> bool:
    if value is None:
        return False
    fnex = float(value)
    return math.isfinite(fnex) and _MIN_PLAUSIBLE_FNEX < fnex < _MAX_PLAUSIBLE_FNEX


def _production_bias_debug_enabled() -> bool:
    return os.environ.get("CPHMD_DEBUG_PRODUCTION_BIAS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _read_live_bias_state(native, ldin_blocks: tuple[int, ...]) -> dict[str, Any]:
    state: dict[str, Any] = {"ldin": [], "bias_count": None, "biases": []}
    get_ldin = getattr(native, "get_ldin_params", None)
    if get_ldin is not None:
        for block_id in ldin_blocks:
            params = get_ldin(block_id)
            state["ldin"].append({"block_id": int(block_id), "params": _jsonable(params)})
    get_bias_count = getattr(native, "get_bias_count", None)
    count = get_bias_count() if get_bias_count is not None else None
    state["bias_count"] = None if count is None else int(count)
    get_bias_params = getattr(native, "get_bias_params", None)
    if count is not None and get_bias_params is not None:
        for index in range(1, int(count) + 1):
            params = get_bias_params(index)
            if params is not None:
                state["biases"].append(_jsonable(params))
    return state


def _expected_bias_payload(spec: ProductionBiasSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "intrinsic": [[item.block_id, item.bias] for item in spec.intrinsic],
        "ldbv_terms": [list(item.as_tuple()) for item in spec.ldbv_terms],
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


__all__ = [
    "IntrinsicBias",
    "LDBVTerm",
    "ProductionBiasSpec",
    "ProductionConfig",
    "ProductionConfigError",
    "ProductionHooks",
    "load_production_biases",
    "load_production_preset_biases",
    "write_production_bias_file",
]
