"""Native ``cphmd run`` command."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import typer

from cphmd.config.loader import (
    NativeRuntimeConfig,
    alf_config_from_mapping,
    load_config,
)
from cphmd.simulation.backends import AnalysisBackend
from cphmd.simulation.context import RunContext, TitratableBlock
from cphmd.simulation.loop import SimulationLoop
from cphmd.simulation.shrinker import LambdaPrecision

logger = logging.getLogger(__name__)
_LEGACY_BLOCK_NREP_RE = re.compile(
    r"^(?P<block>\s*BLOCK\s+\S+)\s+NREP\s+\S+(?P<comment>\s*(?:!.*)?)$",
    re.IGNORECASE,
)
_PHMD_TOKEN_RE = re.compile(r"\bPHMD\b", re.IGNORECASE)


class _NativeSystemProxy:
    """Lazy native-system proxy so CLI import does not require pyCHARMM."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        from cphmd.native import system as native_system

        return getattr(native_system, name)


system: Any = _NativeSystemProxy()


class SegmentLimitHooks:
    """Smoke-test hooks that run fixed segments without ALF analysis."""

    def __init__(self, max_segments: int):
        self.max_segments = int(max_segments)

    def on_system_loaded(self, ctx: RunContext, state=None) -> None:
        if not ctx.ph_enabled:
            return
        from cphmd.native import block

        block.set_ph(float(ctx.ph))
        block.sync_state()

    def after_segment(self, state, lambda_matrix, bias_matrix):
        return None

    def should_trigger_cycle(self, state) -> bool:
        return False

    def is_done(self, state) -> bool:
        return state.segment_idx >= self.max_segments


def run_simulation(
    *,
    config_path: Path,
    rank: int | None = None,
    run_dir: Path | None = None,
) -> None:
    from cphmd.native.mpi import (
        gpu_id_for_rank,
    )
    from cphmd.native.mpi import (
        init as mpi_init,
    )
    from cphmd.native.mpi import (
        rank as get_rank,
    )
    from cphmd.native.mpi import (
        size as get_size,
    )

    comm = mpi_init()
    actual_rank = rank if rank is not None else get_rank(comm)
    world_size = get_size(comm)

    from cphmd.cli._logging import install_excepthook, install_rank_logger

    cfg = load_config(config_path)
    if run_dir is not None:
        cfg = apply_run_dir_override(cfg, config_path, run_dir)
    rank_logger = install_rank_logger(
        rank=actual_rank,
        run_dir=cfg.run_dir,
        level=cfg.logging.level,
    )
    install_excepthook(logger=rank_logger)

    if world_size != cfg.nreps:
        if actual_rank == 0:
            rank_logger.error(
                "MPI world size %d does not match config nreps %d",
                world_size,
                cfg.nreps,
            )
        _abort(comm, 1)

    init_marker = cfg.run_dir / "state" / "initialized.json"
    if not init_marker.exists():
        raise RuntimeError(f"Run cphmd init first; missing {init_marker}")

    init_payload = json.loads(init_marker.read_text())
    if init_payload.get("master_seed") != cfg.master_seed:
        if actual_rank == 0:
            rank_logger.error(
                "config master_seed %s differs from initialized value %s",
                cfg.master_seed,
                init_payload.get("master_seed"),
        )
        raise SystemExit(2)

    if actual_rank == 0:
        _sync_run_prep_metadata(cfg)
    barrier = getattr(comm, "Barrier", None)
    if barrier is not None:
        barrier()

    ctx = build_run_context(
        cfg,
        rank=actual_rank,
        comm=comm,
        gpu_id=gpu_id_for_rank(comm) if _runtime_requires_gpu(cfg) else None,
    )
    if actual_rank == 0:
        _ensure_initial_block_file(cfg, ctx)
        _sync_run_prep_metadata(cfg)
    barrier = getattr(comm, "Barrier", None)
    if barrier is not None:
        barrier()
    _bootstrap_native_system(cfg)
    hooks = build_hooks(cfg, ctx)
    loop = SimulationLoop(ctx, hooks)
    loop.run()


def build_run_context(
    cfg: NativeRuntimeConfig,
    *,
    rank: int,
    comm: Any | None,
    gpu_id: int | None,
) -> RunContext:
    patch_metadata = _lambda_metadata_from_patches(_prep_dir(cfg))
    if patch_metadata is not None:
        nsites, nsubsites, lambda_headers, titratable_blocks = patch_metadata
        n_lambda_blocks = len(lambda_headers)
        rex_signs = _rex_signs_from_patches(_prep_dir(cfg))
    else:
        alf_info = load_alf_info(cfg)
        if not alf_info:
            raise FileNotFoundError(
                "Could not derive lambda topology; missing prep/patches.dat and prep/alf_info.py"
            )
        raw_nsubs = tuple(int(value) for value in alf_info.get("nsubs", [1]))
        nsites = len(raw_nsubs)
        n_lambda_blocks = int(alf_info.get("nblocks") or sum(raw_nsubs))
        nsubsites = _expand_nsubsites(raw_nsubs)
        lambda_headers, titratable_blocks = _lambda_headers_from_patches(
            _prep_dir(cfg).parent,
            n_lambda_blocks=n_lambda_blocks,
            nsubsites=nsubsites,
        )
        rex_signs = tuple(1.0 for _ in lambda_headers)
    replica_label = rank
    rex_requested = _rex_enabled(cfg.replica_exchange)
    if rex_requested and not cfg.ph:
        raise ValueError("replica exchange requires pH to be enabled")
    if rex_requested and cfg.nreps <= 1:
        raise ValueError("replica exchange requires nreps > 1")
    rex_interval = _rex_interval(
        cfg.replica_exchange,
        nsteps_per_segment=cfg.nsteps_per_segment,
        time_step_ps=cfg.time_step_ps,
    )
    ph_values = _native_ph_values(cfg) if cfg.ph else ()
    if cfg.ph and not ph_values:
        ph_values = tuple(float(7.0) for _ in range(cfg.nreps))
    has_explicit_ladder = _has_explicit_ph_ladder(cfg)
    if cfg.ph and has_explicit_ladder and len(ph_values) != cfg.nreps:
        raise ValueError(
            "explicit pH ladder length must match nreps; "
            f"got {len(ph_values)} for nreps={cfg.nreps}"
        )
    rex_enabled = rex_requested and cfg.nreps > 1
    ph_for_rank = ph_values[replica_label] if ph_values and replica_label < len(ph_values) else 7.0

    return RunContext(
        run_dir=cfg.run_dir,
        rank=rank,
        replica_label=replica_label,
        ph=ph_for_rank,
        gpu_id=gpu_id,
        dynamics_backend=cfg.dynamics_backend,
        analysis_backend=cfg.analysis_backend,
        domdec=cfg.domdec,
        lambda_headers=lambda_headers,
        nsubsites=nsubsites,
        nsites=nsites,
        nsteps_per_segment=cfg.nsteps_per_segment,
        nsavl=cfg.nsavl,
        nsavc=cfg.nsavc,
        time_step_ps=cfg.time_step_ps,
        temperature=cfg.temperature,
        checkpoint_every_segments=cfg.checkpoint_every_segments,
        lambda_precision=LambdaPrecision(cfg.archive.lambda_precision),
        master_seed=cfg.master_seed,
        config_hash=config_hash(cfg.config_path),
        ph_enabled=cfg.ph,
        ldin_blocks=tuple(range(2, n_lambda_blocks + 2)),
        rex_enabled=rex_enabled,
        replica_ph_values=ph_values,
        rex_signs=rex_signs,
        rex_exchange_every_segments=rex_interval,
        comm=comm,
        simulation_name=cfg.run_dir.name,
        walltime_safety_factor=cfg.walltime_safety_factor,
        titratable_blocks=titratable_blocks,
        startup_minimization_segments=int(
            os.environ.get("CPHMD_STARTUP_MINIMIZATION_SEGMENTS", "5")
        ),
    )


def _resolve_cli_run_dir(config_path: Path, run_dir: Path) -> Path:
    path = Path(run_dir)
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def apply_run_dir_override(
    cfg: NativeRuntimeConfig,
    config_path: Path,
    run_dir: Path,
) -> NativeRuntimeConfig:
    resolved = _resolve_cli_run_dir(config_path, run_dir)
    master_seed = cfg.master_seed
    if cfg.production is not None:
        repeat_index = _production_repeat_index(resolved)
        if repeat_index is not None:
            master_seed = _base_master_seed(cfg) + repeat_index - 1
    return replace(cfg, run_dir=resolved, master_seed=master_seed)


def _production_repeat_index(run_dir: Path) -> int | None:
    match = re.fullmatch(r"sim_(\d+)", run_dir.name)
    if match is None:
        return None
    repeat_index = int(match.group(1))
    return repeat_index if repeat_index >= 1 else None


def _base_master_seed(cfg: NativeRuntimeConfig) -> int:
    raw_seed = cfg.raw.get("master_seed")
    if raw_seed is not None:
        return int(raw_seed)
    for section_name in ("simulation", "alf"):
        section = cfg.raw.get(section_name)
        if isinstance(section, dict) and section.get("master_seed") is not None:
            return int(section["master_seed"])
    return int(cfg.master_seed)


def _sync_run_prep_metadata(cfg: NativeRuntimeConfig) -> None:
    input_prep = cfg.input_folder / "prep"
    run_prep = cfg.run_dir / "prep"
    if not input_prep.exists() or input_prep.resolve() == run_prep.resolve():
        return
    run_prep.mkdir(parents=True, exist_ok=True)
    for name in ("patches.dat", "block.str", "alf_info.py"):
        source = input_prep / name
        if source.exists():
            shutil.copy2(source, run_prep / name)


def build_hooks(cfg: NativeRuntimeConfig, ctx: RunContext):
    if cfg.production is not None:
        from cphmd.training.production_hooks import ProductionHooks

        return ProductionHooks(ctx, cfg.production)
    analysis_backend = getattr(cfg, "analysis_backend", AnalysisBackend.CUDA_WHAM)
    if analysis_backend is AnalysisBackend.DISABLED:
        return SegmentLimitHooks(max_segments=cfg.end)

    from cphmd.training import (
        ALFCycleRunner,
        ALFHooks,
        ALFTrainingConfig,
        BiasRebuilder,
        NativeALFAnalyzer,
    )

    alf_config = _alf_config_from_native_config(cfg)
    phase_repeats = {
        phase: _alf_repeats_for_phase(
            alf_config,
            phase,
            nsteps_per_segment=ctx.nsteps_per_segment,
            time_step_ps=ctx.time_step_ps,
        )
        for phase in (1, 2, 3)
    }
    repeats = phase_repeats.get(int(cfg.phase), phase_repeats[3])
    nsubs = _nsubs_from_context(ctx)
    alf_info = load_alf_info(cfg)
    if not alf_info:
        alf_info = {
            "name": ctx.simulation_name,
            "nsubs": list(nsubs),
            "nblocks": sum(nsubs),
            "nreps": cfg.nreps,
            "ncentral": cfg.nreps // 2,
            "nnodes": 1,
            "temp": cfg.temperature,
            "ntersite": [0, 0],
        }
    _validate_native_phase_controls(alf_config)
    analyzer = NativeALFAnalyzer(
        config=alf_config,
        ctx=ctx,
        alf_info=alf_info,
        work_dir=cfg.run_dir,
        initial_bias_provider=_native_initial_bias_provider(cfg, nsubs),
    )
    return ALFHooks(
        ALFTrainingConfig(
            cycle_every_segments=repeats,
            end_cycle=cfg.end,
            cache_segments=max(phase_repeats.values()),
            generate_dashboard_plots=bool(getattr(alf_config, "generate_dashboard_plots", True)),
            generate_population_plots=bool(getattr(alf_config, "generate_population_plots", False)),
            generate_g_profiles_2d=bool(getattr(alf_config, "generate_g_profiles_2d", False)),
            generate_g_profiles_3d=bool(getattr(alf_config, "generate_g_profiles_3d", False)),
            phase1_repeats=phase_repeats[1],
            phase2_repeats=phase_repeats[2],
            phase3_repeats=phase_repeats[3],
            phase1_cycles=getattr(alf_config, "phase1_cycles", None),
            phase2_cycles=getattr(alf_config, "phase2_cycles", None),
        ),
        nsubs=nsubs,
        cycle_runner=ALFCycleRunner(
            analyzer=analyzer,
            rebuilder=BiasRebuilder(),
        ),
        replica_ph_values=ctx.replica_ph_values,
        work_dir=cfg.run_dir,
    )


def _native_initial_bias_provider(cfg: NativeRuntimeConfig, nsubs: tuple[int, ...]):
    alf_section = cfg.raw.get("alf", {}) or {}
    if not (isinstance(alf_section, dict) and alf_section.get("bias_guess") is True):
        return None

    def provider():
        import numpy as np
        import pandas as pd

        from cphmd.core.bias_guesser import guess_initial_biases_combined
        from cphmd.training.bias_snapshot import BiasSnapshot

        patch_info = pd.read_csv(_prep_dir(cfg) / "patches.dat")
        nsubs_list = [int(value) for value in nsubs]
        try:
            b, c = guess_initial_biases_combined(
                patch_info,
                nsubs_list,
                fnex=float(cfg.fnex),
                legacy=False,
                reload_system=lambda: _bootstrap_native_system(cfg),
            )
        finally:
            _bootstrap_native_system(cfg)

        nblocks = sum(nsubs_list)
        b, c = _sanitize_initial_bias_arrays(b, c)
        zeros = np.zeros((nblocks, nblocks), dtype=np.float64)
        return BiasSnapshot.from_arrays(
            b=b,
            c=c,
            x=zeros,
            s=zeros,
            nsubs=nsubs,
        )

    return provider


def _sanitize_initial_bias_arrays(b, c):
    import numpy as np

    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    if np.all(np.isfinite(b_arr)) and np.all(np.isfinite(c_arr)):
        return b_arr, c_arr
    print("Warning: non-finite initial bias guess values were replaced with 0.0")
    return (
        np.nan_to_num(b_arr, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(c_arr, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _validate_native_phase_controls(config: Any) -> None:
    if bool(getattr(config, "auto_phase_switch", False)) and (
        getattr(config, "phase1_cycles", None) is not None
        or getattr(config, "phase2_cycles", None) is not None
    ):
        raise ValueError(
            "alf.auto_phase_switch cannot be combined with fixed phase iteration counts"
        )


_DEFAULT_ALF_ITERATION_PS = {1: 100.0, 2: 1000.0, 3: 10000.0}


def _alf_repeats_for_phase(
    config: Any,
    phase: int,
    *,
    nsteps_per_segment: int,
    time_step_ps: float,
) -> int:
    phase = int(phase)
    repeat_attr = {
        1: "phase1_repeats",
        2: "phase2_repeats",
        3: "phase3_repeats",
    }.get(phase, "phase3_repeats")
    ps_attr = f"phase{phase}_iteration_ps"
    steps_attr = f"phase{phase}_iteration_steps"
    values = {
        repeat_attr: getattr(config, repeat_attr, None),
        ps_attr: getattr(config, ps_attr, None),
        steps_attr: getattr(config, steps_attr, None),
    }
    present = [name for name, value in values.items() if value is not None]
    if len(present) > 1:
        keys = ", ".join(f"alf.{name}" for name in present)
        raise ValueError(f"conflicting Phase {phase} ALF iteration length keys: {keys}")

    repeats_value = values[repeat_attr]
    if repeats_value is not None:
        repeats = int(repeats_value)
    elif values[steps_attr] is not None:
        repeats = _alf_repeats_from_steps(
            values[steps_attr],
            nsteps_per_segment=nsteps_per_segment,
            name=f"alf.{steps_attr}",
        )
    else:
        iteration_ps = values[ps_attr]
        if iteration_ps is None:
            iteration_ps = _DEFAULT_ALF_ITERATION_PS.get(phase, _DEFAULT_ALF_ITERATION_PS[3])
        repeats = _alf_repeats_from_ps(
            iteration_ps,
            nsteps_per_segment=nsteps_per_segment,
            time_step_ps=time_step_ps,
            name=f"alf.{ps_attr}",
        )
    if repeats <= 0:
        raise ValueError(f"alf.{repeat_attr} must be positive")
    return repeats


def _alf_repeats_from_ps(
    value: Any,
    *,
    nsteps_per_segment: int,
    time_step_ps: float,
    name: str,
) -> int:
    iteration_ps = float(value)
    if iteration_ps <= 0:
        raise ValueError(f"{name} must be positive")
    md_block_ps = max(1, int(nsteps_per_segment)) * float(time_step_ps)
    ratio = iteration_ps / md_block_ps
    rounded = round(ratio)
    if rounded <= 0 or abs(ratio - rounded) > 1e-9:
        raise ValueError(f"{name} must be divisible by the MD block length")
    return int(rounded)


def _alf_repeats_from_steps(
    value: Any,
    *,
    nsteps_per_segment: int,
    name: str,
) -> int:
    iteration_steps = int(value)
    if iteration_steps <= 0:
        raise ValueError(f"{name} must be positive")
    md_block_steps = max(1, int(nsteps_per_segment))
    if iteration_steps % md_block_steps != 0:
        raise ValueError(f"{name} must be divisible by the MD block length")
    return max(1, iteration_steps // md_block_steps)


def _nsubs_from_context(ctx: RunContext) -> tuple[int, ...]:
    return tuple(
        sum(1 for value in ctx.nsubsites[1:] if value == site) for site in range(1, ctx.nsites + 1)
    )


def _runtime_requires_gpu(cfg: NativeRuntimeConfig) -> bool:
    if cfg.dynamics_backend.requires_gpu:
        return True
    if cfg.production is None and cfg.analysis_backend.requires_gpu:
        return True
    return False


def _native_ph_values(cfg: NativeRuntimeConfig) -> tuple[float, ...]:
    if not cfg.ph:
        return ()
    if _has_explicit_ph_ladder(cfg):
        return cfg.ph_values
    patches_path = _prep_dir(cfg) / "patches.dat"
    if not patches_path.exists():
        return cfg.ph_values

    import pandas as pd

    from cphmd.core.cphmd_params import (
        compute_all_site_parameters,
        get_delta_pKa_for_phase,
        replica_pH,
    )

    patch_info = pd.read_csv(patches_path)
    if "site" not in patch_info.columns or "sub" not in patch_info.columns:
        if "SELECT" not in patch_info.columns:
            return cfg.ph_values
        patch_info[["site", "sub"]] = patch_info["SELECT"].str.extract(r"(?i)s(\d+)s(\d+)")
    params = compute_all_site_parameters(patch_info, cfg.temperature)
    delta_pka = get_delta_pKa_for_phase(cfg.phase)
    ncentral = cfg.nreps // 2
    return tuple(
        replica_pH(params.effective_pH, delta_pka, replica_idx, ncentral)
        for replica_idx in range(cfg.nreps)
    )


def _has_explicit_ph_ladder(cfg: NativeRuntimeConfig) -> bool:
    runtime = _runtime_raw_section(cfg)
    return any(key in runtime for key in ("ph_values", "pH_values", "ph_start", "ph_end")) or (
        (
            "ph" in runtime
            and isinstance(runtime["ph"], (int, float))
            and not isinstance(runtime["ph"], bool)
        )
        or (
            "pH" in runtime
            and isinstance(runtime["pH"], (int, float))
            and not isinstance(runtime["pH"], bool)
        )
    )


def _runtime_raw_section(cfg: NativeRuntimeConfig) -> dict[str, Any]:
    if cfg.production is not None and isinstance(cfg.raw.get("simulation"), dict):
        return cfg.raw["simulation"]
    if isinstance(cfg.raw.get("alf"), dict):
        return cfg.raw["alf"]
    return {}


def _alf_config_from_native_config(cfg: NativeRuntimeConfig):
    alf_section = dict(cfg.raw.get("alf", {}) or {})
    if "input_folder" not in alf_section:
        alf_section["input_folder"] = cfg.input_folder
    alf_config = alf_config_from_mapping(alf_section, include_defaults=True)
    alf_config.input_folder = cfg.input_folder
    alf_config.toppar_dir = cfg.toppar_dir
    alf_config.topology_files = list(cfg.topology_files)
    alf_config.extra_files = list(cfg.extra_files)
    alf_config.start = cfg.start
    alf_config.end = cfg.end
    alf_config.phase = cfg.phase
    alf_config.nreps = cfg.nreps
    alf_config.temperature = cfg.temperature
    alf_config.cutnb = cfg.cutnb
    alf_config.ctofnb = cfg.ctofnb
    alf_config.ctonnb = cfg.ctonnb
    alf_config.elec_type = cfg.elec_type
    alf_config.vdw_type = cfg.vdw_type
    alf_config.fnex = cfg.fnex
    alf_config.ph = cfg.ph
    return alf_config


def find_system_files(cfg: NativeRuntimeConfig) -> tuple[Path, Path]:
    input_prep = cfg.input_folder / "prep"
    run_prep = cfg.run_dir / "prep"
    candidates = (
        (input_prep / "system_hmr.psf", input_prep / "system_hmr.crd"),
        (input_prep / "system.psf", input_prep / "system.crd"),
        (run_prep / "system_hmr.psf", run_prep / "system_hmr.crd"),
        (run_prep / "system.psf", run_prep / "system.crd"),
        (cfg.run_dir / "solvated.psf", cfg.run_dir / "solvated.crd"),
    )
    for psf_path, crd_path in candidates:
        if psf_path.exists() and crd_path.exists():
            return psf_path, crd_path
    searched = ", ".join(f"{psf}/{crd}" for psf, crd in candidates)
    raise FileNotFoundError(f"Could not find initialized PSF/CRD pair; searched {searched}")


def load_alf_info(cfg: NativeRuntimeConfig) -> dict[str, Any]:
    path = _prep_dir(cfg) / "alf_info.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_cphmd_alf_info", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = getattr(module, "alf_info", None)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must define alf_info dictionary")
    return data


def config_hash(config_path: Path) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def _expand_nsubsites(raw_nsubs: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for site_idx, count in enumerate(raw_nsubs, start=1):
        values.extend(site_idx for _ in range(max(0, count)))
    if len(values) == 1:
        values.append(1)
    return tuple(values)


def _lambda_headers_from_patches(
    run_dir: Path,
    *,
    n_lambda_blocks: int,
    nsubsites: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[TitratableBlock, ...]]:
    path = run_dir / "prep" / "patches.dat"
    if path.exists():
        rows = _read_patch_rows(path)
    else:
        rows = []
    blocks: list[TitratableBlock] = []
    if len(rows) == n_lambda_blocks:
        for idx, row in enumerate(rows):
            blocks.append(
                TitratableBlock(
                    block_id=idx + 2,
                    segid=str(row["SEGID"]).strip(),
                    resid=str(row["RESID"]).strip(),
                    resname=str(row["PATCH"]).strip(),
                    site=int(nsubsites[idx + 1]),
                )
            )
    else:
        blocks = [
            TitratableBlock(
                block_id=idx + 2,
                segid="SYS",
                resid=str(idx + 1),
                resname="LAMBDA",
                site=int(nsubsites[idx + 1]) if idx + 1 < len(nsubsites) else 1,
            )
            for idx in range(n_lambda_blocks)
        ]
    return tuple(block.lambda_header for block in blocks), tuple(blocks)


_SELECT_RE = re.compile(r"^s(?P<site>\d+)s(?P<subsite>\d+)$")


def _lambda_metadata_from_patches(
    prep_dir: Path,
) -> tuple[int, tuple[int, ...], tuple[str, ...], tuple[TitratableBlock, ...]] | None:
    path = prep_dir / "patches.dat"
    if not path.exists():
        return None
    rows = _read_patch_rows(path)
    if not rows:
        return None
    parsed_sites: list[int] = []
    blocks: list[TitratableBlock] = []
    for idx, row in enumerate(rows):
        select = str(row.get("SELECT", "")).strip()
        match = _SELECT_RE.fullmatch(select)
        if match is None:
            raise ValueError(f"{path} row {idx + 1} has invalid SELECT value {select!r}")
        site = int(match.group("site"))
        parsed_sites.append(site)
        blocks.append(
            TitratableBlock(
                block_id=idx + 2,
                segid=str(row["SEGID"]).strip(),
                resid=str(row["RESID"]).strip(),
                resname=str(row["PATCH"]).strip(),
                site=site,
            )
        )
    nsites = max(parsed_sites)
    expected_sites = set(range(1, nsites + 1))
    found_sites = set(parsed_sites)
    if found_sites != expected_sites:
        raise ValueError(
            f"{path} SELECT sites must be contiguous 1..{nsites}; found {sorted(found_sites)}"
        )
    nsubsites = (0,) + tuple(parsed_sites)
    return nsites, nsubsites, tuple(block.lambda_header for block in blocks), tuple(blocks)


def _read_patch_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _rex_signs_from_patches(prep_dir: Path) -> tuple[float, ...]:
    rows = _read_patch_rows(prep_dir / "patches.dat")
    signs: list[float] = []
    for row in rows:
        tag = str(row.get("TAG", "NONE")).strip().upper()
        flag = tag.split()[0] if tag else "NONE"
        if flag == "UPOS":
            signs.append(1.0)
        elif flag == "UNEG":
            signs.append(-1.0)
        else:
            signs.append(0.0)
    return tuple(signs)


def _rex_enabled(replica_exchange: Any) -> bool:
    if isinstance(replica_exchange, dict):
        return bool(replica_exchange.get("enabled", False))
    if isinstance(replica_exchange, bool):
        return replica_exchange
    return bool(getattr(replica_exchange, "enabled", False))


def _rex_interval(
    replica_exchange: Any,
    *,
    nsteps_per_segment: int = 1,
    time_step_ps: float = 0.004,
) -> int:
    if isinstance(replica_exchange, dict):
        removed = sorted(
            key for key in ("apply_state", "backend") if key in replica_exchange
        )
        if removed:
            key = removed[0]
            raise ValueError(
                f"replica_exchange.{key} is removed; use the direct pyCHARMM exchange path"
            )
        interval_keys = [
            key
            for key in (
                "exchange_every_segments",
                "exchange_interval_ps",
                "exchange_interval_steps",
                "exchange_freq",
            )
            if key in replica_exchange
        ]
        if len(interval_keys) > 1:
            keys = ", ".join(f"replica_exchange.{key}" for key in interval_keys)
            raise ValueError(f"conflicting replica exchange interval keys: {keys}")
        if "exchange_every_segments" in replica_exchange:
            return max(1, int(replica_exchange["exchange_every_segments"]))
        if "exchange_interval_ps" in replica_exchange:
            return _rex_segments_from_ps(
                replica_exchange["exchange_interval_ps"],
                nsteps_per_segment=nsteps_per_segment,
                time_step_ps=time_step_ps,
            )
        if "exchange_interval_steps" in replica_exchange:
            return _rex_segments_from_steps(
                replica_exchange["exchange_interval_steps"],
                nsteps_per_segment=nsteps_per_segment,
            )
        if "exchange_freq" in replica_exchange:
            return _rex_segments_from_steps(
                replica_exchange["exchange_freq"],
                nsteps_per_segment=nsteps_per_segment,
                name="replica_exchange.exchange_freq",
            )
    else:
        values = {
            "exchange_every_segments": getattr(replica_exchange, "exchange_every_segments", None),
            "exchange_interval_ps": getattr(replica_exchange, "exchange_interval_ps", None),
            "exchange_interval_steps": getattr(
                replica_exchange,
                "exchange_interval_steps",
                None,
            ),
            "exchange_freq": getattr(replica_exchange, "exchange_freq", None),
        }
        present = [key for key, value in values.items() if value is not None]
        if len(present) > 1:
            keys = ", ".join(f"replica_exchange.{key}" for key in present)
            raise ValueError(f"conflicting replica exchange interval keys: {keys}")
        every_segments = values["exchange_every_segments"]
        if every_segments is not None:
            return max(1, int(every_segments))
        exchange_interval_ps = values["exchange_interval_ps"]
        if exchange_interval_ps is not None:
            return _rex_segments_from_ps(
                exchange_interval_ps,
                nsteps_per_segment=nsteps_per_segment,
                time_step_ps=time_step_ps,
            )
        exchange_interval_steps = values["exchange_interval_steps"]
        if exchange_interval_steps is not None:
            return _rex_segments_from_steps(
                exchange_interval_steps,
                nsteps_per_segment=nsteps_per_segment,
            )
        exchange_freq = values["exchange_freq"]
        if exchange_freq is not None:
            return _rex_segments_from_steps(
                exchange_freq,
                nsteps_per_segment=nsteps_per_segment,
                name="replica_exchange.exchange_freq",
            )
    return _rex_segments_from_ps(
        4.0,
        nsteps_per_segment=nsteps_per_segment,
        time_step_ps=time_step_ps,
    )


def _rex_segments_from_ps(
    value: Any,
    *,
    nsteps_per_segment: int,
    time_step_ps: float,
) -> int:
    interval_ps = float(value)
    if interval_ps <= 0:
        raise ValueError("replica_exchange.exchange_interval_ps must be positive")
    md_block_ps = max(1, int(nsteps_per_segment)) * float(time_step_ps)
    ratio = interval_ps / md_block_ps
    rounded = round(ratio)
    if rounded <= 0 or abs(ratio - rounded) > 1e-9:
        raise ValueError(
            "replica_exchange.exchange_interval_ps must be divisible by the MD block length"
        )
    return int(rounded)


def _rex_segments_from_steps(
    value: Any,
    *,
    nsteps_per_segment: int,
    name: str = "replica_exchange.exchange_interval_steps",
) -> int:
    interval_steps = int(value)
    if interval_steps <= 0:
        raise ValueError(f"{name} must be positive")
    segment_steps = max(1, int(nsteps_per_segment))
    if interval_steps % segment_steps != 0:
        raise ValueError(f"{name} must be divisible by the MD block length")
    return max(1, interval_steps // segment_steps)


def _prep_dir(cfg: NativeRuntimeConfig) -> Path:
    input_prep = cfg.input_folder / "prep"
    if input_prep.exists():
        return input_prep
    return cfg.run_dir / "prep"


def _ensure_initial_block_file(cfg: NativeRuntimeConfig, ctx: RunContext) -> Path | None:
    """Create a CpHMD-owned initial BLOCK stream when setup provided only patches.dat."""
    prep_dir = _prep_dir(cfg)
    block_path = prep_dir / "block.str"
    if block_path.exists():
        return _ensure_native_block_stream(cfg, block_path, ph_enabled=ctx.ph_enabled)

    patches_path = prep_dir / "patches.dat"
    if not patches_path.exists():
        return None

    rows = _read_patch_rows(patches_path)
    if not rows:
        return None

    block_path = _generated_block_stream_path(cfg, prep_dir)
    block_path.parent.mkdir(parents=True, exist_ok=True)
    block_path.write_text(_initial_block_stream(rows, cfg, ctx), encoding="utf-8")
    return _ensure_native_block_stream(cfg, block_path, ph_enabled=ctx.ph_enabled)


def _generated_block_stream_path(cfg: NativeRuntimeConfig, prep_dir: Path) -> Path:
    run_prep = cfg.run_dir / "prep"
    if prep_dir.resolve() == run_prep.resolve():
        return prep_dir / "block.str"
    return run_prep / "block.str"


def _ensure_native_block_stream(
    cfg: NativeRuntimeConfig,
    block_path: Path,
    *,
    ph_enabled: bool,
) -> Path:
    text = block_path.read_text(encoding="utf-8")
    normalized, changed = _normalize_native_block_stream(text, ph_enabled=ph_enabled)
    runtime_path = _runtime_block_stream_path(cfg)
    if not changed:
        runtime_path.unlink(missing_ok=True)
        return block_path
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(normalized, encoding="utf-8")
    return runtime_path


def _runtime_block_stream_path(cfg: NativeRuntimeConfig) -> Path:
    return cfg.run_dir / "state" / "block.native.str"


def _block_stream_path_for_bootstrap(cfg: NativeRuntimeConfig, prep_dir: Path) -> Path | None:
    for candidate in (
        _runtime_block_stream_path(cfg),
        prep_dir / "block.str",
        cfg.run_dir / "prep" / "block.str",
    ):
        if candidate.exists():
            return candidate
    return None


def _normalize_native_block_stream(text: str, *, ph_enabled: bool) -> tuple[str, bool]:
    lines: list[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        match = _LEGACY_BLOCK_NREP_RE.match(body)
        if match is not None:
            lines.append(f"{match.group('block')}{match.group('comment')}{newline}")
            changed = True
        elif not ph_enabled and _PHMD_TOKEN_RE.search(body):
            changed = True
        else:
            lines.append(line)
    return "".join(lines), changed


def _initial_block_stream(
    rows: list[dict[str, str]],
    cfg: NativeRuntimeConfig,
    ctx: RunContext,
) -> str:
    patch_info = _patch_rows_with_sites(rows)
    nsubs = _nsubs_from_context(ctx)
    nblocks = len(patch_info) + 1
    lines: list[str] = [
        "! Initial CpHMD BLOCK stream generated from patches.dat",
        f"BLOCK {nblocks}",
        "",
    ]

    for idx, row in enumerate(patch_info):
        lines.extend(_block_call_lines(idx + 2, row))
    lines.append("")

    for site in sorted({int(row["site"]) for row in patch_info}):
        site_indices = [idx for idx, row in enumerate(patch_info) if int(row["site"]) == site]
        for left_pos, left_idx in enumerate(site_indices):
            for right_idx in site_indices[left_pos + 1 :]:
                lines.append(f"ADEXCL {left_idx + 2:<4} {right_idx + 2:<4}")
    lines.append("")

    lines.extend(
        [
            "QLDM THETA",
            f"LANG TEMP {float(ctx.temperature):.2f}",
        ]
    )
    if ctx.ph_enabled:
        lines.append(f"PHMD pH {float(ctx.ph):.3f}")
    elec_type = str(cfg.elec_type).lower()
    lines.extend(
        [
            "SOFT ON",
        ]
    )
    if elec_type in ("pmeex", "pme_ex"):
        lines.append("PMEL EX")
    elif elec_type in ("pmeon", "pme_on"):
        lines.append("PMEL ON")
    elif elec_type in ("pmenn", "pme_nn"):
        lines.append("PMEL NN")
    lines.extend(
        [
            "",
            "LDIN 1    1.0000 0.0000 12.0 0.0000 5.0 ! NONE",
        ]
    )

    for idx, row in enumerate(patch_info):
        site = int(row["site"])
        sub_count = nsubs[site - 1] if 0 < site <= len(nsubs) else 1
        lambda0 = 1.0 / max(1, sub_count)
        tag = str(row.get("TAG", "")).strip()
        comment = f" ! {tag}" if tag else ""
        lines.append(f"LDIN {idx + 2:<4} {lambda0:.4f} 0.0000 12.0 0.0000 5.0{comment}")

    lines.extend(
        [
            "",
            "RMLA BOND THET IMPR",
            "MSLD 0 -",
        ]
    )
    site_values = [int(row["site"]) for row in patch_info]
    for idx, site in enumerate(site_values):
        continuation = " -" if idx < len(site_values) - 1 else " -"
        lines.append(f"{site}{continuation}")
    lines.extend(
        [
            f"FNEX {float(cfg.fnex):.3f}",
            "MSMA",
        ]
    )

    lines.extend(
        [
            "",
            _zero_ldbv_stream(patch_info, fnex=float(cfg.fnex)).rstrip(),
            "END",
            "",
        ]
    )
    return "\n".join(lines)


def _block_call_lines(block_id: int, row: dict[str, Any]) -> list[str]:
    segid = str(row["SEGID"]).strip()
    resid = str(row["RESID"]).strip()
    patch = str(row["PATCH"]).strip()
    atoms = [atom.strip() for atom in str(row.get("ATOMS", "")).split() if atom.strip()]
    if not atoms:
        return [
            f"CALL {block_id:<4} SELEct segid {segid} "
            f".and. resid {resid} "
            f".and. resname {patch} END"
        ]

    lines = [
        f"CALL {block_id:<4} SELEct segid {segid} .and. resid {resid} -",
        f".and. resname {patch} .and. ( -",
    ]
    for atom in atoms:
        lines.append(f"type {atom} .or. -")
    lines.append("none ) END")
    return lines


def _patch_rows_with_sites(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        item: dict[str, Any] = dict(row)
        if "site" in item and "sub" in item and item["site"] and item["sub"]:
            item["site"] = int(item["site"])
            item["sub"] = int(item["sub"])
        else:
            select = str(item.get("SELECT", "")).strip()
            match = _SELECT_RE.fullmatch(select)
            if match is None:
                raise ValueError(f"patch row {idx} has invalid SELECT value {select!r}")
            item["site"] = int(match.group("site"))
            item["sub"] = int(match.group("subsite"))
        parsed.append(item)
    return parsed


def _zero_ldbv_stream(patch_info: list[dict[str, Any]], *, fnex: float) -> str:
    from cphmd.core.bias_constants import derive_bias_constants

    constants = derive_bias_constants(fnex)
    lines: list[str] = []
    idx = 0
    for site in sorted({int(row["site"]) for row in patch_info}):
        site_indices = [i for i, row in enumerate(patch_info) if int(row["site"]) == site]
        for left_pos, left_idx in enumerate(site_indices):
            for right_idx in site_indices[left_pos + 1 :]:
                idx += 1
                lines.append(f"LDBV {idx:<4} {left_idx + 2:<4} {right_idx + 2:<4} 6    0.0 0.0 0")
        for left_idx in site_indices:
            for right_idx in site_indices:
                if left_idx == right_idx:
                    continue
                idx += 1
                lines.append(
                    f"LDBV {idx:<4} {left_idx + 2:<4} {right_idx + 2:<4} "
                    f"8    {constants.chi_offset:.5f} 0.0 0"
                )
        for left_idx in site_indices:
            for right_idx in site_indices:
                if left_idx == right_idx:
                    continue
                idx += 1
                lines.append(
                    f"LDBV {idx:<4} {left_idx + 2:<4} {right_idx + 2:<4} "
                    f"10   {constants.omega_decay:g} 0.0 0"
                )
    return f"LDBI {idx}\n" + "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _AtomSelection:
    raw: str | None = None


_IMAGE_CENTERING_SOLVENT_SELECTION = (
    "segid SOLV .or. segid IONS .or. segid WT00 .or. "
    "resname TIP3 .or. resname SOD .or. resname CLA .or. resname POT .or. resname CES"
)


@dataclass(frozen=True)
class _BoxParameters:
    crystal_type: str
    dimensions: tuple[float, float, float]
    angles: tuple[float, float, float] = (90.0, 90.0, 90.0)

    @classmethod
    def from_file(cls, box_file: Path | str) -> "_BoxParameters":
        lines = Path(box_file).read_text().splitlines()
        if len(lines) < 3:
            raise ValueError(f"{box_file} must contain crystal type, dimensions, and angles")
        return cls(
            crystal_type=lines[0].strip(),
            dimensions=tuple(float(value) for value in lines[1].split()[:3]),
            angles=tuple(float(value) for value in lines[2].split()[:3]),
        )


@dataclass(frozen=True)
class _FFTParameters:
    fftx: int
    ffty: int
    fftz: int

    @classmethod
    def from_file(cls, fft_file: Path | str) -> "_FFTParameters":
        values = Path(fft_file).read_text().split()
        if len(values) < 3:
            raise ValueError(f"{fft_file} must contain three FFT grid dimensions")
        return cls(fftx=int(values[0]), ffty=int(values[1]), fftz=int(values[2]))


@dataclass
class _NonBondedRuntimeConfig:
    cutnb: float = 14.0
    cutim: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: str = "pmeex"
    vdw_type: str = "vswitch"
    kappa: float = 0.320
    order: int = 6
    fftx: int | None = None
    ffty: int | None = None
    fftz: int | None = None

    def to_kwargs(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "elec": True,
            "atom": True,
            "cdie": True,
            "eps": 1,
            "cutnb": self.cutnb,
            "cutim": self.cutim,
            "ctofnb": self.ctofnb,
            "ctonnb": self.ctonnb,
            "inbfrq": -1,
            "imgfrq": -1,
            "nbxmod": 5,
        }

        if self.vdw_type == "vswitch":
            params["vswitch"] = True
        else:
            params["vfswitch"] = True

        if self.elec_type in ("pmeex", "pmeon", "pmenn"):
            params.update(
                {
                    "switch": True,
                    "ewald": True,
                    "pmewald": True,
                    "kappa": self.kappa,
                    "order": self.order,
                }
            )
            if self.fftx is not None:
                params["fftx"] = self.fftx
                params["ffty"] = self.ffty
                params["fftz"] = self.fftz
        elif self.elec_type == "fshift":
            params["fshift"] = True
        else:
            params["fswitch"] = True
        return params


def _read_topology_files(
    toppar_dir: Path | str,
    topology_files: list[str],
    *,
    verbose: bool = False,
) -> None:
    toppar_dir = Path(toppar_dir)

    if not verbose:
        system.set_prnlev(-1)

    rtf_files = [f for f in topology_files if Path(f).suffix.lower() == ".rtf"]
    prm_files = [f for f in topology_files if Path(f).suffix.lower() == ".prm"]
    str_files = [f for f in topology_files if Path(f).suffix.lower() == ".str"]

    system.set_bomb_level(-2)
    system.set_warn_level(-1)
    try:
        if rtf_files:
            system.read_rtf(toppar_dir / rtf_files[0])
            for file_name in rtf_files[1:]:
                system.read_rtf(toppar_dir / file_name, append=True)

        if prm_files:
            system.read_param(toppar_dir / prm_files[0])
            for file_name in prm_files[1:]:
                system.read_param(toppar_dir / file_name, append=True)

        for file_name in str_files:
            system.stream_file(toppar_dir / file_name)
    finally:
        system.set_warn_level(5)
        system.set_bomb_level(0)
        if not verbose:
            system.set_prnlev(5)
        system.set_iofmt(extended=True)


def _read_structure(psf_file: Path | str, crd_file: Path | str) -> None:
    system.set_bomb_level(-1)
    try:
        system.read_psf(psf_file)
        system.read_coor(crd_file)
    finally:
        system.set_bomb_level(0)


def _setup_crystal(
    box_params: _BoxParameters,
    nb_config: _NonBondedRuntimeConfig,
    *,
    use_image_centering: bool = True,
) -> None:
    crystal_type = box_params.crystal_type.strip().upper()
    a, b, c = box_params.dimensions
    alpha, beta, gamma = box_params.angles

    system.crystal_free()
    system.crystal_define(
        shape=crystal_type,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    system.crystal_build(cutoff=nb_config.cutim)
    if use_image_centering:
        system.image_setup(
            byres=True,
            selection=_AtomSelection(raw=_IMAGE_CENTERING_SOLVENT_SELECTION),
        )
        system.image_setup(
            byres=False,
            selection=_AtomSelection(raw=f".not. ({_IMAGE_CENTERING_SOLVENT_SELECTION})"),
        )


def _bootstrap_native_system(cfg: NativeRuntimeConfig) -> None:
    _read_topology_files(cfg.toppar_dir, list(cfg.topology_files), verbose=False)
    for extra_file in cfg.extra_files:
        _load_charmm_file(extra_file)
    if cfg.production is not None:
        for topology_file in cfg.production.topology_files:
            _load_charmm_file(topology_file)
        for extra_file in cfg.production.extra_files:
            _load_charmm_file(extra_file)

    psf_path, crd_path = find_system_files(cfg)
    _read_structure(psf_path, crd_path)

    prep_dir = _prep_dir(cfg)
    nb_config = _nonbonded_config(cfg, prep_dir)
    box_path = prep_dir / "box.dat"
    fft_path = prep_dir / "fft.dat"
    if box_path.exists() and fft_path.exists():
        box = _BoxParameters.from_file(box_path)
        fft = _FFTParameters.from_file(fft_path)
        nb_config.fftx = fft.fftx
        nb_config.ffty = fft.ffty
        nb_config.fftz = fft.fftz
        _setup_crystal(box, nb_config, use_image_centering=not bool(cfg.cent_ncres))

    block_path = _block_stream_path_for_bootstrap(cfg, prep_dir)
    if block_path is not None:
        system.stream_file(block_path)
    elif cfg.production is not None:
        raise FileNotFoundError(
            "production run requires initialized BLOCK file in "
            f"{prep_dir} or {cfg.run_dir / 'prep'}"
        )

    system.nbonds_setup(**nb_config.to_kwargs())

    restrains_path = prep_dir / "restrains.str"
    if restrains_path.exists():
        system.stream_file(restrains_path)


def _nonbonded_config(cfg: NativeRuntimeConfig, prep_dir: Path):
    return _NonBondedRuntimeConfig(
        cutnb=cfg.cutnb,
        cutim=cfg.cutnb,
        ctofnb=cfg.ctofnb,
        ctonnb=cfg.ctonnb,
        elec_type=cfg.elec_type,
        vdw_type=cfg.vdw_type,
    )


def _load_charmm_file(path: Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".rtf":
        system.read_rtf(path, append=True)
    elif suffix == ".prm":
        system.read_param(path, append=True)
    else:
        system.stream_file(path)


def _abort(comm, code: int) -> None:
    abort = getattr(comm, "Abort", None)
    if abort is not None:
        abort(code)
    raise SystemExit(code)


def register(app: typer.Typer) -> None:
    @app.command("run")
    def _cmd(
        config: Path = typer.Option(..., "-c", "--config", exists=True),
        run_dir: Path | None = typer.Option(None, "--run-dir", "-r"),
    ) -> None:
        run_simulation(config_path=config, run_dir=run_dir)
