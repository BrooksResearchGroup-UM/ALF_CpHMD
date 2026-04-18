"""Native ``cphmd run`` command."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import typer

from cphmd.config.loader import NativeRuntimeConfig, load_config
from cphmd.simulation.context import LoopState, RunContext
from cphmd.simulation.loop import SimulationLoop
from cphmd.simulation.shrinker import LambdaPrecision

logger = logging.getLogger(__name__)


class SegmentLimitHooks:
    """Minimal Phase-5 loop hooks that stop after ``max_segments``."""

    def __init__(self, max_segments: int):
        self.max_segments = max_segments

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> None:
        return None

    def is_done(self, state: LoopState) -> bool:
        return state.segment_idx >= self.max_segments


def run_simulation(*, config_path: Path, rank: int | None = None) -> None:
    from cphmd.native.mpi import (
        finalize as mpi_finalize,
    )
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
    from cphmd.native import system

    cfg = load_config(config_path)
    rank_logger = install_rank_logger(
        rank=actual_rank,
        run_dir=cfg.run_dir,
        level=cfg.logging.level,
    )
    install_excepthook(logger=rank_logger)

    try:
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

        psf_path, crd_path = find_system_files(cfg)
        system.read_psf(psf_path)
        system.read_coor(crd_path)

        ctx = build_run_context(
            cfg,
            rank=actual_rank,
            comm=comm,
            gpu_id=gpu_id_for_rank(comm),
        )
        hooks = SegmentLimitHooks(max_segments=cfg.end)
        loop = SimulationLoop(ctx, hooks)
        loop.run()
    finally:
        mpi_finalize()


def build_run_context(
    cfg: NativeRuntimeConfig,
    *,
    rank: int,
    comm: Any | None,
    gpu_id: int,
) -> RunContext:
    alf_info = load_alf_info(cfg)
    nblocks = int(alf_info.get("nblocks", 2))
    raw_nsubs = tuple(int(value) for value in alf_info.get("nsubs", [1]))
    nsites = len(raw_nsubs)
    nsubsites = _expand_nsubsites(raw_nsubs)
    lambda_headers = tuple(f"LAMBDA {idx}" for idx in range(1, nblocks))
    replica_label = rank
    ph_values = cfg.ph_values or tuple(float(7.0) for _ in range(cfg.nreps))
    if len(ph_values) < cfg.nreps:
        ph_values = ph_values + tuple(ph_values[-1] for _ in range(cfg.nreps - len(ph_values)))
    rex_enabled = _rex_enabled(cfg.replica_exchange) and cfg.nreps > 1

    return RunContext(
        run_dir=cfg.run_dir,
        rank=rank,
        replica_label=replica_label,
        ph=ph_values[replica_label] if replica_label < len(ph_values) else ph_values[0],
        gpu_id=gpu_id,
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
        ldin_blocks=tuple(range(1, nblocks)),
        rex_enabled=rex_enabled,
        replica_ph_values=ph_values,
        rex_signs=tuple(1.0 for _ in lambda_headers),
        rex_exchange_every_segments=_rex_interval(cfg.replica_exchange),
        comm=comm,
        simulation_name=cfg.run_dir.name,
        walltime_safety_factor=cfg.walltime_safety_factor,
    )


def find_system_files(cfg: NativeRuntimeConfig) -> tuple[Path, Path]:
    candidates = (
        (cfg.run_dir / "solvated.psf", cfg.run_dir / "solvated.crd"),
        (cfg.run_dir / "prep" / "system_hmr.psf", cfg.run_dir / "prep" / "system_hmr.crd"),
        (cfg.run_dir / "prep" / "system.psf", cfg.run_dir / "prep" / "system.crd"),
    )
    for psf_path, crd_path in candidates:
        if psf_path.exists() and crd_path.exists():
            return psf_path, crd_path
    searched = ", ".join(f"{psf}/{crd}" for psf, crd in candidates)
    raise FileNotFoundError(f"Could not find initialized PSF/CRD pair; searched {searched}")


def load_alf_info(cfg: NativeRuntimeConfig) -> dict[str, Any]:
    path = cfg.run_dir / "prep" / "alf_info.py"
    if not path.exists():
        return {"name": cfg.run_dir.name, "nblocks": 2, "nsubs": [1], "nreps": cfg.nreps}
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
        values.extend(site_idx for _ in range(max(0, count - 1)))
    if len(values) == 1:
        values.append(1)
    return tuple(values)


def _rex_enabled(replica_exchange: Any) -> bool:
    if isinstance(replica_exchange, dict):
        return bool(replica_exchange.get("enabled", False))
    if isinstance(replica_exchange, bool):
        return replica_exchange
    return bool(getattr(replica_exchange, "enabled", False))


def _rex_interval(replica_exchange: Any) -> int:
    if isinstance(replica_exchange, dict):
        return int(replica_exchange.get("exchange_every_segments", 1))
    return int(getattr(replica_exchange, "exchange_every_segments", 1) or 1)


def _abort(comm, code: int) -> None:
    abort = getattr(comm, "Abort", None)
    if abort is not None:
        abort(code)
    raise SystemExit(code)


def register(app: typer.Typer) -> None:
    @app.command("run")
    def _cmd(
        config: Path = typer.Option(..., "-c", "--config", exists=True),
    ) -> None:
        run_simulation(config_path=config)
