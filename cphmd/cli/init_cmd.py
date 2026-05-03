"""Native ``cphmd init`` command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from cphmd.cli.run_cmd import _sync_run_prep_metadata, apply_run_dir_override, build_run_context
from cphmd.config.loader import NativeRuntimeConfig, load_config
from cphmd.simulation.checkpoint import CheckpointManager
from cphmd.simulation.context import LoopState
from cphmd.utils.seeds import get_rng_state, make_rng

RNG_INIT_DOMAINS = ("rex", "dynamics", "velocity_init", "gimp_mc")


def run_init(
    *,
    config_path: Path,
    run_dir: Path | None = None,
    force: bool = False,
    reinit_build: bool = False,
    force_reinit: bool = False,
) -> None:
    cfg = load_config(config_path)
    if run_dir is not None:
        cfg = apply_run_dir_override(cfg, config_path, run_dir)
    state_dir = cfg.run_dir / "state"
    init_marker = state_dir / "initialized.json"
    tmp_dir = cfg.run_dir / ".init_tmp"
    status_summary = cfg.run_dir / "status_summary.json"
    has_checkpoints = any(cfg.run_dir.rglob("res/rep*/checkpoint.json"))

    if init_marker.exists() and not (force or reinit_build):
        raise RuntimeError(f"{init_marker} already exists")
    if force and has_checkpoints:
        raise RuntimeError("cphmd init --force refuses to clobber active checkpoints")
    if reinit_build and has_checkpoints and not force_reinit:
        typer.echo(
            "ERROR: checkpoint present; --reinit-build requires --force-reinit",
            err=True,
        )
        raise SystemExit(2)
    if tmp_dir.exists():
        if not force_reinit:
            typer.echo(
                f"ERROR: stale {tmp_dir} from interrupted init; remove it or pass --force-reinit",
                err=True,
            )
            raise SystemExit(2)
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        status_summary.unlink(missing_ok=True)
        _maybe_run_setup_pipeline(cfg, reinit_build=reinit_build)
        _sync_run_prep_metadata(cfg)
        state_dir.mkdir(parents=True, exist_ok=True)
        _write_initial_checkpoints(cfg)
        _write_rng_states(cfg)
        init_marker.write_text(
            json.dumps(
                {
                    "initialized": True,
                    "master_seed": cfg.master_seed,
                    "nreps": cfg.nreps,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except BaseException:
        raise
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _maybe_run_setup_pipeline(cfg: NativeRuntimeConfig, *, reinit_build: bool = False) -> None:
    prep_dir = cfg.input_folder / "prep"
    prep_ready = (prep_dir / "patches.dat").exists() or (prep_dir / "alf_info.py").exists()
    if cfg.raw.get("solvation") and (reinit_build or not prep_ready):
        from cphmd.config.loader import _run_solvate

        _run_solvate(cfg.config_path)
    if cfg.raw.get("patch") and (reinit_build or not prep_ready):
        from cphmd.config.loader import _run_patch

        _run_patch(cfg.config_path)


def _write_initial_checkpoints(cfg: NativeRuntimeConfig) -> None:
    native_modules = _checkpoint_native_modules()
    for rank in range(cfg.nreps):
        ctx = build_run_context(cfg, rank=rank, comm=None, gpu_id=rank)
        checkpoint = CheckpointManager(ctx, native_modules=native_modules)
        checkpoint.write(
            LoopState(replica_label=rank),
            rng_state=_initial_rng_state(cfg, rank),
        )


def _checkpoint_native_modules():
    from cphmd.simulation.loop import default_native_modules

    return default_native_modules()


def _write_rng_states(cfg: NativeRuntimeConfig) -> None:
    payload = {f"rep{rank:02d}": _initial_rng_state(cfg, rank) for rank in range(cfg.nreps)}
    (cfg.run_dir / "rng_states.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def _initial_rng_state(cfg: NativeRuntimeConfig, rank: int) -> dict:
    return {
        domain: get_rng_state(make_rng(cfg.master_seed, domain, rank))
        for domain in RNG_INIT_DOMAINS
    }


def register(app: typer.Typer) -> None:
    @app.command("init")
    def _cmd(
        config: Path = typer.Option(..., "-c", "--config", exists=True),
        run_dir: Path | None = typer.Option(None, "--run-dir", "-r"),
        force: bool = typer.Option(False, "--force"),
        reinit_build: bool = typer.Option(False, "--reinit-build"),
        force_reinit: bool = typer.Option(False, "--force-reinit"),
    ) -> None:
        run_init(
            config_path=config,
            run_dir=run_dir,
            force=force,
            reinit_build=reinit_build,
            force_reinit=force_reinit,
        )
