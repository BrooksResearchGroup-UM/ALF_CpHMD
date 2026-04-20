"""Native startup minimization helpers."""

from __future__ import annotations

from cphmd.native.errors import SystemLoadError, wrap_exception
from cphmd.simulation.context import LoopState, RunContext

_STARTUP_MINIMIZATION_RUNS = 5


def minimize_startup(ctx: RunContext, state: LoopState, *, use_blade: bool) -> None:
    """Run the legacy-compatible early SD minimization before native dynamics."""

    run_limit = int(getattr(ctx, "startup_minimization_segments", _STARTUP_MINIMIZATION_RUNS))
    if run_limit <= 0 or state.run_idx >= run_limit:
        return

    from cphmd.native import system

    min_crd = ctx.run_dir / "prep" / "system_min.crd"
    if min_crd.exists():
        system.read_coor(min_crd)
        return

    try:
        if ctx.rank == 0:
            min_crd.parent.mkdir(parents=True, exist_ok=True)
            system.shake_on(fast=True, bonh=True, params=True, tol=1.0e-7)
            system.minimize_sd(
                nsteps=2000 if use_blade else 100,
                nprint=200 if use_blade else 50,
                step=0.005,
                tolenr=1.0e-3,
                tolgrd=1.0e-3,
            )
            system.write_coor(min_crd)
        _barrier(ctx)
        if ctx.rank != 0:
            system.read_coor(min_crd)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "running startup minimization") from exc


def _barrier(ctx: RunContext) -> None:
    comm = ctx.comm
    if comm is None:
        return
    barrier = getattr(comm, "Barrier", None) or getattr(comm, "barrier", None)
    if barrier is not None:
        barrier()
