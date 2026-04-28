"""Native startup minimization helpers."""

from __future__ import annotations

import logging

from cphmd.native.errors import SystemLoadError, wrap_exception
from cphmd.simulation.context import LoopState, RunContext

_STARTUP_MINIMIZATION_RUNS = 5
_SHAKE_KWARGS = {"fast": True, "bonh": True, "params": True, "tol": 1.0e-7}
_MINIMIZATION_KWARGS = {"step": 0.005, "tolenr": 1.0e-3, "tolgrd": 1.0e-3}
_STARTUP_MINIMIZATION_STEPS = 1000
logger = logging.getLogger(__name__)


def minimize_startup(
    ctx: RunContext,
    state: LoopState,
    *,
    dynamics_backend: DynamicsBackend | str | None = None,
) -> bool:
    """Run the legacy-compatible early CPU SD minimization before native dynamics."""

    run_limit = int(getattr(ctx, "startup_minimization_segments", _STARTUP_MINIMIZATION_RUNS))
    if run_limit <= 0 or state.run_idx >= run_limit:
        return False

    from cphmd.native import system

    min_crd = ctx.run_dir / "prep" / "system_min.crd"
    if min_crd.exists() and _matches_loaded_topology(min_crd):
        system.read_coor(min_crd)
        _shake_on(system)
        return False
    if min_crd.exists() and ctx.rank == 0:
        logger.warning("Ignoring stale minimized coordinates at %s; regenerating them", min_crd)

    try:
        if ctx.rank == 0:
            min_crd.parent.mkdir(parents=True, exist_ok=True)
            _run_cpu_startup_minimization(system)
            system.write_coor(min_crd)
        _barrier(ctx)
        if ctx.rank != 0:
            system.read_coor(min_crd)
            _shake_on(system)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "running startup minimization") from exc
    return False


def _run_cpu_startup_minimization(system) -> None:
    _shake_on(system)
    system.minimize_sd(
        nsteps=_STARTUP_MINIMIZATION_STEPS,
        nprint=50,
        **_MINIMIZATION_KWARGS,
    )


def _shake_on(system) -> None:
    system.shake_on(**_SHAKE_KWARGS)


def _barrier(ctx: RunContext) -> None:
    comm = ctx.comm
    if comm is None:
        return
    barrier = getattr(comm, "Barrier", None) or getattr(comm, "barrier", None)
    if barrier is not None:
        barrier()


def _matches_loaded_topology(min_crd) -> bool:
    from cphmd.native import system

    try:
        snapshot = system.get_topology_snapshot()
        if not snapshot.atoms:
            return False
        expected = tuple(
            (atom.segid.strip(), str(atom.resid), atom.resname.strip(), atom.atom_name.strip())
            for atom in snapshot.atoms
        )
        actual = _read_crd_signature(min_crd)
    except Exception:
        return False
    return actual == expected


def _read_crd_signature(path) -> tuple[tuple[str, str, str, str], ...]:
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("*"):
            break
        index += 1
    if index >= len(lines):
        raise ValueError(f"{path} is missing a CRD header")
    header = lines[index].split()
    if not header:
        raise ValueError(f"{path} has an empty CRD header")
    natom = int(header[0])
    index += 1

    signature: list[tuple[str, str, str, str]] = []
    while index < len(lines) and len(signature) < natom:
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("*"):
            continue
        fields = stripped.split()
        if len(fields) < 9:
            raise ValueError(f"{path} has a malformed CRD atom line")
        signature.append(
            (fields[7].strip(), fields[8].strip(), fields[2].strip(), fields[3].strip())
        )
    if len(signature) != natom:
        raise ValueError(f"{path} ended before all CRD atoms were read")
    return tuple(signature)
