"""MPI helpers for the native pyCHARMM boundary."""

from __future__ import annotations

import importlib
import os

from cphmd.native.errors import MPIInitError, wrap_exception


def init():
    """Return ``mpi4py.MPI.COMM_WORLD`` or raise ``MPIInitError``."""
    try:
        mpi_module = importlib.import_module("mpi4py.MPI")
        return mpi_module.COMM_WORLD
    except Exception as exc:  # pragma: no cover - exercised via tests
        raise wrap_exception(exc, MPIInitError, "Failed to initialize MPI") from exc


def rank(comm) -> int:
    """Return the rank for ``comm``."""
    return comm.Get_rank()


def size(comm) -> int:
    """Return the communicator size for ``comm``."""
    return comm.Get_size()


def gpu_id_for_rank(comm) -> int:
    """Return the visible CUDA ordinal for this rank."""
    visible_devices = _visible_cuda_devices()
    local_rank = _local_rank(comm)

    if local_rank is None:
        local_rank = _fallback_local_rank(comm, visible_devices)

    if visible_devices and local_rank >= len(visible_devices):
        raise MPIInitError(
            "More local ranks than visible CUDA devices: "
            f"local rank {local_rank} cannot map to a visible CUDA ordinal "
            f"because CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"exposes {len(visible_devices)} device(s)."
        )

    return local_rank


def _local_rank(comm) -> int | None:
    for env_var in (
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
    ):
        value = os.environ.get(env_var)
        if value is not None:
            try:
                return int(value)
            except ValueError as exc:
                raise MPIInitError(f"Invalid {env_var} value {value!r}") from exc
    return None


def _fallback_local_rank(comm, visible_devices: list[str]) -> int:
    if visible_devices:
        return rank(comm)

    gpus_per_node = os.environ.get("CPHMD_GPUS_PER_NODE")
    if gpus_per_node is None:
        raise MPIInitError(
            "Cannot map MPI rank to GPU: set CUDA_VISIBLE_DEVICES, a local-rank "
            "environment variable such as SLURM_LOCALID, or CPHMD_GPUS_PER_NODE."
        )

    try:
        per_node = int(gpus_per_node)
    except ValueError as exc:
        raise MPIInitError(f"Invalid CPHMD_GPUS_PER_NODE value {gpus_per_node!r}") from exc

    if per_node <= 0:
        raise MPIInitError("CPHMD_GPUS_PER_NODE must be positive")

    return rank(comm) % per_node


def _visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw.strip():
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]
