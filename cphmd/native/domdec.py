from __future__ import annotations

from cphmd.native.errors import CpHMDNativeError, wrap_exception
from cphmd.simulation.backends import DomdecConfig


def enable(*, gpu: bool, gpu_id: int | None, config: DomdecConfig) -> None:
    try:
        import pycharmm.domdec as domdec
        import pycharmm.mpi as charmm_mpi

        charmm_mpi.use_self()
        _assert_charmm_comm_self(charmm_mpi)

        kwargs = config.to_kwargs()
        kwargs["gpu"] = gpu
        if gpu_id is not None:
            kwargs["gpuid"] = int(gpu_id)
        domdec.enable(**kwargs)
        _assert_charmm_comm_self(charmm_mpi)
    except Exception as exc:
        raise wrap_exception(exc, CpHMDNativeError, "enabling DOMDEC") from exc


def energy(*, gpu: bool) -> None:
    try:
        import pycharmm.domdec as domdec

        domdec.energy(gpu=gpu)
    except Exception as exc:
        raise wrap_exception(exc, CpHMDNativeError, "running DOMDEC energy") from exc


def disable() -> None:
    try:
        import pycharmm.domdec as domdec

        domdec.disable()
    except Exception as exc:
        raise wrap_exception(exc, CpHMDNativeError, "disabling DOMDEC") from exc


def _assert_charmm_comm_self(charmm_mpi) -> None:
    current_size = getattr(charmm_mpi, "current_size", None)
    if callable(current_size) and int(current_size()) != 1:
        raise RuntimeError("CHARMM/DOMDEC communicator must be MPI_COMM_SELF")
