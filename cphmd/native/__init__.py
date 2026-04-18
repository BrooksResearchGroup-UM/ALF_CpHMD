"""Native pyCHARMM boundary for CpHMD."""

import sys
from importlib import import_module, metadata

from packaging.version import InvalidVersion, Version

from cphmd.native.types import AtomRecord, AtomSelection, CellParameters, TopologySnapshot

PYCHARMM_MIN_VERSION = "0.5.1"


def _require_pycharmm_version() -> Version:
    """Validate that the installed pyCHARMM distribution meets the floor."""
    try:
        version_str = metadata.version("pycharmm")
    except metadata.PackageNotFoundError:
        pycharmm = sys.modules.get("pycharmm")
        if pycharmm is None:
            raise RuntimeError(
                f"cphmd.native requires pyCHARMM >= {PYCHARMM_MIN_VERSION}, "
                "but the 'pycharmm' distribution is not installed."
            )
        version_str = getattr(pycharmm, "__version__", None)
        if version_str is None:
            raise RuntimeError(
                f"cphmd.native requires pyCHARMM >= {PYCHARMM_MIN_VERSION}, "
                "but pyCHARMM does not expose __version__."
            )

    try:
        version = Version(version_str)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"cphmd.native requires pyCHARMM >= {PYCHARMM_MIN_VERSION}, "
            f"but the installed pycharmm version {version_str!r} is invalid."
        ) from exc

    minimum = Version(PYCHARMM_MIN_VERSION)
    if version < minimum:
        raise RuntimeError(
            f"cphmd.native requires pyCHARMM >= {PYCHARMM_MIN_VERSION}, "
            f"but found {version_str}."
        )

    return version


PYCHARMM_VERSION = _require_pycharmm_version()


def __getattr__(name: str):
    if name == "system":
        module = import_module("cphmd.native.system")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AtomRecord",
    "AtomSelection",
    "CellParameters",
    "PYCHARMM_MIN_VERSION",
    "PYCHARMM_VERSION",
    "TopologySnapshot",
    "system",
]
