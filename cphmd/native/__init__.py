"""Native pyCHARMM boundary for CpHMD."""

import warnings
from importlib import import_module

from packaging.version import Version

from cphmd.native.types import (
    AtomRecord,
    AtomSelection,
    CellParameters,
    ResidueRecord,
    TopologySnapshot,
)
from cphmd.utils.pycharmm_version import validate_pycharmm_version

PYCHARMM_MIN_VERSION = "0.5.1"
PYCHARMM_VERSION, PYCHARMM_REQUIREMENT_ERROR = validate_pycharmm_version(PYCHARMM_MIN_VERSION)
if PYCHARMM_REQUIREMENT_ERROR is not None:
    warnings.warn(PYCHARMM_REQUIREMENT_ERROR, RuntimeWarning, stacklevel=2)


def require_pycharmm_runtime() -> Version:
    """Raise a clear error when pyCHARMM does not satisfy the native floor."""
    if PYCHARMM_REQUIREMENT_ERROR is not None:
        raise RuntimeError(PYCHARMM_REQUIREMENT_ERROR)
    assert PYCHARMM_VERSION is not None
    return PYCHARMM_VERSION


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
    "PYCHARMM_REQUIREMENT_ERROR",
    "ResidueRecord",
    "TopologySnapshot",
    "require_pycharmm_runtime",
    "system",
]
