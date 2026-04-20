"""Native pyCHARMM boundary for CpHMD."""

import ast
import sys
from importlib import import_module, metadata, util
from pathlib import Path

from packaging.version import InvalidVersion, Version

from cphmd.native.types import (
    AtomRecord,
    AtomSelection,
    CellParameters,
    ResidueRecord,
    TopologySnapshot,
)

PYCHARMM_MIN_VERSION = "0.5.1"


def _read_pycharmm_source_version() -> str | None:
    """Read pyCHARMM __version__ from source without importing pyCHARMM."""
    try:
        spec = util.find_spec("pycharmm")
    except ValueError:
        return None
    if spec is None or spec.origin is None:
        return None

    try:
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _require_pycharmm_version() -> Version:
    """Validate that the installed pyCHARMM distribution meets the floor."""
    pycharmm = sys.modules.get("pycharmm")
    version_str = getattr(pycharmm, "__version__", None) if pycharmm is not None else None
    if version_str is None:
        version_str = _read_pycharmm_source_version()
    if version_str is None:
        try:
            version_str = metadata.version("pycharmm")
        except metadata.PackageNotFoundError:
            pass

        if version_str is None:
            raise RuntimeError(
                f"cphmd.native requires pyCHARMM >= {PYCHARMM_MIN_VERSION}, "
                "but pyCHARMM is not installed or does not expose __version__."
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
    "ResidueRecord",
    "TopologySnapshot",
    "system",
]
