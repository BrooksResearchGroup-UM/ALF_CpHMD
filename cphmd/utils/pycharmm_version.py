"""Helpers for discovering and validating pyCHARMM versions."""

from __future__ import annotations

import ast
import sys
from importlib import metadata, util
from pathlib import Path

from packaging.version import InvalidVersion, Version


def read_pycharmm_source_version() -> str | None:
    """Read ``pycharmm.__version__`` from source without importing pycharmm."""
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


def resolve_pycharmm_version_string() -> str | None:
    """Best-effort pyCHARMM version discovery without importing pyCHARMM."""
    pycharmm = sys.modules.get("pycharmm")
    version_str = getattr(pycharmm, "__version__", None) if pycharmm is not None else None
    if version_str is None:
        version_str = read_pycharmm_source_version()
    if version_str is None:
        try:
            version_str = metadata.version("pycharmm")
        except metadata.PackageNotFoundError:
            return None
    return version_str


def validate_pycharmm_version(minimum_version: str) -> tuple[Version | None, str | None]:
    """Return a usable version or an advisory message when validation fails."""
    version_str = resolve_pycharmm_version_string()
    if version_str is None:
        return (
            None,
            f"cphmd.native requires pyCHARMM >= {minimum_version}, "
            "but pyCHARMM is not installed or does not expose __version__.",
        )

    try:
        version = Version(version_str)
    except InvalidVersion:
        return (
            None,
            f"cphmd.native requires pyCHARMM >= {minimum_version}, "
            f"but the installed pycharmm version {version_str!r} is invalid.",
        )

    minimum = Version(minimum_version)
    if version < minimum:
        return (
            None,
            f"cphmd.native requires pyCHARMM >= {minimum_version}, but found {version_str}.",
        )

    return version, None
