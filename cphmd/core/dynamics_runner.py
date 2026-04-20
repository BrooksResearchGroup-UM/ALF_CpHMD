"""Deprecated dynamics runner compatibility shim."""

from __future__ import annotations

import warnings
from importlib import import_module

__all__ = ["DynamicsRunner"]  # noqa: F822


def _warn(name: str, replacement: str) -> None:
    warnings.warn(
        f"cphmd.core.dynamics_runner.{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


class _DynamicsRunner:
    """Deprecated runner stub.

    The old pyCHARMM-bound runner has been replaced by
    ``cphmd.simulation.loop.SimulationLoop`` and
    ``cphmd.native.dynamics.run_segment``. This class exists so import
    sites fail only when they actually try to construct or use the old
    runner.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DynamicsRunner is deprecated; use cphmd.simulation.loop.SimulationLoop "
            "with cphmd.native.dynamics.run_segment."
        )


_EXPORTS = {
    "DynamicsRunner": (
        "cphmd.core.dynamics_runner",
        "_DynamicsRunner",
        "cphmd.simulation.loop.SimulationLoop",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name, replacement = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    _warn(name, replacement)
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
