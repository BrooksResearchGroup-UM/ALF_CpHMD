"""cphmd.training -- ALF cycle, convergence, phase switching, production hooks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ALFConfig": ("cphmd.training.config", "ALFConfig"),
    "ALFReplicaExchangeConfig": ("cphmd.training.config", "ALFReplicaExchangeConfig"),
    "ALFCycleRunner": ("cphmd.training.cycle", "ALFCycleRunner"),
    "ALFHooks": ("cphmd.training.alf_hooks", "ALFHooks"),
    "ALFTrainingConfig": ("cphmd.training.alf_hooks", "ALFTrainingConfig"),
    "BiasRebuilder": ("cphmd.training.bias_rebuilder", "BiasRebuilder"),
    "BiasSnapshot": ("cphmd.training.bias_snapshot", "BiasSnapshot"),
    "LDBVTerm": ("cphmd.training.bias_snapshot", "LDBVTerm"),
    "NativeALFAnalyzer": ("cphmd.training.native_analyzer", "NativeALFAnalyzer"),
    "ProductionConfig": ("cphmd.training.production_hooks", "ProductionConfig"),
    "ProductionHooks": ("cphmd.training.production_hooks", "ProductionHooks"),
    "SegmentCache": ("cphmd.training.segment_cache", "SegmentCache"),
    "write_production_bias_file": (
        "cphmd.training.production_hooks",
        "write_production_bias_file",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
