"""Deprecated compatibility shim for the legacy ALF runner module."""

from __future__ import annotations

from cphmd.core import ALFSimulation, run_alf_simulation
from cphmd.training.config import ALFConfig, ALFReplicaExchangeConfig

__all__ = [
    "ALFConfig",
    "ALFReplicaExchangeConfig",
    "ALFSimulation",
    "run_alf_simulation",
]
