"""Deprecated compatibility shim for fixed-bias production helpers."""

from __future__ import annotations

from cphmd.core import (
    build_nsubsites_str,
    build_parquet_metadata,
    find_restart_for_chunk,
    find_resume_point,
)
from cphmd.training.production_hooks import (
    ProductionConfig,
    ProductionConfigError,
    ProductionHooks,
)

ProductionRunner = ProductionHooks

__all__ = [
    "ProductionConfig",
    "ProductionConfigError",
    "ProductionHooks",
    "ProductionRunner",
    "build_nsubsites_str",
    "build_parquet_metadata",
    "find_resume_point",
    "find_restart_for_chunk",
]
