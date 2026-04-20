"""Deprecated compatibility shim for legacy BLOCK command builders."""

from __future__ import annotations

from cphmd.core import BlockConfig, build_block_command, read_variable_file

__all__ = [
    "BlockConfig",
    "build_block_command",
    "read_variable_file",
]
