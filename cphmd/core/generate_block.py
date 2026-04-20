"""Deprecated compatibility shim for legacy BLOCK file generation."""

from __future__ import annotations

from cphmd.core import (
    BlockGeneratorConfig,
    BlockGeneratorResult,
    generate_block_files,
)


def _wrap_charmm_line(line: str, max_len: int = 80) -> str:
    if len(line) <= max_len:
        return line
    return line[:max_len]


def _wrap_cats_line(atom: str, segid: str, resid: str, resname_clause: str) -> str:
    return (
        f" cats sele segid {segid} .and. resid {resid} .and. "
        f"({resname_clause}) .and. type {atom} end"
    )


__all__ = [
    "BlockGeneratorConfig",
    "BlockGeneratorResult",
    "generate_block_files",
    "_wrap_charmm_line",
    "_wrap_cats_line",
]
