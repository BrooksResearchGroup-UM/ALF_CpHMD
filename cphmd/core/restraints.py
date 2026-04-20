"""Deprecated compatibility shim for legacy restraint generators."""

from __future__ import annotations

from cphmd.core import generate_noe_restraints, generate_scat_restraints, write_restraint_file

__all__ = [
    "generate_scat_restraints",
    "generate_noe_restraints",
    "write_restraint_file",
]
