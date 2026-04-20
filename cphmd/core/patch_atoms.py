"""Deprecated compatibility shim for legacy patch atom parsing."""

from __future__ import annotations

from cphmd.core import _stub_class, _stub_function

PatchAtomInfo = _stub_class("PatchAtomInfo", "cphmd.core.patch_spec.PatchSpec")
get_patch_atoms = _stub_function("get_patch_atoms", "cphmd.core.patch_spec")
parse_patch_atoms_from_str = _stub_function(
    "parse_patch_atoms_from_str", "cphmd.core.patch_spec"
)
parse_patch_atoms_from_rtf = _stub_function(
    "parse_patch_atoms_from_rtf", "cphmd.core.patch_spec"
)
load_custom_patches = _stub_function("load_custom_patches", "cphmd.core.patch_spec")
get_atoms_for_residue_type = _stub_function(
    "get_atoms_for_residue_type", "cphmd.core.patch_spec"
)

__all__ = [
    "PatchAtomInfo",
    "get_patch_atoms",
    "parse_patch_atoms_from_str",
    "parse_patch_atoms_from_rtf",
    "load_custom_patches",
    "get_atoms_for_residue_type",
]
