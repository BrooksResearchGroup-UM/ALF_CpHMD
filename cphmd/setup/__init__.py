"""Setup module for CpHMD system preparation."""

from .legacy_convert import LegacyConvertConfig, convert_legacy_system
from .prepare_pdb import PreparePDBConfig, prepare_pdb_system
from .solvate import SolvationConfig, solvate_system


def create_amino_acid(*args, **kwargs):
    from .create_aa import create_amino_acid as _create_amino_acid

    return _create_amino_acid(*args, **kwargs)


def create_nucleic_acid(*args, **kwargs):
    from .create_aa import create_nucleic_acid as _create_nucleic_acid

    return _create_nucleic_acid(*args, **kwargs)


def create_all_templates(*args, **kwargs):
    from .create_aa import create_all_templates as _create_all_templates

    return _create_all_templates(*args, **kwargs)

__all__ = [
    "create_amino_acid",
    "create_nucleic_acid",
    "create_all_templates",
    "PreparePDBConfig",
    "prepare_pdb_system",
    "SolvationConfig",
    "solvate_system",
    "LegacyConvertConfig",
    "convert_legacy_system",
]
