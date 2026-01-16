"""Preset bias parameters for titratable amino acids.

This module provides converged bias parameters from legacy cubic simulations
that can be used to quickly set up CpHMD without running full phase 1 ALF,
or as starting points for bias refinement.

IMPORTANT: All presets were calculated using FNEX 5.5 implicit constraint.
If you use a different constraint type (FPIE) or different FNEX value,
these biases may not be optimal and you should run ALF to converge new biases.

Common settings for cubic presets:
    - Implicit constraint: FNEX 5.5
    - RMLA: bond theta impr
    - Cutoffs: cutnb=14.0, ctofnb=12.0, ctonnb=10.0
    - Temperature: 298.15 K
    - Restraints: SCAT (except noe_h)
    - Hydrogen restraints: OFF (except noe_h)

Available configurations:
    Cubic (SCAT, no H restraints):
    - "pmenn_vswitch" (default): PME NN + vswitch
    - "pmenn_vfswitch": PME NN + vfswitch
    - "fshift_vswitch": FSHIFT + vswitch
    - "fshift_vfswitch": FSHIFT + vfswitch
    Residues: ASP, GLU, HSP, LYS, TYR

    NOE:
    - "noe_noh": PME NN + vswitch, no H restraints
      Residues: ASP, GLU, HSP, LYS, TYR
    - "noe_h": PME NN + vswitch, with H restraints
      Residues: ASP, GLU, HSP, LYS, TYR, ARG, CYS, SER

Usage:
    >>> from cphmd.presets import list_configs, get_preset_biases
    >>> list_configs()
    ['fshift_vfswitch', 'fshift_vswitch', 'noe_h', 'pmenn_vfswitch', 'pmenn_vswitch', 'scat_noh']
    >>> biases = get_preset_biases("ASP")  # uses default (pmenn_vswitch)
    >>> biases = get_preset_biases("ASP", config="fshift_vswitch")
"""

from .biases import (
    PRESET_CONFIG,
    PRESET_BIASES,
    PRESET_CONFIGS,
    get_preset_biases,
    list_presets,
    list_configs,
    get_config,
    write_preset_variables,
    get_bias_params_only,
    DEFAULT_CONFIG,
)

__all__ = [
    # Legacy (default config)
    "PRESET_CONFIG",
    "PRESET_BIASES",
    # Multi-config API
    "PRESET_CONFIGS",
    "DEFAULT_CONFIG",
    "list_configs",
    "get_config",
    "list_presets",
    "get_preset_biases",
    "get_bias_params_only",
    "write_preset_variables",
]
