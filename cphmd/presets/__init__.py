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

Available configurations:
    PME EX (Ewald eXact):
    - "pme_ex_vswitch": PME EX + vswitch
    - "pme_ex_vfswitch": PME EX + vfswitch

    PME NN (Nearest Neighbor):
    - "pme_nn_vswitch" (default): PME NN + vswitch
    - "pme_nn_vfswitch": PME NN + vfswitch

    PME ON:
    - "pme_on_vswitch": PME ON + vswitch
    - "pme_on_vfswitch": PME ON + vfswitch

    FSHIFT:
    - "fshift_vswitch": FSHIFT + vswitch
    - "fshift_vfswitch": FSHIFT + vfswitch

    FSWITCH:
    - "fswitch_vswitch": FSWITCH + vswitch
    - "fswitch_vfswitch": FSWITCH + vfswitch

    Residues: ASP, GLU, HSP, LYS, TYR

Usage:
    >>> from cphmd.presets import list_configs, get_preset_biases
    >>> list_configs()
    ['fshift_vfswitch', 'fshift_vswitch', 'fswitch_vfswitch', 'fswitch_vswitch', 'pme_ex_vfswitch', 'pme_ex_vswitch', 'pme_nn_vfswitch', 'pme_nn_vswitch', 'pme_on_vfswitch', 'pme_on_vswitch']
    >>> biases = get_preset_biases("ASP")  # uses default (pme_nn_vswitch)
    >>> biases = get_preset_biases("ASP", config="pme_on_vswitch")
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
