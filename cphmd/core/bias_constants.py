"""Bias potential constants derived from the FNEX parameter.

The FNEX parameter controls the implicit softmax constraint in lambda dynamics.
The bias potential basis functions (x-term exponential and s-term sigmoid) have
shape parameters that are physically determined by FNEX:

    OMEGA_DECAY = FNEX           (x-term exponential decay rate)
    CHI_OFFSET  = 4 * exp(-FNEX) (s-term sigmoid offset)
    OMEGA_SCALE   = 1 / FNEX      (reciprocal for CUDA kernels)

For the default FNEX=5.5 (matching CHARMM BLOCK FNEX):
    OMEGA_DECAY = 5.5, CHI_OFFSET = 0.01634, OMEGA_SCALE = 0.18182

Historical note: These were previously hardcoded as 5.56, 0.017, and 0.18 —
approximations that drifted from the exact FNEX-derived values.

FNEX should NOT scale with N (number of subsites) because:
1. Per-pair barrier height is FNEX-controlled, independent of N
2. Larger N gives more transition partners, which accelerates mixing
3. G_imp entropy correction already handles N-dependence
4. All precomputed presets assume FNEX=5.5
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

DEFAULT_FNEX = 5.5


class BiasConstants(NamedTuple):
    """Constants derived from a single FNEX value.

    Attributes:
        fnex: The FNEX softmax constraint parameter.
        omega_decay: Exponential decay for x-term LDBV class 10 REF.
        chi_offset: Sigmoid offset for s-term LDBV class 8 REF.
        omega_scale: Reciprocal of omega_decay (1/FNEX), used in CUDA.
    """

    fnex: float
    omega_decay: float
    chi_offset: float
    omega_scale: float


def derive_bias_constants(
    fnex: float = DEFAULT_FNEX,
    *,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
) -> BiasConstants:
    """Derive bias potential constants from a FNEX value.

    Args:
        fnex: The FNEX softmax constraint parameter (default 5.5).
        chi_offset: Override for s-term sigmoid offset. If None, derived
            as 4*exp(-fnex). Set to 0.017 to match legacy ALF behavior.
        omega_decay: Override for x-term exponential decay. If None, derived
            as fnex. Set to 5.56 to match legacy ALF behavior.

    Returns:
        BiasConstants with all derived values.
    """
    actual_omega = omega_decay if omega_decay is not None else fnex
    actual_chi = chi_offset if chi_offset is not None else 4.0 * np.exp(-fnex)
    return BiasConstants(
        fnex=fnex,
        omega_decay=actual_omega,
        chi_offset=actual_chi,
        omega_scale=1.0 / actual_omega,
    )


# Module-level defaults for backward compatibility
_DEFAULT = derive_bias_constants()
OMEGA_DECAY = _DEFAULT.omega_decay
CHI_OFFSET = _DEFAULT.chi_offset
OMEGA_SCALE = _DEFAULT.omega_scale
