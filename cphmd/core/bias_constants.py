"""Bias potential shape constants for LDBV classes.

Shape constants for the bias basis functions used in CHARMM BLOCK LDBV commands.
These are independent of the FNEX softmax constraint parameter (c=5.5).

CHARMM BLOCK LDBV format: LDBV idx block1 block2 class REF amplitude flag

    Class 8  (s-term, endpoint sigmoid): REF = CHI_OFFSET = 0.017
    Class 10 (x-term, exponential skew): REF = OMEGA_DECAY = -5.56

Constants stored with the sign CHARMM receives:
    OMEGA_DECAY = -5.56  (negative, written directly as LDBV class 10 REF)
    CHI_OFFSET  =  0.017 (positive, written directly as LDBV class 8 REF)
    OMEGA_SCALE = -1/OMEGA_DECAY = 1/5.56  (positive, for CUDA kernels)

Provenance (Hayes & Brooks, J. Phys. Chem. B 2017; Hayes et al. 2018):
    - 0.017: empirically optimized s-term offset for FNEX=5.5
    - 5.56: empirically optimized x-term decay for FNEX=5.5
    - 0.012: t/u-term offsets from bcxstu2026 (Hayes et al. 2024)

These values are NOT derived from FNEX. They are independent empirical
parameters that were optimized alongside FNEX=5.5. For other FNEX values,
shape constants may need re-optimization.
"""

from __future__ import annotations

from typing import NamedTuple

DEFAULT_FNEX = 5.5


class BiasConstants(NamedTuple):
    """LDBV shape constants for bias potential basis functions.

    Attributes:
        fnex: FNEX softmax constraint parameter (for BLOCK FNEX and G_imp).
        omega_decay: x-term exponential decay, LDBV class 10 REF (negative).
        chi_offset: s-term sigmoid offset, LDBV class 8 REF (positive).
        omega_scale: Reciprocal magnitude of omega_decay (-1/omega_decay),
            used in CUDA kernels as the exponential scale parameter.
        chi_offset_t: t-term reverse sigmoid offset.
        chi_offset_u: u-term quadratic sigmoid offset.
    """

    fnex: float
    omega_decay: float
    chi_offset: float
    omega_scale: float
    chi_offset_t: float
    chi_offset_u: float


def derive_bias_constants(
    fnex: float = DEFAULT_FNEX,
    *,
    chi_offset: float = 0.017,
    omega_decay: float = -5.56,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
) -> BiasConstants:
    """Build bias constants from explicit values.

    Shape constants are independent of FNEX. They are stored alongside
    fnex for convenience but are NOT derived from it.

    Args:
        fnex: FNEX softmax constraint parameter (default 5.5).
        chi_offset: LDBV class 8 REF (default 0.017).
        omega_decay: LDBV class 10 REF, negative (default -5.56).
        chi_offset_t: t-term reverse sigmoid offset (default 0.012).
        chi_offset_u: u-term quadratic sigmoid offset (default 0.012).

    Returns:
        BiasConstants with all values.
    """
    return BiasConstants(
        fnex=fnex,
        omega_decay=omega_decay,
        chi_offset=chi_offset,
        omega_scale=-1.0 / omega_decay,
        chi_offset_t=chi_offset_t,
        chi_offset_u=chi_offset_u,
    )


# Module-level defaults
_DEFAULT = derive_bias_constants()
OMEGA_DECAY = _DEFAULT.omega_decay
CHI_OFFSET = _DEFAULT.chi_offset
OMEGA_SCALE = _DEFAULT.omega_scale
CHI_OFFSET_T = _DEFAULT.chi_offset_t
CHI_OFFSET_U = _DEFAULT.chi_offset_u
