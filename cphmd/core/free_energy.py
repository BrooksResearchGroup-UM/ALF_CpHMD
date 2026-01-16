"""Free energy optimization for ALF bias updates.

This module reimplements ALF's GetFreeEnergy5 for computing optimal bias
parameter changes from WHAM output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .alf_utils import ALFInfo

logger = logging.getLogger(__name__)


@dataclass
class FreeEnergyResult:
    """Result of free energy optimization.

    Attributes:
        scaling: Scaling factor applied to bias changes (1.0 = converged).
        b_changes: Linear bias changes (1 x nblocks).
        c_changes: Quadratic bias changes (nblocks x nblocks).
        x_changes: Cross-term x bias changes (nblocks x nblocks).
        s_changes: Cross-term s bias changes (nblocks x nblocks).
    """

    scaling: float
    b_changes: np.ndarray
    c_changes: np.ndarray
    x_changes: np.ndarray
    s_changes: np.ndarray


def _solve_with_svd(C: np.ndarray, V: np.ndarray, n0: int) -> np.ndarray:
    """Solve C @ coeff = V using truncated SVD pseudoinverse.

    This is more robust than np.linalg.solve for ill-conditioned matrices.
    Near-zero singular values are truncated to prevent numerical instability.

    Args:
        C: Hessian matrix from WHAM.
        V: Gradient vector from WHAM.
        n0: Number of active parameters.

    Returns:
        Coefficient vector (same shape as V).
    """
    try:
        # Check conditioning first
        cond_num = np.linalg.cond(C)
        if cond_num > 1e10:
            logger.warning(
                f"C matrix is ill-conditioned (condition number: {cond_num:.2e}). "
                "Using truncated SVD for stability."
            )
    except (np.linalg.LinAlgError, ValueError):
        cond_num = np.inf
        logger.warning("Could not compute condition number - matrix may be singular")

    # Compute SVD: C = U @ diag(s) @ Vt
    try:
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed, using zero updates")
        return np.zeros_like(V)

    # Truncate small singular values to avoid division by near-zero
    # Threshold is relative to largest singular value
    if len(s) > 0 and s[0] > 0:
        threshold = 1e-10 * s[0]
    else:
        logger.warning("Matrix has no positive singular values, using zero updates")
        return np.zeros_like(V)

    # Count how many singular values we're keeping
    n_kept = np.sum(s > threshold)
    if n_kept < len(s):
        logger.info(
            f"Truncating {len(s) - n_kept} small singular values "
            f"(keeping {n_kept}/{len(s)})"
        )

    # Compute pseudoinverse: s_inv[i] = 1/s[i] if s[i] > threshold, else 0
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)

    # Solve: coeff = Vt.T @ diag(s_inv) @ U.T @ V
    coeff = Vt.T @ (s_inv * (U.T @ V))

    # Validate result
    if not np.all(np.isfinite(coeff)):
        logger.warning(
            "SVD solve produced NaN/Inf values. Using zero updates for this iteration."
        )
        return np.zeros_like(V)

    return coeff


def get_populations_from_lambda(
    analysis_dir: Path, nsubs: np.ndarray, ndupl: int = 1, lc: float = 0.8
) -> np.ndarray:
    """Compute state populations from Lambda trajectory files.

    Args:
        analysis_dir: Path to analysis directory containing data/ folder.
        nsubs: Array of substituent counts per site.
        ndupl: Number of independent trials (replicas).
        lc: Lambda cutoff for state assignment (state = first lambda > lc).

    Returns:
        Population fractions for each state (sums to 1 within each site).
        Shape: (nblocks,) where nblocks = sum(nsubs).
    """
    nblocks = int(np.sum(nsubs))
    counts = np.zeros(nblocks)
    total_frames = 0

    data_dir = analysis_dir / "data"

    for tag in range(ndupl):
        data_file = data_dir / f"Lambda.{tag}.0.dat"
        if not data_file.exists():
            continue

        data = np.loadtxt(data_file)
        total_frames += data.shape[0]

        # Process each frame
        for t in range(data.shape[0]):
            ibuff = 0
            for isite in range(len(nsubs)):
                # Find current state (first lambda > cutoff)
                for j in range(nsubs[isite]):
                    if data[t, j + ibuff] > lc:
                        counts[ibuff + j] += 1
                        break
                ibuff += nsubs[isite]

    # Convert to fractions
    if total_frames > 0:
        populations = counts / total_frames
    else:
        # Equal populations if no data
        populations = np.ones(nblocks) / nblocks

    return populations


def fallback_bias_update(
    populations: np.ndarray, temperature: float, max_change: float = 2.0
) -> np.ndarray:
    """Compute bias updates from populations when WHAM fails.

    Uses Boltzmann relation: ΔG = RT * ln(p_target / p_current)
    This provides a simple heuristic to push undersampled states toward
    uniform populations when the WHAM matrix is too ill-conditioned.

    Args:
        populations: Current population fractions for each state (should sum to 1).
        temperature: Simulation temperature in Kelvin.
        max_change: Maximum bias change per state (kcal/mol).

    Returns:
        Bias changes (delta_b) to add to current b parameters.
        First state is referenced to 0.
    """
    RT = 0.001987 * temperature  # kcal/mol
    n_states = len(populations)
    target_pop = 1.0 / n_states  # Uniform target

    # Normalize populations just in case
    populations = np.asarray(populations, dtype=float)
    pop_sum = np.sum(populations)
    if pop_sum > 0:
        populations = populations / pop_sum
    else:
        # All zero populations - can't do anything
        logger.warning("All populations are zero, cannot compute fallback update")
        return np.zeros(n_states)

    # Compute needed bias change using Boltzmann inversion
    # ΔG = RT * ln(p_target / p_current)
    # For undersampled states (p < target), this gives positive ΔG (raise bias)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_b = RT * np.log(target_pop / populations)

    # Handle edge cases (zero/inf)
    delta_b = np.nan_to_num(delta_b, nan=0.0, posinf=max_change, neginf=-max_change)

    # Clip to max change
    delta_b = np.clip(delta_b, -max_change, max_change)

    # Reference to first state (first state always 0)
    delta_b = delta_b - delta_b[0]

    logger.info(f"Fallback bias update: {delta_b}")
    return delta_b


def get_free_energy5(
    work_dir: str | Path,
    alf_info: "ALFInfo | dict",
    ms: int,
    msprof: int,
    analysis_idx: int,
) -> float:
    """Solve for optimal bias changes via matrix inversion.

    Performs matrix inversion to solve for optimal bias parameter changes
    based on WHAM output. The WHAM routine computes profiles and linear
    changes to those profiles in response to changes in bias parameters.

    This routine adds regularization to the Hessian matrix and inverts it
    to solve the linear equation dictating the optimal solution. Because
    this represents a linear approximation, there are caps on changes to
    any particular bias parameter.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        ms: Flag for intersite biases (0=none, 1=c/x/s, 2=just c).
        msprof: Flag for intersite profiles (0=no, 1=yes).
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis/5/).

    Returns:
        Scaling factor applied to bias changes. Values of 1.0 indicate
        converged biases; smaller values indicate ongoing convergence.

    Raises:
        FileNotFoundError: If multisite/C.dat or multisite/V.dat not found.

    Example:
        >>> from cphmd.core.free_energy import get_free_energy5
        >>> scaling = get_free_energy5("./my_system", alf_info, ms=1, msprof=1, analysis_idx=5)
        >>> print(f"Scaling factor: {scaling}")
    """
    work_dir = Path(work_dir)
    analysis_dir = work_dir / "analysis" / str(analysis_idx)

    result = _compute_free_energy(analysis_dir, alf_info, ms, msprof)
    return result.scaling


def _compute_free_energy(
    analysis_dir: Path,
    alf_info: "ALFInfo | dict",
    ms: int,
    msprof: int,
) -> FreeEnergyResult:
    """Internal implementation of free energy calculation.

    Args:
        analysis_dir: Path to analysis directory (e.g., work_dir/analysis/5/).
        alf_info: ALF configuration.
        ms: Intersite bias flag.
        msprof: Intersite profile flag.

    Returns:
        FreeEnergyResult with scaling and bias changes.
    """
    # Extract parameters from alf_info
    if hasattr(alf_info, "temp"):
        temp = alf_info.temp
        nsubs = np.array(alf_info.nsubs)
        nblocks = alf_info.nblocks
    else:
        temp = alf_info["temp"]
        nsubs = np.array(alf_info["nsubs"])
        nblocks = alf_info["nblocks"]

    kT = 0.001987 * temp
    krest = 1

    # Maximum change caps for each parameter type
    cutb = 2
    cutc = 8
    cutx = 2
    cuts = 1
    cutc2 = 2  # Inter-site c
    cutx2 = 0.5  # Inter-site x
    cuts2 = 0.5  # Inter-site s

    # Load previous bias parameters
    b_prev = np.loadtxt(analysis_dir / "b_prev.dat")
    c_prev = np.loadtxt(analysis_dir / "c_prev.dat")
    x_prev = np.loadtxt(analysis_dir / "x_prev.dat")
    s_prev = np.loadtxt(analysis_dir / "s_prev.dat")

    # Initialize output arrays
    b = np.zeros((1, nblocks))
    c = np.zeros((nblocks, nblocks))
    x = np.zeros((nblocks, nblocks))
    s = np.zeros((nblocks, nblocks))

    # Count total parameters
    nparm = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                nparm += n1 + 5 * n2
            elif ms == 1:
                nparm += 5 * n3
            elif ms == 2:
                nparm += n3

    # Build cutlist (max change per parameter) and reglist (regularization)
    cutlist = np.zeros((nparm,))
    reglist = np.zeros((nparm,))
    n0 = 0
    iblock = 0

    for isite in range(len(nsubs)):
        jblock = iblock
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2

        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]

            if isite == jsite:
                # Intra-site parameters
                cutlist[n0 : n0 + n1] = cutb
                n0 += n1
                cutlist[n0 : n0 + n2] = cutc
                n0 += n2
                cutlist[n0 : n0 + 2 * n2] = cutx
                n0 += 2 * n2
                cutlist[n0 : n0 + 2 * n2] = cuts
                n0 += 2 * n2

            elif ms == 1:
                # Inter-site parameters with full coupling
                cutlist[n0 : n0 + n3] = cutc2
                n0 += n3

                # x cross-terms with regularization
                ind = n0
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        reglist[ind] = -x_prev[iblock + i, jblock + j]
                        ind += 1
                        reglist[ind] = -x_prev[jblock + j, iblock + i]
                        ind += 1
                cutlist[n0 : n0 + 2 * n3] = cutx2
                n0 += 2 * n3

                # s cross-terms with regularization
                ind = n0
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        reglist[ind] = -s_prev[iblock + i, jblock + j]
                        ind += 1
                        reglist[ind] = -s_prev[jblock + j, iblock + i]
                        ind += 1
                cutlist[n0 : n0 + 2 * n3] = cuts2
                n0 += 2 * n3

            elif ms == 2:
                # Inter-site with only c coupling
                cutlist[n0 : n0 + n3] = cutc2
                n0 += n3

            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    # Load WHAM output matrices
    c_file = analysis_dir / "multisite" / "C.dat"
    v_file = analysis_dir / "multisite" / "V.dat"

    if not c_file.exists():
        raise FileNotFoundError(
            f"{c_file} not found. "
            "RunWham probably failed - check output and error files."
        )
    if not v_file.exists():
        raise FileNotFoundError(
            f"{v_file} not found. "
            "RunWham probably failed - check output and error files."
        )

    C = np.loadtxt(c_file)
    V = np.loadtxt(v_file)

    # Add regularization to diagonal
    for i in range(n0):
        C[i, i] += krest * cutlist[i] ** -2

    # Ensure no zero diagonal elements
    for i in range(C.shape[0]):
        if C[i, i] == 0:
            C[i, i] = 1

    # Add harmonic restraint to x and s cross terms
    for i in range(n0):
        V[i] += (krest * cutlist[i] ** -2) * reglist[i]

    # Solve linear system using truncated SVD pseudoinverse
    # This is more robust than np.linalg.solve for ill-conditioned matrices
    coeff = _solve_with_svd(C, V, n0)

    # Scale coefficients if max change exceeds 1.5x cutlist
    max_change = np.max(np.abs(coeff[0:n0] / cutlist))
    use_fallback = False

    if max_change == 0:
        # SVD produced zero coefficients (ill-conditioned matrix)
        # Apply population-based fallback to break the convergence deadlock
        logger.warning(
            "WHAM produced zero updates (ill-conditioned matrix). "
            "Applying population-based fallback."
        )
        use_fallback = True
        scaling = 1.0

        # Get populations from Lambda files
        try:
            # Try to get nreps from alf_info for ndupl
            if hasattr(alf_info, "nreps"):
                nreps = alf_info.nreps
            else:
                nreps = alf_info.get("nreps", 1)

            populations = get_populations_from_lambda(analysis_dir, nsubs, ndupl=nreps)
            delta_b = fallback_bias_update(populations, temp, max_change=cutb)

            # Apply fallback to b coefficients (only linear biases)
            # The coeff array has b terms first (n1 per site)
            ibuff = 0
            for isite in range(len(nsubs)):
                for j in range(nsubs[isite]):
                    coeff[ibuff + j] = delta_b[ibuff + j]
                ibuff += nsubs[isite]

            logger.info(f"Fallback bias updates applied: {delta_b}")

        except Exception as e:
            logger.warning(f"Fallback bias update failed: {e}. Using zero updates.")
            # Keep coeff as zeros
    else:
        scaling = min(1.0, 1.5 / max_change)
        coeff *= scaling

    logger.info(f"Free energy scaling: {scaling} (fallback={use_fallback})")

    # Unpack coefficients into bias matrices
    ind = 0
    iblock = 0

    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite == jsite:
                # Intra-site: b terms
                for i in range(nsubs[isite]):
                    b[0, iblock + i] = coeff[ind]
                    ind += 1

                # Intra-site: c terms (upper triangle)
                for i in range(nsubs[isite]):
                    for j in range(i + 1, nsubs[isite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1

                # Intra-site: x terms
                for i in range(nsubs[isite]):
                    for j in range(nsubs[isite]):
                        if i != j:
                            x[iblock + i, jblock + j] = coeff[ind]
                            ind += 1

                # Intra-site: s terms
                for i in range(nsubs[isite]):
                    for j in range(nsubs[isite]):
                        if i != j:
                            s[iblock + i, jblock + j] = coeff[ind]
                            ind += 1

            elif ms == 1:
                # Inter-site: c terms
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1

                # Inter-site: x terms (both directions)
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        x[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        x[jblock + j, iblock + i] = coeff[ind]
                        ind += 1

                # Inter-site: s terms (both directions)
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        s[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        s[jblock + j, iblock + i] = coeff[ind]
                        ind += 1

            elif ms == 2:
                # Inter-site: only c terms
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1

            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    # Subtract baseline reference for inter-site c terms
    iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite != jsite:
                for i in range(nsubs[isite]):
                    b[0, iblock + i] += c[iblock + i, jblock]
                    c[iblock + i, jblock : jblock + nsubs[jsite]] -= c[
                        iblock + i, jblock
                    ]
                for j in range(nsubs[jsite]):
                    b[0, jblock + j] += c[iblock, jblock + j]
                    c[iblock : iblock + nsubs[isite], jblock + j] -= c[
                        iblock, jblock + j
                    ]
            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    # Subtract baseline reference for b terms (first substituent at each site)
    iblock = 0
    for isite in range(len(nsubs)):
        b[0, iblock : iblock + nsubs[isite]] -= b[0, iblock]
        iblock += nsubs[isite]

    # Save output files to analysis directory
    np.savetxt(analysis_dir / "b.dat", b, fmt=" %7.2f")
    np.savetxt(analysis_dir / "c.dat", c, fmt=" %7.2f")
    np.savetxt(analysis_dir / "x.dat", x, fmt=" %7.2f")
    np.savetxt(analysis_dir / "s.dat", s, fmt=" %7.2f")

    return FreeEnergyResult(
        scaling=scaling,
        b_changes=b,
        c_changes=c,
        x_changes=x,
        s_changes=s,
    )
