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


def _solve_with_svd(
    C: np.ndarray,
    V: np.ndarray,
    n0: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve C @ coeff = V using truncated SVD pseudoinverse.

    This is more robust than np.linalg.solve for ill-conditioned matrices.
    Near-zero singular values are truncated to prevent numerical instability.

    When weights are provided, applies diagonal preconditioning:
    C_w = diag(W) @ C @ diag(W), V_w = diag(W) @ V, then un-scales
    the result. This gives higher-weight (better-sampled) parameters
    more influence in the solve.

    Args:
        C: Hessian matrix from WHAM.
        V: Gradient vector from WHAM.
        n0: Number of active parameters.
        weights: Per-parameter SVD weights (higher = more trust).
            Applied as diagonal preconditioning before SVD.

    Returns:
        Coefficient vector (same shape as V).
    """
    # Apply diagonal preconditioning if weights provided
    W = None
    if weights is not None:
        W = np.ones(C.shape[0])
        W[:n0] = np.sqrt(np.maximum(weights[:n0], 0.01))
        C = np.diag(W) @ C @ np.diag(W)
        V = W * V
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

    # Undo column scaling from preconditioning
    if W is not None:
        coeff = W * coeff

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


def _identify_param_type(idx: int, nsubs: np.ndarray) -> str:
    """Map a parameter index back to its type (cutb/cutc/cutx/cuts).

    Walks the full parameter layout (matching WHAM C/V dimensions) to
    determine which parameter type owns the given index.
    """
    offset = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2

        if offset <= idx < offset + n1:
            return "cutb"
        offset += n1

        if offset <= idx < offset + n2:
            return "cutc"
        offset += n2

        if offset <= idx < offset + 2 * n2:
            return "cutx"
        offset += 2 * n2

        if offset <= idx < offset + 2 * n2:
            return "cuts"
        offset += 2 * n2

    return "cutc2"  # Must be an inter-site parameter


def get_free_energy5(
    alf_info: "ALFInfo | dict",
    ms: int = 0,
    msprof: int = 0,
    cutb: float = 2.0,
    cutc: float = 8.0,
    cutx: float = 0.2,
    cuts: float = 0.2,
    cutc2: float = 1.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
    calc_phi: bool = True,
    calc_psi: bool = True,
    calc_chi: bool = True,
    calc_omega: bool = True,
    site_populations: list[np.ndarray] | None = None,
    transition_weights: np.ndarray | None = None,
    connectivity: float | None = None,
    analysis_dir: str | Path | None = None,
) -> float:
    """Solve for optimal bias changes via matrix inversion.

    Can be called in two ways:
    1. From inside an analysis directory (analysis_dir=None, uses cwd)
    2. With an explicit analysis_dir path

    The WHAM routine produces C.dat (Hessian) and V.dat (gradient) in
    multisite/. This routine adds regularization and inverts to find
    optimal bias parameter changes, saved as b.dat, c.dat, x.dat, s.dat.

    Args:
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        ms: Flag for intersite biases (0=none, 1=c/x/s, 2=just c).
        msprof: Flag for intersite profiles (0=no, 1=yes).
        cutb: Maximum change cap for b (linear/phi) parameters.
        cutc: Maximum change cap for c (coupling/psi) parameters.
        cutx: Maximum change cap for x (chi) parameters.
        cuts: Maximum change cap for s (omega) parameters.
        cutc2: Maximum change cap for inter-site c parameters.
        cutx2: Maximum change cap for inter-site x parameters.
        cuts2: Maximum change cap for inter-site s parameters.
        calc_phi: Include b (linear) parameters in the matrix solve.
        calc_psi: Include c (coupling) parameters in the matrix solve.
        calc_chi: Include x (exponential) parameters in the matrix solve.
        calc_omega: Include s (sigmoid) parameters in the matrix solve.
        transition_weights: Per-parameter regularization weights from transition counts.
            Weight > 1.0 means stronger regularization (less trust in poorly-sampled coupling).
        analysis_dir: Path to analysis directory. If None, uses cwd.

    Returns:
        Scaling factor applied to bias changes.
    """
    if analysis_dir is None:
        analysis_dir = Path.cwd()
    else:
        analysis_dir = Path(analysis_dir)

    result = _compute_free_energy(
        analysis_dir, alf_info, ms, msprof,
        cutb=cutb, cutc=cutc, cutx=cutx, cuts=cuts,
        cutc2=cutc2, cutx2=cutx2, cuts2=cuts2,
        calc_phi=calc_phi, calc_psi=calc_psi,
        calc_chi=calc_chi, calc_omega=calc_omega,
        site_populations=site_populations,
        transition_weights=transition_weights,
        connectivity=connectivity,
    )
    return result.scaling


def _compute_free_energy(
    analysis_dir: Path,
    alf_info: "ALFInfo | dict",
    ms: int,
    msprof: int,
    cutb: float = 2.0,
    cutc: float = 8.0,
    cutx: float = 0.2,
    cuts: float = 0.2,
    cutc2: float = 1.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
    calc_phi: bool = True,
    calc_psi: bool = True,
    calc_chi: bool = True,
    calc_omega: bool = True,
    site_populations: list[np.ndarray] | None = None,
    transition_weights: np.ndarray | None = None,
    connectivity: float | None = None,
) -> FreeEnergyResult:
    """Internal implementation of free energy calculation.

    Args:
        analysis_dir: Path to analysis directory.
        alf_info: ALF configuration.
        ms: Intersite bias flag.
        msprof: Intersite profile flag.
        cutb-cuts2: Maximum change caps for each parameter type.
        calc_phi: Include b (linear) parameters.
        calc_psi: Include c (coupling) parameters.
        calc_chi: Include x (exponential) parameters.
        calc_omega: Include s (sigmoid) parameters.
        site_populations: Per-site population arrays for dampening.
            Each entry is a 1D array of normalized populations for one site.
            When provided, bias updates for poorly-sampled sites are dampened.

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

    # Count total parameters (always full — must match WHAM C.dat/V.dat dimensions)
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

    # Build cutlist (max change per parameter), reglist (regularization),
    # and param_active mask (for selective scaling/zeroing of excluded types)
    cutlist = np.zeros((nparm,))
    reglist = np.zeros((nparm,))
    param_active = np.ones((nparm,), dtype=bool)
    param_dampen = np.ones((nparm,))

    # Compute per-site dampening factors from populations
    site_dampening = None
    if site_populations is not None:
        site_dampening = []
        for isite in range(len(nsubs)):
            target_pop = 1.0 / nsubs[isite]
            min_pop = float(np.min(site_populations[isite]))
            dampen = max(0.5, min(1.0, min_pop / target_pop))
            site_dampening.append(dampen)
        dampen_str = ", ".join(f"{d:.2f}" for d in site_dampening)
        logger.info(f"Per-site population dampening: [{dampen_str}]")

    n0 = 0
    iblock = 0

    for isite in range(len(nsubs)):
        jblock = iblock
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2

        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            n0_block_start = n0  # Track start for dampening

            if isite == jsite:
                # Intra-site parameters
                cutlist[n0 : n0 + n1] = cutb
                if not calc_phi:
                    param_active[n0 : n0 + n1] = False
                n0 += n1

                cutlist[n0 : n0 + n2] = cutc
                if not calc_psi:
                    param_active[n0 : n0 + n2] = False
                n0 += n2

                cutlist[n0 : n0 + 2 * n2] = cutx
                if not calc_chi:
                    param_active[n0 : n0 + 2 * n2] = False
                n0 += 2 * n2

                cutlist[n0 : n0 + 2 * n2] = cuts
                if not calc_omega:
                    param_active[n0 : n0 + 2 * n2] = False
                n0 += 2 * n2

            elif ms == 1:
                # Inter-site parameters with full coupling
                cutlist[n0 : n0 + n3] = cutc2
                if not calc_psi:
                    param_active[n0 : n0 + n3] = False
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
                if not calc_chi:
                    param_active[n0 : n0 + 2 * n3] = False
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
                if not calc_omega:
                    param_active[n0 : n0 + 2 * n3] = False
                n0 += 2 * n3

            elif ms == 2:
                # Inter-site with only c coupling
                cutlist[n0 : n0 + n3] = cutc2
                if not calc_psi:
                    param_active[n0 : n0 + n3] = False
                n0 += n3

            # Apply per-site dampening to this block's parameters
            if site_dampening is not None:
                if isite == jsite:
                    d = site_dampening[isite]
                else:
                    d = min(site_dampening[isite], site_dampening[jsite])
                param_dampen[n0_block_start:n0] = d

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

    # Decouple inactive params: zero their rows/columns in C and V,
    # then set diagonal to 1. This fully isolates them from the solve.
    for i in range(n0):
        if not param_active[i]:
            C[i, :] = 0.0
            C[:, i] = 0.0
            C[i, i] = 1.0
            V[i] = 0.0

    # Add regularization to diagonal for active params
    # transition_weights scale regularization: higher weight = stronger regularization
    for i in range(n0):
        if param_active[i]:
            tw = transition_weights[i] if transition_weights is not None else 1.0
            C[i, i] += tw * krest * cutlist[i] ** -2

    # Ensure no zero diagonal elements (for profile terms beyond n0)
    for i in range(n0, C.shape[0]):
        if C[i, i] == 0:
            C[i, i] = 1

    # Add harmonic restraint to x and s cross terms (active params only)
    for i in range(n0):
        if param_active[i]:
            tw = transition_weights[i] if transition_weights is not None else 1.0
            V[i] += (tw * krest * cutlist[i] ** -2) * reglist[i]

    # Build SVD weights from transition weights (inverse: high reg weight = low SVD trust)
    if transition_weights is not None:
        svd_weights = 1.0 / np.maximum(transition_weights, 0.1)
    else:
        svd_weights = None

    # Solve linear system using truncated SVD pseudoinverse
    # This is more robust than np.linalg.solve for ill-conditioned matrices
    coeff = _solve_with_svd(C, V, n0, weights=svd_weights)

    # Zero out excluded parameters and extra WHAM profile terms
    # (coeff may be larger than nparm due to WHAM profile entries)
    coeff[n0:] = 0.0
    coeff[0:n0][~param_active] = 0.0

    # Apply per-site population dampening (before scaling computation)
    # When a site is poorly sampled, WHAM estimates for that site are unreliable.
    # Dampening reduces the bias update proportionally to sampling quality.
    if site_dampening is not None:
        coeff[0:n0] *= param_dampen
        coeff[0:n0][~param_active] = 0.0

    # Per-parameter clipping: cap each active parameter at its cutoff independently.
    # This prevents one bottleneck parameter type from dragging all others down.
    active_ratios = np.abs(coeff[0:n0][param_active] / cutlist[param_active])
    max_change = np.max(active_ratios) if len(active_ratios) > 0 else 0.0
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

            # Apply fallback to b coefficients (only linear biases, if active)
            if calc_phi:
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
        # Clip each active parameter independently to its cutoff
        for i in range(n0):
            if param_active[i] and abs(coeff[i]) > cutlist[i]:
                coeff[i] = cutlist[i] * np.sign(coeff[i])
        # Report equivalent scaling for adaptive cutoff feedback
        # (worst-case ratio: what the most-constrained param got vs wanted)
        scaling = min(1.0, 1.0 / max_change)

    logger.info(f"Free energy scaling: {scaling} (fallback={use_fallback})")

    # Identify bottleneck parameter type for scaling diagnostics
    # Only consider active params to avoid division by zero cutlist values
    if max_change > 0 and not use_fallback:
        safe_cutlist = np.where(param_active, cutlist, 1.0)  # avoid /0
        ratios = np.abs(coeff[0:n0] / safe_cutlist)
        ratios[~param_active] = 0.0  # inactive params can't be bottleneck
        bottleneck_idx = int(np.argmax(ratios))
        bottleneck_type = _identify_param_type(bottleneck_idx, nsubs)
    else:
        bottleneck_type = "none"

    # Save scaling diagnostics to scaling.dat
    scaling_file = analysis_dir / "scaling.dat"
    with open(scaling_file, "w") as f:
        if connectivity is not None:
            f.write("# scaling bottleneck cutb cutc cutx cuts connectivity\n")
            f.write(f"{scaling:.6f} {bottleneck_type} {cutb:.4f} {cutc:.4f} "
                    f"{cutx:.4f} {cuts:.4f} {connectivity:.4f}\n")
        else:
            f.write("# scaling bottleneck cutb cutc cutx cuts\n")
            f.write(f"{scaling:.6f} {bottleneck_type} {cutb:.4f} {cutc:.4f} "
                    f"{cutx:.4f} {cuts:.4f}\n")

    # Unpack coefficients into bias matrices
    # (excluded params were zeroed in coeff, so output arrays get zeros naturally)
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
    from .alf_utils import _clean_negzero
    np.savetxt(analysis_dir / "b.dat", _clean_negzero(b), fmt=" %10.5f")
    np.savetxt(analysis_dir / "c.dat", _clean_negzero(c), fmt=" %10.5f")
    np.savetxt(analysis_dir / "x.dat", _clean_negzero(x), fmt=" %10.5f")
    np.savetxt(analysis_dir / "s.dat", _clean_negzero(s), fmt=" %10.5f")

    return FreeEnergyResult(
        scaling=scaling,
        b_changes=b,
        c_changes=c,
        x_changes=x,
        s_changes=s,
    )
