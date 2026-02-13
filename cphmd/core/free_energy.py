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

from .alf_utils import ensure_alf_info

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
        t_changes: Cross-term t bias changes (nblocks x nblocks).
        u_changes: Cross-term u bias changes (nblocks x nblocks).
    """

    scaling: float
    b_changes: np.ndarray
    c_changes: np.ndarray
    x_changes: np.ndarray
    s_changes: np.ndarray
    t_changes: np.ndarray
    u_changes: np.ndarray


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

    from cphmd.utils.lambda_io import read_lambda_values

    for tag in range(ndupl):
        # Prefer .parquet, fall back to .dat for old runs
        data_file = data_dir / f"Lambda.{tag}.0.parquet"
        if not data_file.exists():
            data_file = data_dir / f"Lambda.{tag}.0.dat"
        if not data_file.exists():
            continue

        data = read_lambda_values(data_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        total_frames += data.shape[0]

        # Vectorized state assignment: for each site, find the first
        # lambda > cutoff using argmax on the boolean mask.
        ibuff = 0
        for isite in range(len(nsubs)):
            n = int(nsubs[isite])
            site_lambdas = data[:, ibuff : ibuff + n]  # (frames, n)
            above = site_lambdas > lc  # boolean mask

            # argmax returns the first True index per row; if no lambda
            # exceeds the cutoff, the frame contributes no count.
            has_state = above.any(axis=1)  # which frames have a state
            state_idx = above.argmax(axis=1)  # first True per row

            # Count occurrences of each state
            for j in range(n):
                counts[ibuff + j] += np.sum((state_idx == j) & has_state)

            ibuff += n

    # Convert to fractions
    if total_frames > 0:
        populations = counts / total_frames
    else:
        # Equal populations if no data
        populations = np.ones(nblocks) / nblocks

    return populations


def fallback_bias_update(
    populations: np.ndarray,
    nsubs: np.ndarray,
    temperature: float,
    max_change: float = 2.0,
) -> np.ndarray:
    """Compute bias updates from populations when WHAM fails.

    Uses Boltzmann relation: ΔG = RT * ln(p_target / p_current)
    This provides a simple heuristic to push undersampled states toward
    uniform populations when the WHAM matrix is too ill-conditioned.

    Normalization and targeting are done per-site: each site's populations
    are normalized independently, and the uniform target is 1/nsubs[i]
    (not 1/nblocks).

    Args:
        populations: Current population fractions for each state.
            Shape: (nblocks,) where nblocks = sum(nsubs).
        nsubs: Substituent counts per site.
        temperature: Simulation temperature in Kelvin.
        max_change: Maximum bias change per state (kcal/mol).

    Returns:
        Bias changes (delta_b) to add to current b parameters.
        Referenced to first state within each site.
    """
    RT = 0.001987 * temperature  # kcal/mol
    nblocks = int(np.sum(nsubs))
    populations = np.asarray(populations, dtype=float)
    delta_b = np.zeros(nblocks)

    ibuff = 0
    for isite in range(len(nsubs)):
        n = int(nsubs[isite])
        site_pops = populations[ibuff : ibuff + n]
        site_sum = np.sum(site_pops)

        if site_sum <= 0:
            logger.warning(f"Site {isite}: all populations zero, skipping")
            ibuff += n
            continue

        # Normalize within this site
        site_pops = site_pops / site_sum
        target = 1.0 / n

        with np.errstate(divide="ignore", invalid="ignore"):
            site_delta = RT * np.log(target / site_pops)

        site_delta = np.nan_to_num(
            site_delta, nan=0.0, posinf=max_change, neginf=-max_change
        )
        site_delta = np.clip(site_delta, -max_change, max_change)

        # Reference to first state of this site
        site_delta -= site_delta[0]

        delta_b[ibuff : ibuff + n] = site_delta
        ibuff += n

    logger.info(f"Fallback bias update: {delta_b}")
    return delta_b


def _identify_param_type(idx: int, nsubs: np.ndarray, ntriangle: int = 5) -> str:
    """Map a parameter index back to its type (cutb/cutc/cutx/cuts/cutt/cutu).

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

        if ntriangle >= 7:
            if offset <= idx < offset + 2 * n2:
                return "cutt"
            offset += 2 * n2

        if ntriangle >= 9:
            if offset <= idx < offset + 2 * n2:
                return "cutu"
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
    cutt: float = 0.0,
    cutu: float = 0.0,
    cutc2: float = 1.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
    calc_phi: bool = True,
    calc_psi: bool = True,
    calc_chi: bool = True,
    calc_omega: bool = True,
    calc_omega2: bool = False,
    calc_omega3: bool = False,
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
    optimal bias parameter changes, saved as b.dat, c.dat, x.dat, s.dat,
    t.dat, u.dat.

    Args:
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        ms: Flag for intersite biases (0=none, 1=c/x/s, 2=just c).
        msprof: Flag for intersite profiles (0=no, 1=yes).
        cutb: Maximum change cap for b (linear/phi) parameters.
        cutc: Maximum change cap for c (coupling/psi) parameters.
        cutx: Maximum change cap for x (chi) parameters.
        cuts: Maximum change cap for s (omega) parameters.
        cutt: Maximum change cap for t parameters (0 = disabled).
        cutu: Maximum change cap for u parameters (0 = disabled).
        cutc2: Maximum change cap for inter-site c parameters.
        cutx2: Maximum change cap for inter-site x parameters.
        cuts2: Maximum change cap for inter-site s parameters.
        calc_phi: Include b (linear) parameters in the matrix solve.
        calc_psi: Include c (coupling) parameters in the matrix solve.
        calc_chi: Include x (exponential) parameters in the matrix solve.
        calc_omega: Include s (sigmoid) parameters in the matrix solve.
        calc_omega2: Include t parameters in the matrix solve.
        calc_omega3: Include u parameters in the matrix solve.
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
        cutt=cutt, cutu=cutu,
        cutc2=cutc2, cutx2=cutx2, cuts2=cuts2,
        calc_phi=calc_phi, calc_psi=calc_psi,
        calc_chi=calc_chi, calc_omega=calc_omega,
        calc_omega2=calc_omega2, calc_omega3=calc_omega3,
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
    cutt: float = 0.0,
    cutu: float = 0.0,
    cutc2: float = 1.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
    calc_phi: bool = True,
    calc_psi: bool = True,
    calc_chi: bool = True,
    calc_omega: bool = True,
    calc_omega2: bool = False,
    calc_omega3: bool = False,
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
        cutb-cutu: Maximum change caps for each parameter type.
        calc_phi: Include b (linear) parameters.
        calc_psi: Include c (coupling) parameters.
        calc_chi: Include x (exponential) parameters.
        calc_omega: Include s (sigmoid) parameters.
        calc_omega2: Include t parameters.
        calc_omega3: Include u parameters.
        site_populations: Per-site population arrays for dampening.
            Each entry is a 1D array of normalized populations for one site.
            When provided, bias updates for poorly-sampled sites are dampened.

    Returns:
        FreeEnergyResult with scaling and bias changes.
    """
    # Normalize alf_info to ALFInfo dataclass
    alf_info = ensure_alf_info(alf_info)
    temp = alf_info.temp
    nsubs = np.array(alf_info.nsubs)
    nblocks = alf_info.nblocks

    # Reserved parameters (accepted for API compatibility, not yet implemented)
    if transition_weights is not None:
        logger.debug("transition_weights provided but not yet used in regularization")
    if site_populations is not None:
        logger.debug("site_populations provided but not yet used for dampening")

    kT = 0.001987 * temp
    krest = 1

    # Load previous bias parameters
    b_prev = np.loadtxt(analysis_dir / "b_prev.dat")
    c_prev = np.loadtxt(analysis_dir / "c_prev.dat")
    x_prev = np.loadtxt(analysis_dir / "x_prev.dat")
    s_prev = np.loadtxt(analysis_dir / "s_prev.dat")

    # Determine ntriangle from active parameter types
    # ntriangle = 1(c) + 2(x) + 2(s) [+ 2(t)] [+ 2(u)]
    ntriangle = 5
    if cutt > 0 or calc_omega2:
        ntriangle = 7
    if cutu > 0 or calc_omega3:
        ntriangle = 9

    # Load t/u previous biases (create zeros if files don't exist yet)
    t_prev_path = analysis_dir / "t_prev.dat"
    u_prev_path = analysis_dir / "u_prev.dat"
    t_prev = np.loadtxt(t_prev_path) if t_prev_path.exists() else np.zeros((nblocks, nblocks))
    u_prev = np.loadtxt(u_prev_path) if u_prev_path.exists() else np.zeros((nblocks, nblocks))

    # Initialize output arrays
    b = np.zeros((1, nblocks))
    c = np.zeros((nblocks, nblocks))
    x = np.zeros((nblocks, nblocks))
    s = np.zeros((nblocks, nblocks))
    t = np.zeros((nblocks, nblocks))
    u = np.zeros((nblocks, nblocks))

    # Count total parameters (always full — must match WHAM C.dat/V.dat dimensions)
    nparm = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                nparm += n1 + ntriangle * n2
            elif ms == 1:
                nparm += ntriangle * n3
            elif ms == 2:
                nparm += n3

    # Build cutlist (max change per parameter), reglist (regularization),
    # and param_active mask (for selective scaling/zeroing of excluded types)
    cutlist = np.zeros((nparm,))
    reglist = np.zeros((nparm,))
    param_active = np.ones((nparm,), dtype=bool)

    n0 = 0
    iblock = 0

    # Track parameter indices by type (needed for C matrix symmetrization)
    b_indx: list[int] = []
    c_indx: list[int] = []
    x_indx: list[int] = []
    s_indx: list[int] = []
    t_indx: list[int] = []
    u_indx: list[int] = []
    c2_indx: list[int] = []
    x2_indx: list[int] = []
    s2_indx: list[int] = []

    for isite in range(len(nsubs)):
        jblock = iblock
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2

        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]

            if isite == jsite:
                # Intra-site parameters
                cutlist[n0 : n0 + n1] = cutb
                b_indx.extend(range(n0, n0 + n1))
                if not calc_phi:
                    param_active[n0 : n0 + n1] = False
                n0 += n1

                cutlist[n0 : n0 + n2] = cutc
                c_indx.extend(range(n0, n0 + n2))
                if not calc_psi:
                    param_active[n0 : n0 + n2] = False
                n0 += n2

                cutlist[n0 : n0 + 2 * n2] = cutx
                x_indx.extend(range(n0, n0 + 2 * n2))
                if not calc_chi:
                    param_active[n0 : n0 + 2 * n2] = False
                n0 += 2 * n2

                cutlist[n0 : n0 + 2 * n2] = cuts
                s_indx.extend(range(n0, n0 + 2 * n2))
                if not calc_omega:
                    param_active[n0 : n0 + 2 * n2] = False
                n0 += 2 * n2

                if ntriangle >= 7:
                    cutlist[n0 : n0 + 2 * n2] = cutt
                    t_indx.extend(range(n0, n0 + 2 * n2))
                    if not calc_omega2:
                        param_active[n0 : n0 + 2 * n2] = False
                    n0 += 2 * n2

                if ntriangle >= 9:
                    cutlist[n0 : n0 + 2 * n2] = cutu
                    u_indx.extend(range(n0, n0 + 2 * n2))
                    if not calc_omega3:
                        param_active[n0 : n0 + 2 * n2] = False
                    n0 += 2 * n2

            elif ms == 1:
                # Inter-site parameters with full coupling
                cutlist[n0 : n0 + n3] = cutc2
                c2_indx.extend(range(n0, n0 + n3))
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
                x2_indx.extend(range(n0, n0 + 2 * n3))
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
                s2_indx.extend(range(n0, n0 + 2 * n3))
                if not calc_omega:
                    param_active[n0 : n0 + 2 * n3] = False
                n0 += 2 * n3

                # t cross-terms with regularization
                if ntriangle >= 7:
                    ind = n0
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[jsite]):
                            reglist[ind] = -t_prev[iblock + i, jblock + j]
                            ind += 1
                            reglist[ind] = -t_prev[jblock + j, iblock + i]
                            ind += 1
                    cutlist[n0 : n0 + 2 * n3] = cutt
                    if not calc_omega2:
                        param_active[n0 : n0 + 2 * n3] = False
                    n0 += 2 * n3

                # u cross-terms with regularization
                if ntriangle >= 9:
                    ind = n0
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[jsite]):
                            reglist[ind] = -u_prev[iblock + i, jblock + j]
                            ind += 1
                            reglist[ind] = -u_prev[jblock + j, iblock + i]
                            ind += 1
                    cutlist[n0 : n0 + 2 * n3] = cutu
                    if not calc_omega3:
                        param_active[n0 : n0 + 2 * n3] = False
                    n0 += 2 * n3

            elif ms == 2:
                # Inter-site with only c coupling
                cutlist[n0 : n0 + n3] = cutc2
                c2_indx.extend(range(n0, n0 + n3))
                if not calc_psi:
                    param_active[n0 : n0 + n3] = False
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

    # Validate WHAM output — fail the iteration rather than silently
    # patching NaN/Inf with zeros (which corrupts the Hessian and
    # produces plausible-looking but wrong bias updates).
    if not np.all(np.isfinite(C)):
        nan_c = int(np.isnan(C).sum())
        inf_c = int(np.isinf(C).sum())
        raise ValueError(
            f"C.dat contains {nan_c} NaN and {inf_c} Inf values. "
            "WHAM produced a corrupt Hessian — check GPU output."
        )

    if not np.all(np.isfinite(V)):
        nan_v = int(np.isnan(V).sum())
        inf_v = int(np.isinf(V).sum())
        raise ValueError(
            f"V.dat contains {nan_v} NaN and {inf_v} Inf values. "
            "WHAM produced a corrupt gradient — check GPU output."
        )

    # Symmetrize C matrix for c/c2 parameters (numerical stability).
    # WHAM's Hessian should be symmetric for pairwise coupling params,
    # but numerical noise creates slight asymmetry. Extract the submatrix
    # for each group, symmetrize in one shot, and write back.
    for indx in [c_indx, c2_indx]:
        if len(indx) > 1:
            idx = np.array(indx)
            sub = C[np.ix_(idx, idx)]
            C[np.ix_(idx, idx)] = 0.5 * (sub + sub.T)

    # Decouple inactive params: zero their rows/columns in C and V,
    # then set diagonal to 1. This fully isolates them from the solve.
    for i in range(n0):
        if not param_active[i]:
            C[i, :] = 0.0
            C[:, i] = 0.0
            C[i, i] = 1.0
            V[i] = 0.0

    # Add Tikhonov regularization to diagonal (active params only).
    # krest/cutlist^2 acts as a harmonic restraint against zero change,
    # preventing ill-conditioning and bounding the solution magnitude.
    for i in range(n0):
        if param_active[i]:
            C[i, i] += krest * cutlist[i] ** -2

    # Ensure no zero diagonal elements (for profile terms beyond n0)
    for i in range(n0, C.shape[0]):
        if C[i, i] == 0:
            C[i, i] = 1

    # Add harmonic restraint to x and s cross terms (active params only)
    for i in range(n0):
        if param_active[i]:
            V[i] += (krest * cutlist[i] ** -2) * reglist[i]

    # Solve linear system: C @ coeff = V
    # Direct solve is exact and fast for well-conditioned matrices.
    # Regularization above ensures C is non-singular; lstsq is the fallback.
    try:
        coeff = np.linalg.solve(C, V)
    except np.linalg.LinAlgError:
        logger.warning("C is singular after regularization, using least squares")
        coeff, _, _, _ = np.linalg.lstsq(C, V, rcond=None)

    # Validate result
    if not np.all(np.isfinite(coeff)):
        logger.warning("Solve produced NaN/Inf values. Using zero updates.")
        coeff = np.zeros_like(V)

    # Zero out excluded parameters and extra WHAM profile terms
    # (coeff may be larger than nparm due to WHAM profile entries)
    coeff[n0:] = 0.0
    coeff[0:n0][~param_active] = 0.0

    # Compute worst-case ratio of |correction| to cutoff across active parameters.
    # Used to decide whether global scaling is needed.
    active_ratios = np.abs(coeff[0:n0][param_active] / cutlist[param_active])
    max_change = np.max(active_ratios) if len(active_ratios) > 0 else 0.0
    use_fallback = False

    if max_change == 0:
        # Solver produced zero coefficients (ill-conditioned matrix)
        # Apply population-based fallback to break the convergence deadlock
        logger.warning(
            "WHAM produced zero updates (ill-conditioned matrix). "
            "Applying population-based fallback."
        )
        use_fallback = True
        scaling = 1.0

        # Get populations from Lambda files
        try:
            # alf_info is normalized to ALFInfo at function start
            nreps = alf_info.nreps if alf_info.nreps else 1

            populations = get_populations_from_lambda(analysis_dir, nsubs, ndupl=nreps)
            delta_b = fallback_bias_update(populations, nsubs, temp, max_change=cutb)

            # Apply fallback to b coefficients using b_indx (correct coeff
            # offsets).  b_indx maps block index → coeff index, skipping
            # over c/x/s parameter slots between sites.
            if calc_phi:
                block_idx = 0
                for isite in range(len(nsubs)):
                    for j in range(nsubs[isite]):
                        coeff[b_indx[block_idx + j]] = delta_b[block_idx + j]
                    block_idx += nsubs[isite]

            logger.info(f"Fallback bias updates applied: {delta_b}")

        except Exception as e:
            logger.warning(f"Fallback bias update failed: {e}. Using zero updates.")
            # Keep coeff as zeros
    else:
        # Global scaling: if the worst-case ratio exceeds 1.5x, scale the
        # entire correction vector down proportionally.  This preserves the
        # optimizer's chosen proportions across all parameter types — unlike
        # per-parameter clipping, which distorts the coupled WHAM solution.
        # The 1.5 factor matches the legacy algorithm that converged reliably.
        scaling = 1.5 / max_change
        # Safety: cap at 1.0, handle degenerate values (legacy guard)
        if scaling > 1.0 or np.isnan(scaling) or np.isinf(scaling):
            scaling = 1.0
        coeff[0:n0] *= scaling
        coeff[0:n0][~param_active] = 0.0

    logger.info(f"Free energy scaling: {scaling} (fallback={use_fallback})")

    # Identify bottleneck parameter type for scaling diagnostics
    # Only consider active params to avoid division by zero cutlist values
    if max_change > 0 and not use_fallback:
        safe_cutlist = np.where(param_active, cutlist, 1.0)  # avoid /0
        ratios = np.abs(coeff[0:n0] / safe_cutlist)
        ratios[~param_active] = 0.0  # inactive params can't be bottleneck
        bottleneck_idx = int(np.argmax(ratios))
        bottleneck_type = _identify_param_type(bottleneck_idx, nsubs, ntriangle)
    else:
        bottleneck_type = "none"

    # Save scaling diagnostics to scaling.dat
    scaling_file = analysis_dir / "scaling.dat"
    with open(scaling_file, "w") as f:
        if connectivity is not None:
            f.write("# scaling bottleneck cutb cutc cutx cuts cutt cutu connectivity\n")
            f.write(f"{scaling:.6f} {bottleneck_type} {cutb:.4f} {cutc:.4f} "
                    f"{cutx:.4f} {cuts:.4f} {cutt:.4f} {cutu:.4f} {connectivity:.4f}\n")
        else:
            f.write("# scaling bottleneck cutb cutc cutx cuts cutt cutu\n")
            f.write(f"{scaling:.6f} {bottleneck_type} {cutb:.4f} {cutc:.4f} "
                    f"{cutx:.4f} {cuts:.4f} {cutt:.4f} {cutu:.4f}\n")

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

                # Intra-site: t terms (sign-flipped)
                if ntriangle >= 7:
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[isite]):
                            if i != j:
                                t[iblock + i, jblock + j] = -coeff[ind]
                                ind += 1

                # Intra-site: u terms
                if ntriangle >= 9:
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[isite]):
                            if i != j:
                                u[iblock + i, jblock + j] = coeff[ind]
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

                # Inter-site: t terms (both directions, sign-flipped)
                if ntriangle >= 7:
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[jsite]):
                            t[iblock + i, jblock + j] = -coeff[ind]
                            ind += 1
                            t[jblock + j, iblock + i] = -coeff[ind]
                            ind += 1

                # Inter-site: u terms (both directions)
                if ntriangle >= 9:
                    for i in range(nsubs[isite]):
                        for j in range(nsubs[jsite]):
                            u[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            u[jblock + j, iblock + i] = coeff[ind]
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
    np.savetxt(analysis_dir / "t.dat", _clean_negzero(t), fmt=" %10.5f")
    np.savetxt(analysis_dir / "u.dat", _clean_negzero(u), fmt=" %10.5f")

    return FreeEnergyResult(
        scaling=scaling,
        b_changes=b,
        c_changes=c,
        x_changes=x,
        s_changes=s,
        t_changes=t,
        u_changes=u,
    )
