"""Lambda transition counting for convergence diagnostics and regularization.

This module reimplements ALF's GetTrans for counting transitions between
alchemical states to assess simulation convergence. Also provides transition-
based regularization weights for the SVD free energy solve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .alf_utils import ALFInfo

logger = logging.getLogger(__name__)


@dataclass
class TransitionResult:
    """Result of transition counting.

    Attributes:
        matrices: Dictionary mapping site index to transition matrix.
        from_counts: Dictionary mapping site index to counts of transitions
            from each state (row sums of transition matrix).
        to_counts: Dictionary mapping site index to counts of transitions
            to each state (column sums of transition matrix).
        total_transitions: Total number of transitions across all sites.
    """

    matrices: dict[int, np.ndarray] = field(default_factory=dict)
    from_counts: dict[int, np.ndarray] = field(default_factory=dict)
    to_counts: dict[int, np.ndarray] = field(default_factory=dict)
    total_transitions: int = 0


def get_transitions(
    work_dir: str | Path,
    alf_info: "ALFInfo | dict",
    analysis_idx: int,
    ndupl: int | None = None,
    lc: float = 0.8,
    verbose: bool = True,
) -> TransitionResult:
    """Count transitions in alchemical trajectories.

    Counts transitions between lambda states at the specified cutoff threshold.
    This is useful for assessing convergence of ALF simulations - adequate
    sampling requires many transitions between states.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis/5/).
        ndupl: Number of independent trials. If None, treated as flattening
            run with single trial.
        lc: Lambda cutoff above which lambda must rise for a transition to
            count. Common values are 0.8 (lenient) and 0.99 (stringent).
            Stringent cutoffs result in 2-3x fewer counted transitions.
        verbose: If True, print transition matrices to stdout.

    Returns:
        TransitionResult containing transition matrices and counts per site.

    Example:
        >>> from cphmd.core.transitions import get_transitions
        >>> result = get_transitions("./my_system", alf_info, analysis_idx=5, ndupl=10)
        >>> print(f"Total transitions: {result.total_transitions}")
        >>> print(result.matrices[0])
    """
    work_dir = Path(work_dir)
    analysis_dir = work_dir / "analysis" / str(analysis_idx)

    # Extract parameters from alf_info
    if hasattr(alf_info, "nsubs"):
        nsubs = np.array(alf_info.nsubs)
        nblocks = alf_info.nblocks
    else:
        nsubs = np.array(alf_info["nsubs"])
        nblocks = alf_info["nblocks"]

    if ndupl is None:
        ndupl = 1

    # Initialize result
    result = TransitionResult()
    for isite in range(len(nsubs)):
        result.matrices[isite] = np.zeros((nsubs[isite], nsubs[isite]), dtype=int)
        result.from_counts[isite] = np.zeros(nsubs[isite], dtype=int)
        result.to_counts[isite] = np.zeros(nsubs[isite], dtype=int)

    data_dir = analysis_dir / "data"

    from cphmd.utils.lambda_io import read_lambda_values

    # Process each trial
    for tag in range(ndupl):
        # Prefer .parquet, fall back to .dat for old runs
        data_file = data_dir / f"Lambda.{tag}.0.parquet"
        if not data_file.exists():
            data_file = data_dir / f"Lambda.{tag}.0.dat"
        if not data_file.exists():
            if verbose:
                logger.warning(f"{data_file} not found, skipping")
            continue

        data = read_lambda_values(data_file)

        # Process each site
        ibuff = 0
        for isite in range(len(nsubs)):
            trans = np.zeros((nsubs[isite], nsubs[isite]), dtype=int)

            i_curr = -1
            for t in range(data.shape[0]):
                i_prev = i_curr

                # Find current state (first lambda > cutoff)
                for j in range(nsubs[isite]):
                    if data[t, j + ibuff] > lc:
                        i_curr = j
                        break

                # Count transition if state changed
                if i_prev >= 0 and i_prev != i_curr:
                    trans[i_prev, i_curr] += 1

            # Accumulate results
            result.matrices[isite] += trans

            if verbose:
                logger.info(f"Trial {tag}, site {isite}")
                logger.info("Transition matrix")
                logger.info(f"\n{trans}")
                logger.info(f"Transitions from: {np.sum(trans, axis=1)}")
                logger.info(f"Transitions to: {np.sum(trans, axis=0)}")

            ibuff += nsubs[isite]

    # Compute totals
    for isite in range(len(nsubs)):
        result.from_counts[isite] = np.sum(result.matrices[isite], axis=1)
        result.to_counts[isite] = np.sum(result.matrices[isite], axis=0)
        result.total_transitions += np.sum(result.matrices[isite])

    return result


def summarize_transitions(result: TransitionResult) -> str:
    """Generate a text summary of transition results.

    Args:
        result: TransitionResult from get_transitions().

    Returns:
        Formatted string summary.
    """
    lines = ["Transition Summary", "=" * 40]

    for isite in sorted(result.matrices.keys()):
        mat = result.matrices[isite]
        lines.append(f"\nSite {isite}:")
        lines.append(f"  States: {mat.shape[0]}")
        lines.append(f"  Total transitions: {np.sum(mat)}")
        lines.append(f"  Transitions from: {result.from_counts[isite]}")
        lines.append(f"  Transitions to:   {result.to_counts[isite]}")

        # Check for poorly sampled states
        from_min = np.min(result.from_counts[isite])
        to_min = np.min(result.to_counts[isite])
        if from_min < 10:
            lines.append("  WARNING: Low transitions from state(s)")
        if to_min < 10:
            lines.append("  WARNING: Low transitions to state(s)")

    lines.append(f"\nTotal transitions (all sites): {result.total_transitions}")

    return "\n".join(lines)


# ============================================================================
# Transition-aware optimization functions (for regularization and adaptation)
# ============================================================================


def compute_transition_matrix(
    lambda_data: np.ndarray,
    nsubs: list[int],
    threshold: float = 0.985,
) -> list[np.ndarray]:
    """Compute per-site transition count matrices from Lambda trajectories.

    A transition from state i to state j is counted when:
    - Frame t has lambda[i] > threshold (system in state i)
    - Frame t+1 has lambda[j] > threshold (system in state j)
    - i != j (actual transition, not self-loop)

    Args:
        lambda_data: Combined lambda trajectory (nframes x nblocks).
        nsubs: Number of substates per site (e.g., [3] or [2, 3]).
        threshold: Lambda threshold for physical state assignment.

    Returns:
        List of transition count matrices, one per site.
        Each matrix is (nsubs[site] x nsubs[site]) with T[i,j] = count of i->j transitions.
    """
    nframes, nblocks = lambda_data.shape
    if nframes < 2:
        return [np.zeros((n, n), dtype=int) for n in nsubs]

    # Assign each frame to a state per site (or -1 if no state above threshold)
    results = []
    col_offset = 0
    for site_idx, n in enumerate(nsubs):
        site_lambdas = lambda_data[:, col_offset:col_offset + n]
        # State assignment: argmax if any lambda > threshold, else -1
        above = site_lambdas > threshold
        any_above = above.any(axis=1)
        states = np.full(nframes, -1, dtype=int)
        states[any_above] = np.argmax(site_lambdas[any_above], axis=1)

        # Count transitions using last-known physical state
        # (intermediate frames with no lambda > threshold are skipped)
        T = np.zeros((n, n), dtype=int)
        last_state = -1
        for t in range(nframes):
            if states[t] >= 0:
                if last_state >= 0 and states[t] != last_state:
                    T[last_state, states[t]] += 1
                last_state = states[t]

        results.append(T)
        col_offset += n

    return results


def _weight_from_count(
    count: int, min_transitions: int, max_weight: float,
) -> float:
    """Inverse-sqrt weight: 10 trans -> ~1.0, 1 -> ~3.2, 100 -> ~0.3."""
    count = max(count, min_transitions)
    return min(max_weight, (min_transitions / count) ** 0.5)


def transition_matrix_to_coupling_weights(
    trans_matrices: list[np.ndarray],
    nsubs: list[int],
    ms: int = 0,
    min_transitions: int = 10,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Convert transition counts to per-coupling regularization weights.

    The weight array mirrors the exact parameter layout of _compute_free_energy:
    for each (isite, jsite) pair with jsite >= isite, intra-site blocks use
    per-pair transition counts; inter-site blocks use weight=1.0 (neutral)
    since cross-site transitions are not directly measurable.

    Args:
        trans_matrices: Per-site transition count matrices from compute_transition_matrix.
        nsubs: Number of substates per site.
        ms: Multisite coupling flag (0=none, 1=full c/x/s, 2=c-only).
        min_transitions: Floor for transition count (avoids division by zero).
        max_weight: Maximum regularization multiplier.

    Returns:
        1D array of regularization weights matching the parameter order
        in _compute_free_energy. Weight of 1.0 = normal, >1.0 = stronger.
    """
    weights = []

    for isite in range(len(nsubs)):
        ni = nsubs[isite]
        n2 = ni * (ni - 1) // 2
        T_sym_i = trans_matrices[isite] + trans_matrices[isite].T

        for jsite in range(isite, len(nsubs)):
            nj = nsubs[jsite]
            n3 = ni * nj

            if isite == jsite:
                # --- Intra-site block: use per-pair transition counts ---
                # b parameters (linear): no coupling dependence
                for i in range(ni):
                    weights.append(1.0)

                # c parameters (upper triangle pairs)
                for i in range(ni):
                    for j in range(i + 1, ni):
                        w = _weight_from_count(
                            T_sym_i[i, j], min_transitions, max_weight
                        )
                        weights.append(w)

                # x parameters (all off-diagonal ordered pairs)
                for i in range(ni):
                    for j in range(ni):
                        if i != j:
                            w = _weight_from_count(
                                T_sym_i[i, j], min_transitions, max_weight
                            )
                            weights.append(w)

                # s parameters (all off-diagonal ordered pairs)
                for i in range(ni):
                    for j in range(ni):
                        if i != j:
                            w = _weight_from_count(
                                T_sym_i[i, j], min_transitions, max_weight
                            )
                            weights.append(w)

            elif ms == 1:
                # --- Inter-site full coupling: neutral weights ---
                # c cross-terms
                for _ in range(n3):
                    weights.append(1.0)
                # x cross-terms (both directions)
                for _ in range(2 * n3):
                    weights.append(1.0)
                # s cross-terms (both directions)
                for _ in range(2 * n3):
                    weights.append(1.0)

            elif ms == 2:
                # --- Inter-site c-only: neutral weights ---
                for _ in range(n3):
                    weights.append(1.0)

    return np.array(weights)


def compute_connectivity_metric(
    trans_matrices: list[np.ndarray],
    expected_per_pair: int = 50,
) -> tuple[float, list[tuple[int, int, int]]]:
    """Compute a connectivity quality metric from transition matrices.

    The metric is the minimum pairwise transition count across all sites,
    normalized by the expected count. A value of 1.0 means all pairs have
    at least the expected number of transitions.

    Args:
        trans_matrices: Per-site transition count matrices.
        expected_per_pair: Expected transitions per pair for quality=1.0.

    Returns:
        Tuple of:
        - connectivity: float in [0, 2.0], where <1.0 means some pairs undersampled
        - weak_pairs: list of (site, state_i, state_j) with fewest transitions
    """
    min_count = float('inf')
    weak_pairs = []

    for site_idx, T in enumerate(trans_matrices):
        n = T.shape[0]
        T_sym = T + T.T
        for i in range(n):
            for j in range(i + 1, n):
                count = T_sym[i, j]
                if count < min_count:
                    min_count = count
                    weak_pairs = [(site_idx, i, j)]
                elif count == min_count:
                    weak_pairs.append((site_idx, i, j))

    if min_count == float('inf'):
        return 0.0, []

    connectivity = min(min_count / expected_per_pair, 2.0)
    return connectivity, weak_pairs


def find_weakest_transitions(
    trans_matrices: list[np.ndarray],
    nsubs: list[int],
    threshold_count: int = 5,
) -> dict[int, tuple[int, int]]:
    """Find the weakest transition pair per site.

    Returns:
        Dict of {site_idx: (state_i, state_j)} for sites that have
        a transition pair below threshold_count. Only includes the
        single weakest pair per site.
    """
    result = {}
    for site_idx, T in enumerate(trans_matrices):
        n = T.shape[0]
        if n < 2:
            continue
        T_sym = T + T.T
        min_count = float('inf')
        min_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                if T_sym[i, j] < min_count:
                    min_count = T_sym[i, j]
                    min_pair = (i, j)
        if min_count < threshold_count and min_pair is not None:
            result[site_idx] = min_pair
    return result


def save_transition_matrix(trans_matrices: list[np.ndarray], filepath: Path) -> None:
    """Save transition matrices to a human-readable file."""
    with open(filepath, "w") as f:
        f.write("# Transition count matrices per site\n")
        for site_idx, T in enumerate(trans_matrices):
            f.write(f"# Site {site_idx} ({T.shape[0]} states)\n")
            for row in T:
                f.write(" ".join(f"{v:6d}" for v in row) + "\n")
            total = T.sum()
            min_pair = T[np.triu_indices_from(T, k=1)]
            f.write(f"# Total transitions: {total}, "
                    f"min pair: {min_pair.min() if len(min_pair) > 0 else 0}\n\n")
