"""Lambda transition counting for convergence diagnostics.

This module reimplements ALF's GetTrans for counting transitions between
alchemical states to assess simulation convergence.
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

    # Process each trial
    for tag in range(ndupl):
        data_file = data_dir / f"Lambda.{tag}.0.dat"
        if not data_file.exists():
            if verbose:
                logger.warning(f"{data_file} not found, skipping")
            continue

        data = np.loadtxt(data_file)

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
