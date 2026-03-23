"""Free energy variance estimation using histogram-based estimator.

This module reimplements ALF's GetVariance for computing free energy
changes with bootstrap uncertainty estimation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .bias_constants import CHI_OFFSET, OMEGA_DECAY

if TYPE_CHECKING:
    from .alf_utils import ALFInfo

logger = logging.getLogger(__name__)


@dataclass
class VarianceResult:
    """Result of variance/free energy calculation.

    Attributes:
        values: Free energy values per ligand state (relative to reference).
        errors: Bootstrap uncertainty estimates.
        indices: State indices (combinations of substituents per site).
        g_matrix: Raw free energy matrix (nf x nlig).
    """

    values: np.ndarray
    errors: np.ndarray
    indices: np.ndarray
    g_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


def _build_analysis_dir(work_dir: Path, analysis_idx: int) -> Path:
    """Build path to an analysis directory.

    The ALF runner creates directories at work_dir/analysis{N} (e.g., analysis5),
    not work_dir/analysis/{N}.

    Args:
        work_dir: Path to system work directory.
        analysis_idx: Analysis cycle index.

    Returns:
        Path to the analysis directory.
    """
    return work_dir / f"analysis{analysis_idx}"


def _resolve_nbshift_dir(analysis_dir: Path, work_dir: Path) -> Path:
    """Resolve nbshift directory, preferring per-analysis snapshot.

    Each analysisN/ directory gets a frozen copy of nbshift/ at the time
    that run was executed (via shutil.copytree in _alf_analysis). This
    snapshot preserves the phase-specific delta_pKa. Falls back to the
    top-level work_dir/nbshift/ for old runs that lack per-analysis copies.

    Args:
        analysis_dir: Path to analysisN/ directory.
        work_dir: Path to system work directory.

    Returns:
        Path to the nbshift directory to use.
    """
    local = analysis_dir / "nbshift"
    if local.is_dir() and (local / "b_shift.dat").exists():
        return local
    return work_dir / "nbshift"


def get_variance(
    work_dir: str | Path,
    alf_info: "ALFInfo | dict",
    nf: int,
    analysis_idx: int,
    nbs: int = 50,
    lc: float = 0.99,
    seed: int = 2401,
) -> VarianceResult:
    """Compute free energies using histogram estimator with bootstrap.

    Estimates free energy changes using the histogram-based estimator and
    alchemical trajectories in analysis{idx}/data/Lambda.[idupl].[irep].{parquet,dat}.
    The histogram estimator uses a lambda cutoff (default 0.99) to count
    endpoint occupancy.

    Free energies are computed as -kT*log(P), where P is the number of times
    an alchemical state occurs at the endpoint. The energy of the biases at
    the endpoints is added back in.

    Uncertainty is estimated by bootstrapping from the independent trials
    NBS times (default 50).

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        nf: Number of independent trials run during production.
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis5/).
        nbs: Number of bootstrap samples (default 50).
        lc: Lambda cutoff for endpoint detection (default 0.99).
        seed: Random seed for bootstrap reproducibility (default 2401).

    Returns:
        VarianceResult with free energy values, errors, and indices.

    Note:
        Results are written to Result.txt and Result.[irep].txt files
        in the analysis directory.

    Example:
        >>> from cphmd.core.variance import get_variance
        >>> result = get_variance("./my_system", alf_info, nf=10, analysis_idx=5)
        >>> print(f"Free energies: {result.values}")
        >>> print(f"Uncertainties: {result.errors}")
    """
    work_dir = Path(work_dir)
    analysis_dir = _build_analysis_dir(work_dir, analysis_idx)
    nbshift_dir = _resolve_nbshift_dir(analysis_dir, work_dir)

    return _compute_variance(
        analysis_dir, nbshift_dir, alf_info, nf, nbs, lc, seed
    )


def _compute_variance(
    analysis_dir: Path,
    nbshift_dir: Path,
    alf_info: "ALFInfo | dict",
    nf: int,
    nbs: int,
    lc: float,
    seed: int,
) -> VarianceResult:
    """Internal implementation of variance calculation.

    Args:
        analysis_dir: Path to analysis directory.
        nbshift_dir: Path to nbshift directory.
        alf_info: ALF configuration.
        nf: Number of independent trials.
        nbs: Number of bootstrap samples.
        lc: Lambda cutoff.
        seed: Random seed for reproducibility.

    Returns:
        VarianceResult with computed values.
    """
    # Extract parameters from alf_info
    if hasattr(alf_info, "temp"):
        temp = alf_info.temp
        nsubs = np.array(alf_info.nsubs)
        nblocks = alf_info.nblocks
        nreps = alf_info.nreps
        ncentral = alf_info.ncentral
    else:
        temp = alf_info["temp"]
        nsubs = np.array(alf_info["nsubs"])
        nblocks = alf_info["nblocks"]
        nreps = alf_info["nreps"]
        ncentral = alf_info["ncentral"]

    kT = 0.001987 * temp
    nlig = int(np.prod(nsubs))

    # Load bias parameters
    b = np.loadtxt(analysis_dir / "b_prev.dat").flatten()
    b_corr_file = analysis_dir / "b_corr.dat"
    if b_corr_file.exists():
        b = b + np.loadtxt(b_corr_file).flatten()
    c = np.loadtxt(analysis_dir / "c_prev.dat")
    x = np.loadtxt(analysis_dir / "x_prev.dat")
    s = np.loadtxt(analysis_dir / "s_prev.dat")

    # Load pH shift biases
    b_shift = np.loadtxt(nbshift_dir / "b_shift.dat").flatten()
    c_shift = np.loadtxt(nbshift_dir / "c_shift.dat")
    x_shift = np.loadtxt(nbshift_dir / "x_shift.dat")
    s_shift = np.loadtxt(nbshift_dir / "s_shift.dat")

    # Load WHAM free energies
    f = np.loadtxt(analysis_dir / "f.dat")

    # Build index mapping from ligand state to block indices
    ind = np.zeros((nlig, len(nsubs)), dtype=int)
    for i in range(1, nlig):
        ind[i, :] = ind[i - 1, :]
        for j in range(len(nsubs) - 1, -1, -1):
            if (ind[i, j] + 1) < nsubs[j]:
                ind[i, j] += 1
                break
            else:
                ind[i, j] = 0

    # Convert to block indices
    blk = copy.deepcopy(ind)
    for i in range(1, len(nsubs)):
        blk[:, i:] += nsubs[i - 1]

    # Pre-compute energies for all states
    Eall = np.zeros((nreps, nf, nlig))
    Eshift = np.zeros((nreps, nf, nlig))
    lndenom = np.zeros((nreps, nf, nlig))
    nframes = np.zeros((nreps, nf))

    from cphmd.utils.lambda_io import read_lambda_values

    data_dir = analysis_dir / "data"

    for irep in range(nreps):
        for itrial in range(nf):
            isim = itrial * nreps + irep
            lf = data_dir / f"Lambda.{itrial}.{irep}.parquet"
            if not lf.exists():
                lf = data_dir / f"Lambda.{itrial}.{irep}.dat"
            L = read_lambda_values(lf)
            nframes[irep, itrial] = L.shape[0]

            for j in range(nlig):
                LList = np.zeros((nblocks,))
                LList[blk[j, :]] = 1

                # Compute bias energy at endpoint
                Eall[irep, itrial, j] = (
                    np.dot(LList, -b)
                    + np.dot(np.dot(LList, -c), LList)
                    + np.dot(np.dot(1 - np.exp(OMEGA_DECAY * LList), -x), LList)
                    + np.dot(np.dot(LList / (LList + CHI_OFFSET), -s), LList)
                )

                # Compute pH shift energy
                Eshift[irep, itrial, j] = (irep - ncentral) * (
                    np.dot(LList, -b_shift)
                    + np.dot(np.dot(LList, -c_shift), LList)
                    + np.dot(np.dot(1 - np.exp(OMEGA_DECAY * LList), -x_shift), LList)
                    + np.dot(np.dot(LList / (LList + CHI_OFFSET), -s_shift), LList)
                )

                # Log denominator for WHAM reweighting
                lndenom[irep, itrial, j] = (
                    np.log(nframes[irep, itrial])
                    + f[isim]
                    - (Eshift[irep, itrial, j]) / kT
                )

    # Count endpoint occupancy and compute free energies per replica
    PkeepA = np.zeros((nreps, nf, nlig))
    rng = np.random.default_rng(seed)

    for irep in range(nreps):
        Pkeep = np.zeros((nf, nlig))
        G = np.zeros((nf, nlig))

        for itrial in range(nf):
            lf2 = data_dir / f"Lambda.{itrial}.{irep}.parquet"
            if not lf2.exists():
                lf2 = data_dir / f"Lambda.{itrial}.{irep}.dat"
            L = read_lambda_values(lf2)

            for j in range(nlig):
                # Count frames where all site lambdas > cutoff
                P = np.sum(np.all(L[:, blk[j, :]] > lc, axis=1))
                Pkeep[itrial, j] = P

                # Handle zero counts gracefully
                if P == 0:
                    G[itrial, j] = np.inf
                else:
                    G[itrial, j] = (
                        -Eall[irep, itrial, j]
                        - Eshift[irep, itrial, j]
                        - kT * np.log(P)
                    )

        PkeepA[irep, :, :] = Pkeep
        np.savetxt(analysis_dir / f"G.{irep}.dat", G)

        # Compute free energies for this replica (handle inf values)
        finite_mask = np.isfinite(G)
        if not np.all(finite_mask):
            logger.warning(
                f"Replica {irep}: {np.sum(~finite_mask)} states with zero counts"
            )

        Gmin = np.min(np.where(finite_mask, G, np.inf), axis=0)
        Value_rep = Gmin - kT * np.log(
            np.mean(np.where(finite_mask, np.exp(-(G - Gmin) / kT), 0), axis=0)
        )
        Value_rep -= Value_rep[0]

        # Bootstrap uncertainty
        GS = np.zeros((nbs, nlig))
        for i in range(nbs):
            bs_idx = rng.integers(0, nf, size=nf)
            G_bs = G[bs_idx, :]
            finite_mask_bs = np.isfinite(G_bs)
            GS[i, :] = Gmin - kT * np.log(
                np.mean(
                    np.where(finite_mask_bs, np.exp(-(G_bs - Gmin) / kT), 0),
                    axis=0
                )
            )
        Error_rep = np.std(GS, axis=0)

        # Write per-replica results
        with open(analysis_dir / f"Result.{irep}.txt", "w") as fp:
            for i in range(nlig):
                for j in range(len(nsubs)):
                    fp.write(f"{ind[i, j]:2d} ")
                fp.write(f"{Value_rep[i]:8.3f} +/- {Error_rep[i]:5.3f}\n")

    # Combine replicas using WHAM reweighting
    G = np.zeros((nf, nlig))
    for itrial in range(nf):
        for j in range(nlig):
            total_counts = np.sum(PkeepA[:, itrial, j])
            if total_counts == 0:
                G[itrial, j] = np.inf
            else:
                G[itrial, j] = (
                    -Eall[ncentral, itrial, j]
                    - kT * np.log(total_counts)
                    + kT * np.log(np.sum(np.exp(lndenom[:, itrial, j])))
                )

    np.savetxt(analysis_dir / "G.dat", G)

    # Final free energy estimate
    finite_mask = np.isfinite(G)
    Gmin = np.min(np.where(finite_mask, G, np.inf), axis=0)
    Value = Gmin - kT * np.log(
        np.mean(np.where(finite_mask, np.exp(-(G - Gmin) / kT), 0), axis=0)
    )
    Value -= Value[0]

    # Bootstrap uncertainty for combined estimate
    GS = np.zeros((nbs, nlig))
    for i in range(nbs):
        bs_idx = rng.integers(0, nf, size=nf)
        G_bs = G[bs_idx, :]
        finite_mask_bs = np.isfinite(G_bs)
        GS[i, :] = Gmin - kT * np.log(
            np.mean(
                np.where(finite_mask_bs, np.exp(-(G_bs - Gmin) / kT), 0),
                axis=0
            )
        )
    Error = np.std(GS, axis=0)

    # Write combined results
    with open(analysis_dir / "Result.txt", "w") as fp:
        for i in range(nlig):
            for j in range(len(nsubs)):
                fp.write(f"{ind[i, j]:2d} ")
            fp.write(f"{Value[i]:8.3f} +/- {Error[i]:5.3f}\n")

    return VarianceResult(
        values=Value,
        errors=Error,
        indices=ind,
        g_matrix=G,
    )
