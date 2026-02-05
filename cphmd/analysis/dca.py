"""Direct Coupling Analysis (DCA) / Potts Model estimator for multi-site systems.

This module reimplements ALF's DCA analysis routines for computing free energies
using the Potts model estimator, which is more appropriate for large multi-site
perturbation systems.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cphmd.core.alf_utils import ALFInfo

from cphmd.core.alf_utils import ensure_alf_info

logger = logging.getLogger(__name__)


@dataclass
class DCAResult:
    """Result of DCA analysis.

    Attributes:
        h_model: Single-site field parameters (h) including bias.
        J_model: Pairwise coupling parameters (J) including bias.
        values: Free energy values per ligand state.
        errors: Bootstrap uncertainty estimates.
        indices: State indices (no-gap indexing).
    """

    h_model: np.ndarray = field(default_factory=lambda: np.array([]))
    J_model: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    errors: np.ndarray = field(default_factory=lambda: np.array([]))
    indices: np.ndarray = field(default_factory=lambda: np.array([]))


def get_model_dca(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    nf: int,
    analysis_idx: int,
    nbs: int = 50,
) -> DCAResult:
    """Compute h and J fields from Potts model estimator.

    Combines the Potts model parameters (h, J) from likelihood maximization
    with the bias energies to produce the complete model parameters.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        nf: Number of independent trials during production.
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis/5/).
        nbs: Number of bootstrap samples to process.

    Returns:
        DCAResult containing h_model and J_model parameters.

    Note:
        Expects either h.LM.dat/J.LM.dat (Likelihood Maximization) or
        h.PLM.dat/J.PLM.dat (Pseudo-Likelihood Maximization) files.
    """
    work_dir = Path(work_dir)
    analysis_dir = work_dir / "analysis" / str(analysis_idx)
    data_dir = analysis_dir / "data"

    return _compute_model_dca(analysis_dir, data_dir, alf_info, nf, nbs)


def _compute_model_dca(
    analysis_dir: Path,
    data_dir: Path,
    alf_info: ALFInfo | dict,
    nf: int,
    nbs: int,
) -> DCAResult:
    """Internal implementation of model computation.

    Args:
        analysis_dir: Path to analysis directory.
        data_dir: Path to data directory with h/J files.
        alf_info: ALF configuration.
        nf: Number of independent trials.
        nbs: Number of bootstrap samples.

    Returns:
        DCAResult with h_model and J_model.
    """
    # Normalize alf_info to ALFInfo dataclass
    alf_info = ensure_alf_info(alf_info)
    temp = alf_info.temp
    nsubs = np.array(alf_info.nsubs)

    kT = 0.001987 * temp
    nsites = len(nsubs)
    nblocks = int(np.sum(nsubs))

    # Determine which estimator was used
    if (data_dir / "h.LM.dat").exists():
        tag = "LM"
    elif (data_dir / "h.PLM.dat").exists():
        tag = "PLM"
    else:
        raise FileNotFoundError(
            f"Neither h.LM.dat nor h.PLM.dat found in {data_dir}"
        )

    # Build block-to-site mapping
    block2site = np.zeros((nblocks,), dtype=int)
    k = 0
    for i in range(nsites):
        for j in range(nsubs[i]):
            block2site[k] = i
            k += 1

    # Load bias parameters from analysis directory
    b = np.loadtxt(analysis_dir / "b_prev.dat").flatten()
    b_corr_file = analysis_dir / "b_corr.dat"
    if b_corr_file.exists():
        b = b + np.loadtxt(b_corr_file).flatten()
    c = np.loadtxt(analysis_dir / "c_prev.dat")
    x = np.loadtxt(analysis_dir / "x_prev.dat")
    s = np.loadtxt(analysis_dir / "s_prev.dat")

    # Lambda endpoint values for bias energy calculation
    c1 = 1
    x1 = 1 - np.exp(-5.56 * 1)
    s1 = 1 / (1 + 0.017)

    # Compute bias contributions to h and J
    h_bias = np.zeros((nblocks,))
    J_bias = np.zeros((nblocks, nblocks))

    for i in range(nblocks):
        h_bias[i] = b[i]
        for j in range(nblocks):
            if block2site[i] != block2site[j]:
                J_bias[i, j] = (
                    c1 * (c[i, j] + c[j, i])
                    + x1 * (x[i, j] + x[j, i])
                    + s1 * (s[i, j] + s[j, i])
                )

    np.savetxt(data_dir / "h.bias.dat", h_bias, fmt=" %10.6f")
    np.savetxt(data_dir / "J.bias.dat", J_bias, fmt=" %10.6f")

    # Load and process main h and J files
    h = np.loadtxt(data_dir / f"h.{tag}.dat")
    J = np.loadtxt(data_dir / f"J.{tag}.dat")

    # Remove gap states (first state at each site)
    block0 = 0
    for i in range(nsites):
        h = np.delete(h, block0, axis=0)
        J = np.delete(J, block0, axis=0)
        J = np.delete(J, block0, axis=1)
        block0 += nsubs[i]

    h *= -kT
    J *= -kT

    h_model = h + h_bias
    J_model = J + J_bias

    np.savetxt(data_dir / "h.model.dat", h_model, fmt=" %10.6f")
    np.savetxt(data_dir / "J.model.dat", J_model, fmt=" %10.6f")

    # Process bootstrap samples
    for iB in range(nbs):
        logger.debug(f"Processing bootstrap sample {iB}")
        h = np.loadtxt(data_dir / f"h.bs{iB}.{tag}.dat")
        J = np.loadtxt(data_dir / f"J.bs{iB}.{tag}.dat")

        block0 = 0
        for i in range(nsites):
            h = np.delete(h, block0, axis=0)
            J = np.delete(J, block0, axis=0)
            J = np.delete(J, block0, axis=1)
            block0 += nsubs[i]

        h *= -kT
        J *= -kT

        np.savetxt(data_dir / f"h.bs{iB}.model.dat", h + h_bias, fmt=" %10.6f")
        np.savetxt(data_dir / f"J.bs{iB}.model.dat", J + J_bias, fmt=" %10.6f")

    return DCAResult(h_model=h_model, J_model=J_model)


def get_variance_dca(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    nf: int,
    analysis_idx: int,
    nbs: int = 50,
) -> DCAResult:
    """Compute free energies using Potts model estimator.

    Alternative to histogram-based estimator for systems with many sites.
    Will bail out if there are more than 2^20 alchemical states.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        nf: Number of independent trials during production.
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis/5/).
        nbs: Number of bootstrap samples.

    Returns:
        DCAResult containing free energy values and errors.
    """
    work_dir = Path(work_dir)
    analysis_dir = work_dir / "analysis" / str(analysis_idx)
    data_dir = analysis_dir / "data"

    return _compute_variance_dca(analysis_dir, data_dir, alf_info, nf, nbs)


def _compute_variance_dca(
    analysis_dir: Path,
    data_dir: Path,
    alf_info: ALFInfo | dict,
    nf: int,
    nbs: int,
) -> DCAResult:
    """Internal implementation of DCA variance calculation.

    Args:
        analysis_dir: Path to analysis directory.
        data_dir: Path to data directory with h/J files.
        alf_info: ALF configuration.
        nf: Number of independent trials.
        nbs: Number of bootstrap samples.

    Returns:
        DCAResult with free energy values and errors.
    """
    # Normalize alf_info to ALFInfo dataclass
    alf_info = ensure_alf_info(alf_info)
    temp = alf_info.temp
    nsubs = np.array(alf_info.nsubs) + 0  # Copy

    kT = 0.001987 * temp

    # Check state space size
    if np.prod(nsubs) > 1024 * 1024:
        raise ValueError("Too many states (>2^20) for DCA variance calculation")

    # Add gap states
    nblocks = int(np.sum(nsubs)) + len(nsubs)
    nsubs = nsubs + 1
    nlig = int(np.prod(nsubs))
    nlig_ng = int(np.prod(nsubs - 1))  # No gaps

    # Determine estimator type
    if (data_dir / "h.LM.dat").exists():
        tag = "LM"
    elif (data_dir / "h.PLM.dat").exists():
        tag = "PLM"
    else:
        raise FileNotFoundError(
            f"Neither h.LM.dat nor h.PLM.dat found in {data_dir}"
        )

    # Load bias parameters from analysis directory
    b = np.loadtxt(analysis_dir / "b_prev.dat").flatten()
    b_corr_file = analysis_dir / "b_corr.dat"
    if b_corr_file.exists():
        b = b + np.loadtxt(b_corr_file).flatten()
    c = np.loadtxt(analysis_dir / "c_prev.dat")
    x = np.loadtxt(analysis_dir / "x_prev.dat")
    s = np.loadtxt(analysis_dir / "s_prev.dat")

    # Build index mappings
    ind = np.zeros((nlig, len(nsubs)), dtype=int)
    for i in range(1, nlig):
        ind[i, :] = ind[i - 1, :]
        for j in range(len(nsubs) - 1, -1, -1):
            if (ind[i, j] + 1) < nsubs[j]:
                ind[i, j] += 1
                break
            else:
                ind[i, j] = 0

    blk = copy.deepcopy(ind)
    for i in range(1, len(nsubs)):
        blk[:, i:] += nsubs[i - 1]

    # No-gap indexing
    ind_ng = np.zeros((nlig_ng, len(nsubs)), dtype=int)
    for i in range(1, nlig_ng):
        ind_ng[i, :] = ind_ng[i - 1, :]
        for j in range(len(nsubs) - 1, -1, -1):
            if (ind_ng[i, j] + 1) < (nsubs[j] - 1):
                ind_ng[i, j] += 1
                break
            else:
                ind_ng[i, j] = 0

    blk_ng = copy.deepcopy(ind_ng)
    for i in range(1, len(nsubs)):
        blk_ng[:, i:] += nsubs[i - 1] - 1

    # Load main h and J files
    h = np.loadtxt(data_dir / f"h.{tag}.dat")
    J = np.loadtxt(data_dir / f"J.{tag}.dat")

    # Compute free energies
    G = np.zeros((nlig_ng,))
    jno0 = 0

    for j in range(nlig):
        Epotts = (
            np.sum(h[blk[j, :]])
            + 0.5 * np.sum(np.sum(J[blk[j, :]][:, blk[j, :]]))
        )

        if np.all(ind[j, :] > 0):
            LList = np.zeros((nblocks - len(nsubs),))
            LList[blk_ng[jno0, :]] = 1

            E = (
                np.dot(LList, b)
                + np.dot(np.dot(LList, c), LList)
                + np.dot(np.dot(1 - np.exp(-5.56 * LList), x), LList)
                + np.dot(np.dot(LList / (LList + 0.017), s), LList)
            )
            G[jno0] = E - kT * Epotts
            jno0 += 1

    # Load bootstrap samples
    h_bs = np.zeros((1, nblocks, nbs))
    J_bs = np.zeros((nblocks, nblocks, nbs))

    for i in range(nbs):
        logger.debug(f"Loading bootstrap sample {i}")
        h_bs[:, :, i] = np.loadtxt(data_dir / f"h.bs{i}.{tag}.dat")
        J_bs[:, :, i] = np.loadtxt(data_dir / f"J.bs{i}.{tag}.dat")

    # Compute bootstrap free energies
    GS = np.zeros((nbs, nlig_ng))
    Error = np.zeros((nlig_ng,))

    jno0 = 0
    for j in range(nlig):
        if np.all(ind[j, :] > 0):
            LList = np.zeros((nblocks - len(nsubs),))
            LList[blk_ng[jno0, :]] = 1

            E = (
                np.dot(LList, b)
                + np.dot(np.dot(LList, c), LList)
                + np.dot(np.dot(1 - np.exp(-5.56 * LList), x), LList)
                + np.dot(np.dot(LList / (LList + 0.017), s), LList)
            )

            for i in range(nbs):
                Epotts = (
                    np.sum(h_bs[0, blk[j, :], i])
                    + 0.5 * np.sum(np.sum(J_bs[blk[j, :]][:, blk[j, :], i]))
                )
                GS[i, jno0] = E - kT * Epotts

            Error[jno0] = np.sqrt(np.mean((GS[:, jno0] - G[jno0]) ** 2, axis=0))
            jno0 += 1

    Value = G - G[0]

    # Save results to analysis directory
    np.savetxt(analysis_dir / "G.dat", G)
    np.savetxt(analysis_dir / "GS.dat", GS)

    with open(analysis_dir / "Result.txt", "w") as fp:
        for i in range(nlig_ng):
            for j in range(len(nsubs)):
                fp.write(f"{ind_ng[i, j]:2d} ")
            fp.write(f"{Value[i]:8.3f} +/- {Error[i]:5.3f}\n")

    return DCAResult(
        values=Value,
        errors=Error,
        indices=ind_ng,
    )


def bootstrap_moments_dca(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    nf: int,
    analysis_idx: int,
    nbs: int = 50,
    seed: int = 2401,
) -> None:
    """Bootstrap moments for Potts model analysis.

    Reads first and second moment files for each independent trial,
    averages them, and creates bootstrap samples.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF configuration dictionary or ALFInfo dataclass.
        nf: Number of independent trials during production.
        analysis_idx: Index of the analysis directory (e.g., 5 for analysis/5/).
        nbs: Number of bootstrap samples to create.
        seed: Random seed for bootstrap reproducibility (default 2401).

    Note:
        Creates the following files:
        - m1.obs.dat, m2.obs.dat: Averaged moments
        - m1.bs{i}.obs.dat, m2.bs{i}.obs.dat: Bootstrap samples
        - bs{i}.dat: Bootstrap indices used
    """
    work_dir = Path(work_dir)
    analysis_dir = work_dir / "analysis" / str(analysis_idx)
    data_dir = analysis_dir / "data"

    _compute_bootstrap_moments(data_dir, alf_info, nf, nbs, seed)


def _compute_bootstrap_moments(
    data_dir: Path,
    alf_info: ALFInfo | dict,
    nf: int,
    nbs: int,
    seed: int,
) -> None:
    """Internal implementation of bootstrap moments.

    Args:
        data_dir: Path to data directory with moment files.
        alf_info: ALF configuration.
        nf: Number of independent trials.
        nbs: Number of bootstrap samples.
        seed: Random seed.
    """
    # Normalize alf_info to ALFInfo dataclass
    alf_info = ensure_alf_info(alf_info)
    nsubs = np.array(alf_info.nsubs) + 0  # Copy

    # Add gap states
    nsubs = nsubs + 1
    nblocks = int(np.sum(nsubs))
    nsites = len(nsubs)

    nblocks += nsites

    # Load moment files for each trial
    m1 = np.zeros((1, nblocks, nf))
    m2 = np.zeros((nblocks, nblocks, nf))

    for ifile in range(nf):
        m1[:, :, ifile] = np.loadtxt(data_dir / f"m1.{ifile}.obs.dat")
        m2[:, :, ifile] = np.loadtxt(data_dir / f"m2.{ifile}.obs.dat")

    # Save averaged moments
    np.savetxt(data_dir / "m1.obs.dat", np.mean(m1, axis=2))
    np.savetxt(data_dir / "m2.obs.dat", np.mean(m2, axis=2))

    # Create bootstrap samples with isolated RNG
    rng = np.random.default_rng(seed)
    for i in range(nbs):
        bs = rng.integers(0, nf, size=(nf,))
        m1mean = np.mean(m1[:, :, bs], axis=2)
        m2mean = np.mean(m2[:, :, bs], axis=2)

        np.savetxt(data_dir / f"bs{i}.dat", bs, fmt="%d")
        np.savetxt(data_dir / f"m1.bs{i}.obs.dat", m1mean)
        np.savetxt(data_dir / f"m2.bs{i}.obs.dat", m2mean)
