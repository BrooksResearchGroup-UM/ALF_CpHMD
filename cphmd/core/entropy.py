"""Entropy computation for implicit constraints (G_imp).

Computes the entropic contribution of FNEX or FPIE constraints
via Monte Carlo sampling. Results are cached in ~/.cache/cphmd/G_imp/.

The G_imp values correct for the non-uniform sampling induced by the
implicit constraint. Without this correction, WHAM would try to flatten
out the beneficial endpoint focusing that the constraint provides.

Usage:
    >>> from cphmd.core.entropy import compute_g_imp, ensure_g_imp_available
    >>> G1, G2 = compute_g_imp("fnex", ndim=2, fnex=5.5)
    >>> g_imp_dir = ensure_g_imp_available("fnex", nsubs=[2, 3], fnex=5.5)
"""

from __future__ import annotations

import hashlib
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _available_cores() -> int:
    """Get number of CPU cores available to this process.

    Respects SLURM cgroup restrictions via os.sched_getaffinity,
    unlike os.cpu_count() which returns the total node count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_NMC = 5_000_000  # Monte Carlo samples
DEFAULT_BINS = 32        # 2D bins (1D = bins²)
CUTLSUM = 0.8            # Conditional threshold for G12
MAX_BUNDLED_NDIM = 15    # Maximum ndim in bundled data


def get_cache_dir() -> Path:
    """Get the G_imp cache directory.

    Returns:
        Path to ~/.cache/cphmd/G_imp/
    """
    cache_dir = Path.home() / ".cache" / "cphmd" / "G_imp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(
    constraint_type: str,
    bins: int,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    cutlsum: float = 0.8,
) -> Path:
    """Get cache directory path for given constraint parameters.

    Args:
        constraint_type: "fnex" or "fpie"
        bins: Bin count for 2D histogram (1D uses bins²)
        fnex: FNEX parameter (if constraint_type="fnex")
        fpie_width: FPIE well width (if constraint_type="fpie")
        fpie_force: FPIE force constant (if constraint_type="fpie")
        cutlsum: G12 conditional threshold (only appended when non-default)

    Returns:
        Path to cache directory (e.g., ~/.cache/cphmd/G_imp/fnex_5.5_bins32/)
    """
    if constraint_type == "fnex":
        dirname = f"fnex_{fnex}_bins{bins}"
    else:
        dirname = f"fpie_{fpie_width}_{fpie_force}_bins{bins}"

    if cutlsum != 0.8:
        dirname += f"_cut{cutlsum}"

    return get_cache_dir() / dirname


def _solve_fpie_constraint(
    theta: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> float:
    """Newton solver: find θ₀ such that Σf(θᵢ - θ₀) = 1.

    Where f(x) = x³ if x > 0, else 0 (cubic polynomial).

    This matches the CHARMM/BLaDE implementation in msld.cu.

    Args:
        theta: Array of theta values for all substituents.
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.

    Returns:
        Solved θ₀ value.
    """
    # Initial guess: max(theta) - 1
    x0 = np.max(theta) - 1.0

    for _ in range(max_iter):
        # Constraint: c = Σ(θᵢ - x0)³ - 1 (for θᵢ > x0)
        dist = theta - x0
        dist_pos = np.maximum(dist, 0)
        c = np.sum(dist_pos ** 3) - 1.0

        # Derivative: dc/dx0 = -Σ 3(θᵢ - x0)² (negative because d/dx0)
        dc = -np.sum(3 * dist_pos ** 2)

        if abs(dc) < 1e-15:
            # Avoid division by zero
            break

        # Newton step
        x1 = x0 - c / dc

        if abs(x1 - x0) <= tol:
            return x1
        x0 = x1

    return x0


def _compute_fnex_lambda(theta: np.ndarray, fnex: float) -> np.ndarray:
    """Compute lambda values using FNEX constraint.

    λ = exp(c·sin(πθ - π/2)) / Σexp(c·sin(πθ - π/2))

    Args:
        theta: Random theta values in [0, 1).
        fnex: FNEX parameter (typically 5.5).

    Returns:
        Normalized lambda values.
    """
    # sin(πθ - π/2) = -cos(πθ)
    unnorm = np.exp(fnex * np.sin(np.pi * theta - np.pi / 2))
    return unnorm / np.sum(unnorm)


def _compute_fpie_lambda(theta: np.ndarray) -> np.ndarray:
    """Compute lambda values using FPIE constraint.

    λᵢ = (θᵢ - θ₀)³ for θᵢ > θ₀, else 0

    Note: fpie_width and fpie_force affect dynamics but not the
    lambda mapping for entropy calculation.

    Args:
        theta: Random theta values.

    Returns:
        Lambda values from cubic polynomial constraint.
    """
    theta0 = _solve_fpie_constraint(theta)
    dist = theta - theta0
    return np.where(dist > 0, dist ** 3, 0)


# Numba-optimized versions if available
if HAS_NUMBA:
    @jit(nopython=True)
    def _solve_fpie_constraint_numba(theta: np.ndarray) -> float:
        """Numba-optimized Newton solver."""
        x0 = np.max(theta) - 1.0
        tol = 1e-12

        for _ in range(50):
            dist = theta - x0
            c = 0.0
            dc = 0.0
            for i in range(len(dist)):
                if dist[i] > 0:
                    c += dist[i] ** 3
                    dc -= 3 * dist[i] ** 2
            c -= 1.0

            if abs(dc) < 1e-15:
                break

            x1 = x0 - c / dc
            if abs(x1 - x0) <= tol:
                return x1
            x0 = x1

        return x0

    @jit(nopython=True, parallel=True)
    def _compute_histogram_1d_fnex_numba(
        nmc: int,
        ndim: int,
        bins: int,
        fnex: float,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized 1D histogram for FNEX."""
        np.random.seed(seed)
        bins_1d = bins * bins  # 1D uses bins²
        histogram = np.zeros(bins_1d)

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            unnorm = np.exp(fnex * np.sin(np.pi * theta - np.pi / 2))
            lam = unnorm / np.sum(unnorm)

            for j in range(ndim):
                ind = int(np.floor(lam[j] * bins_1d))
                if 0 <= ind < bins_1d:
                    histogram[ind] += 1

        return histogram

    @jit(nopython=True, parallel=True)
    def _compute_histogram_2d_fnex_numba(
        nmc: int,
        ndim: int,
        bins: int,
        fnex: float,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized 2D histogram for FNEX."""
        np.random.seed(seed)
        histogram = np.zeros((bins, bins))

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            unnorm = np.exp(fnex * np.sin(np.pi * theta - np.pi / 2))
            lam = unnorm / np.sum(unnorm)

            for j in range(ndim):
                for k in range(j + 1, ndim):
                    ind_j = int(np.floor(lam[j] * bins))
                    ind_k = int(np.floor(lam[k] * bins))
                    if 0 <= ind_j < bins and 0 <= ind_k < bins:
                        histogram[ind_j, ind_k] += 1

        return histogram

    @jit(nopython=True, parallel=True)
    def _compute_histogram_1d_fpie_numba(
        nmc: int,
        ndim: int,
        bins: int,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized 1D histogram for FPIE."""
        np.random.seed(seed)
        bins_1d = bins * bins
        histogram = np.zeros(bins_1d)

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            theta0 = _solve_fpie_constraint_numba(theta)

            for j in range(ndim):
                dist = theta[j] - theta0
                lam_j = dist ** 3 if dist > 0 else 0.0
                ind = int(np.floor(lam_j * bins_1d))
                if 0 <= ind < bins_1d:
                    histogram[ind] += 1

        return histogram

    @jit(nopython=True, parallel=True)
    def _compute_histogram_2d_fpie_numba(
        nmc: int,
        ndim: int,
        bins: int,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized 2D histogram for FPIE."""
        np.random.seed(seed)
        histogram = np.zeros((bins, bins))

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            theta0 = _solve_fpie_constraint_numba(theta)

            lam = np.zeros(ndim)
            for j in range(ndim):
                dist = theta[j] - theta0
                lam[j] = dist ** 3 if dist > 0 else 0.0

            for j in range(ndim):
                for k in range(j + 1, ndim):
                    ind_j = int(np.floor(lam[j] * bins))
                    ind_k = int(np.floor(lam[k] * bins))
                    if 0 <= ind_j < bins and 0 <= ind_k < bins:
                        histogram[ind_j, ind_k] += 1

        return histogram

    @jit(nopython=True, parallel=True)
    def _compute_histogram_g12_fnex_numba(
        nmc: int,
        ndim: int,
        bins: int,
        fnex: float,
        cutlsum: float,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized G12 conditional ratio histogram for FNEX."""
        np.random.seed(seed)
        bins_1d = bins * bins
        histogram = np.zeros(bins_1d)

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            unnorm = np.exp(fnex * np.sin(np.pi * theta - np.pi / 2))
            lam = unnorm / np.sum(unnorm)

            for j in range(ndim):
                for k in range(j + 1, ndim):
                    lisum = lam[j] + lam[k]
                    if lisum > cutlsum:
                        ind = int(np.floor((lam[j] / lisum) * bins_1d))
                        if 0 <= ind < bins_1d:
                            histogram[ind] += 1

        return histogram

    @jit(nopython=True, parallel=True)
    def _compute_histogram_g12_fpie_numba(
        nmc: int,
        ndim: int,
        bins: int,
        cutlsum: float,
        seed: int,
    ) -> np.ndarray:
        """Numba-optimized G12 conditional ratio histogram for FPIE."""
        np.random.seed(seed)
        bins_1d = bins * bins
        histogram = np.zeros(bins_1d)

        for _ in prange(nmc):
            theta = np.random.random(ndim)
            theta0 = _solve_fpie_constraint_numba(theta)

            lam = np.zeros(ndim)
            for j in range(ndim):
                dist = theta[j] - theta0
                lam[j] = dist ** 3 if dist > 0 else 0.0

            for j in range(ndim):
                for k in range(j + 1, ndim):
                    lisum = lam[j] + lam[k]
                    if lisum > cutlsum:
                        ind = int(np.floor((lam[j] / lisum) * bins_1d))
                        if 0 <= ind < bins_1d:
                            histogram[ind] += 1

        return histogram


def _compute_histogram_1d_chunk(args: tuple) -> np.ndarray:
    """Compute 1D histogram chunk for parallel processing."""
    chunk_size, seed, ndim, bins, constraint_type, fnex = args
    bins_1d = bins * bins

    if chunk_size <= 0:
        return np.zeros(bins_1d)

    if HAS_NUMBA:
        if constraint_type == "fnex":
            return _compute_histogram_1d_fnex_numba(chunk_size, ndim, bins, fnex, seed)
        else:
            return _compute_histogram_1d_fpie_numba(chunk_size, ndim, bins, seed)

    # Fallback to numpy
    np.random.seed(seed)
    histogram = np.zeros(bins_1d)

    for _ in range(chunk_size):
        theta = np.random.random(ndim)

        if constraint_type == "fnex":
            lam = _compute_fnex_lambda(theta, fnex)
        else:
            lam = _compute_fpie_lambda(theta)

        for j in range(ndim):
            ind = int(np.floor(lam[j] * bins_1d))
            if 0 <= ind < bins_1d:
                histogram[ind] += 1

    return histogram


def _compute_histogram_2d_chunk(args: tuple) -> np.ndarray:
    """Compute 2D histogram chunk for parallel processing."""
    chunk_size, seed, ndim, bins, constraint_type, fnex = args

    if chunk_size <= 0:
        return np.zeros((bins, bins))

    if HAS_NUMBA:
        if constraint_type == "fnex":
            return _compute_histogram_2d_fnex_numba(chunk_size, ndim, bins, fnex, seed)
        else:
            return _compute_histogram_2d_fpie_numba(chunk_size, ndim, bins, seed)

    # Fallback to numpy
    np.random.seed(seed)
    histogram = np.zeros((bins, bins))

    for _ in range(chunk_size):
        theta = np.random.random(ndim)

        if constraint_type == "fnex":
            lam = _compute_fnex_lambda(theta, fnex)
        else:
            lam = _compute_fpie_lambda(theta)

        for j in range(ndim):
            for k in range(j + 1, ndim):
                ind_j = int(np.floor(lam[j] * bins))
                ind_k = int(np.floor(lam[k] * bins))
                if 0 <= ind_j < bins and 0 <= ind_k < bins:
                    histogram[ind_j, ind_k] += 1

    return histogram


def _compute_histogram_g12_chunk(args: tuple) -> np.ndarray:
    """Compute G12 histogram chunk for parallel processing."""
    chunk_size, seed, ndim, bins, constraint_type, fnex, cutlsum = args
    bins_1d = bins * bins

    if chunk_size <= 0:
        return np.zeros(bins_1d)

    if HAS_NUMBA:
        if constraint_type == "fnex":
            return _compute_histogram_g12_fnex_numba(
                chunk_size, ndim, bins, fnex, cutlsum, seed,
            )
        else:
            return _compute_histogram_g12_fpie_numba(
                chunk_size, ndim, bins, cutlsum, seed,
            )

    # Fallback to numpy
    np.random.seed(seed)
    histogram = np.zeros(bins_1d)

    for _ in range(chunk_size):
        theta = np.random.random(ndim)

        if constraint_type == "fnex":
            lam = _compute_fnex_lambda(theta, fnex)
        else:
            lam = _compute_fpie_lambda(theta)

        for j in range(ndim):
            for k in range(j + 1, ndim):
                lisum = lam[j] + lam[k]
                if lisum > cutlsum:
                    ind = int(np.floor((lam[j] / lisum) * bins_1d))
                    if 0 <= ind < bins_1d:
                        histogram[ind] += 1

    return histogram


def compute_g_imp(
    constraint_type: str,
    ndim: int,
    bins: int = DEFAULT_BINS,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    nmc: int = DEFAULT_NMC,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute G_imp for 1D and 2D histograms.

    Returns cached result if available, otherwise computes and caches.

    Args:
        constraint_type: "fnex" or "fpie"
        ndim: Number of substituents at the site
        bins: Bin count for 2D histogram (1D uses bins²)
        fnex: FNEX parameter (if constraint_type="fnex")
        fpie_width: FPIE well width (unused for entropy, stored for cache key)
        fpie_force: FPIE force constant (unused for entropy, stored for cache key)
        nmc: Number of Monte Carlo samples
        use_cache: Whether to use/update cache

    Returns:
        Tuple of (G1, G2):
        - G1: 1D free energy array (bins²,)
        - G2: 2D free energy array (bins, bins)
    """
    if constraint_type not in ("fnex", "fpie"):
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

    if ndim < 2:
        raise ValueError(f"ndim must be >= 2, got {ndim}")

    # Check cache
    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)
    g1_file = cache_dir / f"G1_{ndim}.dat"
    g2_file = cache_dir / f"G2_{ndim}.dat"

    if use_cache and g1_file.exists() and g2_file.exists():
        logger.info(f"Loading cached G_imp from {cache_dir}")
        G1 = np.loadtxt(g1_file)
        G2 = np.loadtxt(g2_file)
        return G1, G2

    num_cores = _available_cores()
    logger.info(
        f"Computing G_imp for {constraint_type} ndim={ndim} bins={bins} "
        f"(nmc={nmc:,}, cores={num_cores})"
    )

    if HAS_NUMBA:
        # Numba @jit(parallel=True) already parallelises via prange threads.
        # Do NOT wrap in multiprocessing.Pool — that causes a
        # processes × threads memory explosion (OOM on SLURM).
        import numba
        numba.set_num_threads(min(num_cores, numba.config.NUMBA_NUM_THREADS))

        if constraint_type == "fnex":
            histogram_1d = _compute_histogram_1d_fnex_numba(
                nmc, ndim, bins, fnex, 42,
            )
            histogram_2d = _compute_histogram_2d_fnex_numba(
                nmc, ndim, bins, fnex, 43,
            )
        else:
            histogram_1d = _compute_histogram_1d_fpie_numba(
                nmc, ndim, bins, 42,
            )
            histogram_2d = _compute_histogram_2d_fpie_numba(
                nmc, ndim, bins, 43,
            )
    else:
        # Numpy fallback: use multiprocessing Pool for parallelism
        chunk_size = max(1, nmc // num_cores)

        chunks_1d = [
            (chunk_size, i * 1000, ndim, bins, constraint_type, fnex)
            for i in range(num_cores)
        ]
        chunks_2d = [
            (chunk_size, i * 1000 + 500, ndim, bins, constraint_type, fnex)
            for i in range(num_cores)
        ]

        remainder = nmc % num_cores
        if remainder > 0:
            chunks_1d.append(
                (remainder, num_cores * 1000, ndim, bins, constraint_type, fnex)
            )
            chunks_2d.append(
                (remainder, num_cores * 1000 + 500, ndim, bins, constraint_type, fnex)
            )

        with Pool(num_cores) as pool:
            results_1d = pool.map(_compute_histogram_1d_chunk, chunks_1d)
            results_2d = pool.map(_compute_histogram_2d_chunk, chunks_2d)

        bins_1d = bins * bins
        histogram_1d = np.zeros(bins_1d)
        for result in results_1d:
            histogram_1d += result

        histogram_2d = np.zeros((bins, bins))
        for result in results_2d:
            histogram_2d += result

    # Symmetrize 2D histogram
    histogram_2d = histogram_2d + histogram_2d.T

    # Convert to free energy: G = -ln(P) + const
    # Add small regularization to prevent log(0)
    histogram_1d_safe = histogram_1d + 1e-300
    histogram_2d_safe = histogram_2d + 1e-300

    S1 = np.log(histogram_1d_safe)
    G1 = -S1 + np.log(np.mean(histogram_1d_safe))

    S2 = np.log(histogram_2d_safe)
    G2 = -S2 + np.log(np.mean(histogram_2d_safe))

    # Cache results
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(g1_file, G1)
        np.savetxt(g2_file, G2)
        logger.info(f"Cached G_imp to {cache_dir}")

    return G1, G2


def compute_g12(
    constraint_type: str,
    ndim: int,
    bins: int = DEFAULT_BINS,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    cutlsum: float = CUTLSUM,
    nmc: int = DEFAULT_NMC,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute G12 conditional profile via Monte Carlo.

    G12 captures the conditional ratio λ_i/(λ_i+λ_j) when
    λ_i + λ_j > cutlsum. This corrects for pairwise correlations
    imposed by the FNEX/FPIE constraint.

    Args:
        constraint_type: "fnex" or "fpie"
        ndim: Number of substituents at the site
        bins: Bin count (1D histogram uses bins²)
        fnex: FNEX parameter (if constraint_type="fnex")
        fpie_width: FPIE well width
        fpie_force: FPIE force constant
        cutlsum: Conditional threshold (default 0.8)
        nmc: Number of Monte Carlo samples
        use_cache: Whether to use/update cache

    Returns:
        G12 array of shape (bins²,) — symmetrized conditional profile.
    """
    if constraint_type not in ("fnex", "fpie"):
        raise ValueError(f"Unknown constraint_type: {constraint_type}")
    if ndim < 2:
        raise ValueError(f"ndim must be >= 2, got {ndim}")

    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force, cutlsum)
    g12_file = cache_dir / f"G12_{ndim}.dat"

    if use_cache and g12_file.exists():
        logger.info(f"Loading cached G12 from {g12_file}")
        return np.loadtxt(g12_file)

    bins_1d = bins * bins
    num_cores = _available_cores()
    logger.info(
        f"Computing G12 for {constraint_type} ndim={ndim} bins={bins} "
        f"(nmc={nmc:,}, cores={num_cores})"
    )

    if HAS_NUMBA:
        import numba
        numba.set_num_threads(min(num_cores, numba.config.NUMBA_NUM_THREADS))

        if constraint_type == "fnex":
            histogram = _compute_histogram_g12_fnex_numba(
                nmc, ndim, bins, fnex, cutlsum, 44,
            )
        else:
            histogram = _compute_histogram_g12_fpie_numba(
                nmc, ndim, bins, cutlsum, 44,
            )
    else:
        chunk_size = max(1, nmc // num_cores)
        chunks = [
            (chunk_size, i * 1000 + 700, ndim, bins, constraint_type, fnex, cutlsum)
            for i in range(num_cores)
        ]
        remainder = nmc % num_cores
        if remainder > 0:
            chunks.append(
                (remainder, num_cores * 1000 + 700, ndim, bins,
                 constraint_type, fnex, cutlsum)
            )

        with Pool(num_cores) as pool:
            results = pool.map(_compute_histogram_g12_chunk, chunks)

        histogram = np.zeros(bins_1d)
        for result in results:
            histogram += result

    # Symmetrize: histogram + reversed histogram
    histogram = histogram + histogram[::-1]

    # Convert to free energy
    histogram_safe = histogram + 1e-300
    G12 = -np.log(histogram_safe) + np.log(np.mean(histogram_safe))

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(g12_file, G12)
        logger.info(f"Cached G12 to {g12_file}")

    return G12


def compute_g1_cross(
    constraint_type: str,
    ndim_i: int,
    ndim_j: int,
    bins: int = DEFAULT_BINS,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    nmc: int = DEFAULT_NMC,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute cross-site G1_{i}_{j} from marginals of G1_i and G1_j.

    No Monte Carlo sampling needed — derived from existing G1 profiles.
    Each G1 is marginalized from its (bins, bins) shape to (bins,),
    then the outer sum gives the cross-site 2D profile.

    Args:
        constraint_type: "fnex" or "fpie"
        ndim_i: Number of substituents at site i
        ndim_j: Number of substituents at site j
        bins: Bin count
        fnex: FNEX parameter
        fpie_width: FPIE well width
        fpie_force: FPIE force constant
        nmc: MC samples (passed through if G1 needs computing)
        use_cache: Whether to use/update cache

    Returns:
        Cross-site profile of shape (bins, bins).
    """
    if ndim_i < 2 or ndim_j < 2:
        raise ValueError(f"Both ndim values must be >= 2, got {ndim_i}, {ndim_j}")

    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)
    cross_file = cache_dir / f"G1_{ndim_i}_{ndim_j}.dat"

    if use_cache and cross_file.exists():
        logger.info(f"Loading cached G1_cross from {cross_file}")
        return np.loadtxt(cross_file)

    logger.info(f"Computing G1_cross for ndim_i={ndim_i}, ndim_j={ndim_j}")

    def _marginalize_g1(ndim: int) -> np.ndarray:
        """Load G1, reshape to (bins, bins), marginalize to (bins,)."""
        G1, _ = compute_g_imp(
            constraint_type=constraint_type,
            ndim=ndim,
            bins=bins,
            fnex=fnex,
            fpie_width=fpie_width,
            fpie_force=fpie_force,
            nmc=nmc,
            use_cache=True,
        )
        # G1 is flat (bins²,) → reshape to (bins, bins) and marginalize
        g1_2d = G1.reshape(bins, bins)
        h = np.sum(np.exp(-g1_2d), axis=1)  # marginalize over second dim
        h_safe = h + 1e-300
        return -np.log(h_safe) + np.log(np.mean(h_safe))

    marginal_i = _marginalize_g1(ndim_i)
    marginal_j = _marginalize_g1(ndim_j)

    # Outer sum: G1_cross[a, b] = marginal_i[a] + marginal_j[b]
    G1_cross = marginal_i[:, None] + marginal_j[None, :]

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(cross_file, G1_cross)
        logger.info(f"Cached G1_cross to {cross_file}")

    return G1_cross


def ensure_g_imp_available(
    constraint_type: str,
    nsubs: list[int] | np.ndarray,
    bins: int = DEFAULT_BINS,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    cutlsum: float = CUTLSUM,
    nmc: int = DEFAULT_NMC,
) -> Path:
    """Ensure G_imp files exist for all sites, computing if needed.

    Args:
        constraint_type: "fnex" or "fpie"
        nsubs: Array of substituent counts per site
        bins: Bin count for 2D histogram
        fnex: FNEX parameter
        fpie_width: FPIE well width
        fpie_force: FPIE force constant
        cutlsum: G12 conditional threshold
        nmc: Monte Carlo samples if computing

    Returns:
        Path to cache directory containing all G_imp files
    """
    import shutil

    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force, cutlsum)
    # G1/G2 are cutlsum-independent — they live in the base cache
    base_cache = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)

    # Compute G_imp for each unique subsite count.
    # CUDA indexes G_imp files by dimension: G1_{nsubs}.dat, G12_{nsubs}.dat, etc.
    unique_ndims = sorted(set(nsubs)) if len(nsubs) > 0 else []
    for ndim in unique_ndims:
        # G1 + G2 (1D marginal + 2D intra-site joint) — written to base_cache
        compute_g_imp(
            constraint_type=constraint_type,
            ndim=ndim,
            bins=bins,
            fnex=fnex,
            fpie_width=fpie_width,
            fpie_force=fpie_force,
            nmc=nmc,
            use_cache=True,
        )
        # G12 (conditional ratio profile) — written to cache_dir (cutlsum-aware)
        compute_g12(
            constraint_type=constraint_type,
            ndim=ndim,
            bins=bins,
            fnex=fnex,
            fpie_width=fpie_width,
            fpie_force=fpie_force,
            cutlsum=cutlsum,
            nmc=nmc,
            use_cache=True,
        )

    # Cross-site profiles: all (i, j) pairs of unique ndims — written to base_cache
    for ndim_i in unique_ndims:
        for ndim_j in unique_ndims:
            compute_g1_cross(
                constraint_type=constraint_type,
                ndim_i=ndim_i,
                ndim_j=ndim_j,
                bins=bins,
                fnex=fnex,
                fpie_width=fpie_width,
                fpie_force=fpie_force,
                nmc=nmc,
                use_cache=True,
            )

    # When cutlsum != default, G1/G2/G1_cross live in base_cache while G12 is
    # in cache_dir.  Copy the cutlsum-independent files into cache_dir so CUDA
    # finds everything in one directory.
    if cache_dir != base_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ndim in unique_ndims:
            for prefix in ("G1", "G2"):
                src = base_cache / f"{prefix}_{ndim}.dat"
                dst = cache_dir / f"{prefix}_{ndim}.dat"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        for ndim_i in unique_ndims:
            for ndim_j in unique_ndims:
                src = base_cache / f"G1_{ndim_i}_{ndim_j}.dat"
                dst = cache_dir / f"G1_{ndim_i}_{ndim_j}.dat"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

    return cache_dir


def clear_cache(
    constraint_type: str | None = None,
    bins: int | None = None,
    fnex: float | None = None,
    fpie_width: float | None = None,
    fpie_force: float | None = None,
) -> int:
    """Clear cached G_imp files.

    If parameters are specified, only matching cache entries are cleared.
    If no parameters are specified, clears all cached entries.

    Args:
        constraint_type: Filter by constraint type
        bins: Filter by bin count
        fnex: Filter by fnex value
        fpie_width: Filter by fpie_width
        fpie_force: Filter by fpie_force

    Returns:
        Number of cache directories removed
    """
    import shutil

    cache_dir = get_cache_dir()
    count = 0

    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue

        name = entry.name

        # Parse cache entry name
        if name.startswith("fnex_"):
            entry_type = "fnex"
            parts = name.split("_")
            entry_fnex = float(parts[1])
            entry_bins = int(parts[2].replace("bins", ""))
            entry_fpie_width = None
            entry_fpie_force = None
        elif name.startswith("fpie_"):
            entry_type = "fpie"
            parts = name.split("_")
            entry_fpie_width = float(parts[1])
            entry_fpie_force = float(parts[2])
            entry_bins = int(parts[3].replace("bins", ""))
            entry_fnex = None
        else:
            continue

        # Check filters
        if constraint_type is not None and entry_type != constraint_type:
            continue
        if bins is not None and entry_bins != bins:
            continue
        if fnex is not None and entry_fnex != fnex:
            continue
        if fpie_width is not None and entry_fpie_width != fpie_width:
            continue
        if fpie_force is not None and entry_fpie_force != fpie_force:
            continue

        # Remove matching entry
        shutil.rmtree(entry)
        count += 1
        logger.info(f"Removed cache entry: {entry}")

    return count
