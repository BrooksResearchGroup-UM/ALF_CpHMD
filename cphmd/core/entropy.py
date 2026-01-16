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
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_NMC = 5_000_000  # Monte Carlo samples
DEFAULT_BINS = 32        # 2D bins (1D = bins²)


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
) -> Path:
    """Get cache directory path for given constraint parameters.

    Args:
        constraint_type: "fnex" or "fpie"
        bins: Bin count for 2D histogram (1D uses bins²)
        fnex: FNEX parameter (if constraint_type="fnex")
        fpie_width: FPIE well width (if constraint_type="fpie")
        fpie_force: FPIE force constant (if constraint_type="fpie")

    Returns:
        Path to cache directory (e.g., ~/.cache/cphmd/G_imp/fnex_5.5_bins32/)
    """
    if constraint_type == "fnex":
        dirname = f"fnex_{fnex}_bins{bins}"
    else:
        dirname = f"fpie_{fpie_width}_{fpie_force}_bins{bins}"

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

    logger.info(
        f"Computing G_imp for {constraint_type} ndim={ndim} bins={bins} "
        f"(nmc={nmc:,})"
    )

    # Parallel computation
    num_cores = cpu_count()
    chunk_size = max(1, nmc // num_cores)

    # Build chunks with different random seeds
    chunks_1d = [
        (chunk_size, i * 1000, ndim, bins, constraint_type, fnex)
        for i in range(num_cores)
    ]
    chunks_2d = [
        (chunk_size, i * 1000 + 500, ndim, bins, constraint_type, fnex)
        for i in range(num_cores)
    ]

    # Handle remainder
    remainder = nmc % num_cores
    if remainder > 0:
        chunks_1d.append((remainder, num_cores * 1000, ndim, bins, constraint_type, fnex))
        chunks_2d.append((remainder, num_cores * 1000 + 500, ndim, bins, constraint_type, fnex))

    # Compute histograms in parallel
    with Pool(num_cores) as pool:
        results_1d = pool.map(_compute_histogram_1d_chunk, chunks_1d)
        results_2d = pool.map(_compute_histogram_2d_chunk, chunks_2d)

    # Combine results
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


def ensure_g_imp_available(
    constraint_type: str,
    nsubs: list[int] | np.ndarray,
    bins: int = DEFAULT_BINS,
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
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
        nmc: Monte Carlo samples if computing

    Returns:
        Path to cache directory containing G_imp files
    """
    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)

    # Compute G_imp for each unique ndim value
    unique_ndims = set(nsubs)
    for ndim in unique_ndims:
        if ndim < 2:
            continue
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
