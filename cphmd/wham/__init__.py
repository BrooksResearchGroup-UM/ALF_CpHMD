"""WHAM and LMALF analysis module.

This module provides GPU-accelerated analysis for ALF simulations:
- WHAM (Weighted Histogram Analysis Method): Iterative histogram reweighting
- LMALF (Likelihood Maximization ALF): L-BFGS optimization of bias parameters

Both methods compute optimal bias parameters from simulation trajectories.
LMALF is an alternative to WHAM that uses direct optimization.

The library must be compiled before use:
    cd cphmd/wham && ./build.sh
    # or manually:
    nvcc -shared -Xcompiler -fPIC -o ../libwham.so wham.cu

Usage (preferred — in-memory, no intermediate files):
    from cphmd.wham import run_wham_from_memory, run_lmalf_from_memory

    run_wham_from_memory(lambda_arrays, energy_matrix, ...)
    run_lmalf_from_memory(lambda_combined, ...)

Legacy usage (file-based):
    from cphmd.wham import run_wham, run_lmalf

    run_wham(analysis_dir, nf=10, temp=298.15, nts0=1, nts1=1)
    run_lmalf(analysis_dir, nf=10, temp=298.15)
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from cphmd.core.bias_constants import derive_bias_constants

if TYPE_CHECKING:
    from cphmd.core.alf_utils import ALFInfo

logger = logging.getLogger(__name__)

# Path to the compiled WHAM shared library
_WHAM_LIB_PATH = Path(__file__).parent / "libwham.so"

# Lazy-loaded library cache: avoids repeated ctypes.CDLL() and argtypes setup
_wham_lib_cache: ctypes.CDLL | None = None


def _get_wham_lib() -> ctypes.CDLL:
    """Load and cache the WHAM shared library.

    Returns the cached library handle on subsequent calls, avoiding
    repeated dlopen() and symbol resolution overhead.

    Raises:
        FileNotFoundError: If libwham.so doesn't exist.
        RuntimeError: If the library can't be loaded (e.g. CUDA unavailable).
    """
    global _wham_lib_cache
    if _wham_lib_cache is not None:
        return _wham_lib_cache

    if not _WHAM_LIB_PATH.exists():
        raise FileNotFoundError(
            f"WHAM library not found at {_WHAM_LIB_PATH}. "
            "Please compile: cd cphmd/wham/src && make"
        )

    try:
        lib = ctypes.CDLL(str(_WHAM_LIB_PATH))
    except OSError as e:
        raise RuntimeError(
            f"Failed to load WHAM library: {e}. "
            "Ensure CUDA is available and library is compiled correctly."
        ) from e

    # Configure wham() function signature once
    lib.wham.argtypes = [
        ctypes.c_int,                   # nf
        ctypes.c_double,                # temp
        ctypes.c_int,                   # nts0
        ctypes.c_int,                   # nts1
        ctypes.c_int,                   # use_gshift
        ctypes.POINTER(ctypes.c_int),   # nsubs
        ctypes.c_int,                   # nsites
        ctypes.c_char_p,               # g_imp_path
        ctypes.c_double,                # chi_offset
        ctypes.c_double,                # omega_scale
        ctypes.c_double,                # cutlsum
        ctypes.c_double,                # chi_offset_t
        ctypes.c_double,                # chi_offset_u
        ctypes.c_int,                   # ntriangle
    ]
    lib.wham.restype = ctypes.c_int

    # Configure lmalf() function signature once
    lib.lmalf.argtypes = [
        ctypes.c_int,                   # nf
        ctypes.c_double,                # temp
        ctypes.c_int,                   # ms
        ctypes.c_int,                   # msprof
        ctypes.c_int,                   # max_iter
        ctypes.c_double,                # tolerance
        ctypes.POINTER(ctypes.c_int),   # nsubs
        ctypes.c_int,                   # nsites
        ctypes.c_char_p,               # g_imp_path
        ctypes.c_double,                # fnex
        ctypes.c_double,                # chi_offset
        ctypes.c_double,                # omega_scale
        ctypes.c_double,                # chi_offset_t
        ctypes.c_double,                # chi_offset_u
        ctypes.c_int,                   # ntriangle
    ]
    lib.lmalf.restype = ctypes.c_int

    # Configure wham_from_memory() function signature
    lib.wham_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # nts0
        ctypes.c_int,                         # nts1
        ctypes.c_int,                         # use_gshift
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # cutlsum
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # D_flat
        ctypes.POINTER(ctypes.c_int),         # sim_indices
        ctypes.POINTER(ctypes.c_int),         # frame_counts
        ctypes.c_int,                         # total_frames
        ctypes.POINTER(ctypes.c_double),      # gshift_flat
    ]
    lib.wham_from_memory.restype = ctypes.c_int

    # Configure wham_iterate_from_memory() function signature (Phase A: f-convergence)
    lib.wham_iterate_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # nts0
        ctypes.c_int,                         # nts1
        ctypes.c_int,                         # use_gshift
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # cutlsum
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # D_flat
        ctypes.POINTER(ctypes.c_int),         # sim_indices
        ctypes.POINTER(ctypes.c_int),         # frame_counts
        ctypes.c_int,                         # total_frames
        ctypes.POINTER(ctypes.c_double),      # gshift_flat
        ctypes.POINTER(ctypes.c_double),      # f_out
        ctypes.POINTER(ctypes.c_int),         # nf_out
    ]
    lib.wham_iterate_from_memory.restype = ctypes.c_int

    # Configure wham_profiles_from_memory() function signature (Phase B: profile subset)
    lib.wham_profiles_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # nts0
        ctypes.c_int,                         # nts1
        ctypes.c_int,                         # use_gshift
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # cutlsum
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # D_flat
        ctypes.POINTER(ctypes.c_int),         # sim_indices
        ctypes.POINTER(ctypes.c_int),         # frame_counts
        ctypes.c_int,                         # total_frames
        ctypes.POINTER(ctypes.c_double),      # gshift_flat
        ctypes.POINTER(ctypes.c_double),      # f_in
        ctypes.c_int,                         # f_size
        ctypes.c_int,                         # profile_start
        ctypes.c_int,                         # profile_end
        ctypes.POINTER(ctypes.c_double),      # C_out
        ctypes.POINTER(ctypes.c_double),      # V_out
        ctypes.POINTER(ctypes.c_int),         # dim_out
    ]
    lib.wham_profiles_from_memory.restype = ctypes.c_int

    # Configure wham_profiles_slim_from_memory() (Phase B with slim D + pre-computed lnDenom)
    lib.wham_profiles_slim_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # nts0
        ctypes.c_int,                         # nts1
        ctypes.c_int,                         # use_gshift
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # cutlsum
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # D_flat (slim: no cross-energies)
        ctypes.c_int,                         # ndim (slim stride)
        ctypes.POINTER(ctypes.c_int),         # sim_indices
        ctypes.POINTER(ctypes.c_int),         # frame_counts
        ctypes.c_int,                         # total_frames
        ctypes.POINTER(ctypes.c_double),      # gshift_flat
        ctypes.POINTER(ctypes.c_double),      # f_in
        ctypes.c_int,                         # f_size
        ctypes.POINTER(ctypes.c_double),      # lnDenom_in
        ctypes.c_int,                         # profile_start
        ctypes.c_int,                         # profile_end
        ctypes.POINTER(ctypes.c_double),      # C_out
        ctypes.POINTER(ctypes.c_double),      # V_out
        ctypes.POINTER(ctypes.c_int),         # dim_out
    ]
    lib.wham_profiles_slim_from_memory.restype = ctypes.c_int

    # Configure lmalf_from_memory() function signature
    lib.lmalf_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # ms
        ctypes.c_int,                         # msprof
        ctypes.c_int,                         # max_iter
        ctypes.c_double,                      # tolerance
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # fnex
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # lambda_flat
        ctypes.POINTER(ctypes.c_double),      # ensweight_flat
        ctypes.c_int,                         # n_frames
        ctypes.POINTER(ctypes.c_double),      # x_prev_flat
        ctypes.POINTER(ctypes.c_double),      # s_prev_flat
        ctypes.c_int,                         # nblocks_sq
    ]
    lib.lmalf_from_memory.restype = ctypes.c_int

    # Configure nonlinear_from_memory() function signature (L-BFGS without profiles)
    lib.nonlinear_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # ms
        ctypes.c_int,                         # msprof
        ctypes.c_int,                         # max_iter
        ctypes.c_double,                      # tolerance
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_double,                      # fnex
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # lambda_flat
        ctypes.POINTER(ctypes.c_double),      # ensweight_flat
        ctypes.c_int,                         # n_frames
        ctypes.POINTER(ctypes.c_double),      # x_prev_flat
        ctypes.POINTER(ctypes.c_double),      # s_prev_flat
        ctypes.c_int,                         # nblocks_sq
    ]
    lib.nonlinear_from_memory.restype = ctypes.c_int

    # Configure wham_compute_weights_from_memory() function signature
    lib.wham_compute_weights_from_memory.argtypes = [
        ctypes.c_int,                         # nf
        ctypes.c_double,                      # temp
        ctypes.c_int,                         # nts0
        ctypes.c_int,                         # nts1
        ctypes.c_int,                         # use_gshift
        ctypes.POINTER(ctypes.c_int),         # nsubs
        ctypes.c_int,                         # nsites
        ctypes.c_char_p,                      # g_imp_path
        ctypes.c_double,                      # chi_offset
        ctypes.c_double,                      # omega_scale
        ctypes.c_double,                      # cutlsum
        ctypes.c_double,                      # chi_offset_t
        ctypes.c_double,                      # chi_offset_u
        ctypes.c_int,                         # ntriangle
        ctypes.POINTER(ctypes.c_double),      # D_flat
        ctypes.POINTER(ctypes.c_int),         # sim_indices
        ctypes.POINTER(ctypes.c_int),         # frame_counts
        ctypes.c_int,                         # total_frames
        ctypes.POINTER(ctypes.c_double),      # gshift_flat
        ctypes.POINTER(ctypes.c_double),      # weights_out
        ctypes.POINTER(ctypes.c_double),      # f_out
        ctypes.POINTER(ctypes.c_int),         # nf_out
    ]
    lib.wham_compute_weights_from_memory.restype = ctypes.c_int

    _wham_lib_cache = lib
    return lib


def _prepare_nsubs(
    nsubs: np.ndarray | list[int] | None,
) -> tuple[np.ndarray | None, ctypes.POINTER(ctypes.c_int) | None, int]:
    """Convert nsubs to ctypes-compatible format."""
    if nsubs is not None:
        arr = np.asarray(nsubs, dtype=np.int32)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        return arr, ptr, len(arr)
    return None, None, 0


def _to_bytes(path: str | Path | None) -> bytes | None:
    """Convert a path to UTF-8 bytes for ctypes c_char_p."""
    return str(path).encode("utf-8") if path is not None else None


@contextmanager
def _chdir_context(path: Path) -> Generator[None, None, None]:
    """Context manager for safe directory change.

    WHAM expects files in current directory, so we must chdir temporarily.
    This ensures we return to original directory even if an exception occurs.

    Args:
        path: Directory to change to.

    Yields:
        None
    """
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


@contextmanager
def _redirect_c_output(log_path: Path | None) -> Generator[None, None, None]:
    """Redirect C-level stdout and stderr to a file.

    Python's contextlib.redirect_stdout only affects sys.stdout.
    C libraries using fprintf(stdout, ...) write to OS file descriptor 1
    directly, so we must redirect at the fd level with os.dup2().
    """
    if log_path is None:
        yield
        return

    import sys

    # Flush Python buffers so nothing pending goes to the wrong fd
    sys.stdout.flush()
    sys.stderr.flush()

    # Save original file descriptors
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    # Get libc for flushing C-level stdio buffers
    libc = ctypes.CDLL(None)

    try:
        log_f = open(log_path, "a")
        os.dup2(log_f.fileno(), 1)  # redirect fd 1 (C stdout)
        os.dup2(log_f.fileno(), 2)  # redirect fd 2 (C stderr)
        log_f.close()               # fd is now duplicated, safe to close
        yield
    finally:
        # Flush C-level stdio buffers BEFORE restoring fds, so buffered
        # fprintf output goes to the log file, not back to the terminal
        libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def run_wham(
    analysis_dir: str | Path,
    nf: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run WHAM analysis using bundled GPU-accelerated library.

    The WHAM routine computes profiles and linear changes to those profiles
    in response to changes in bias parameters. It also computes a quadratic
    penalty function that penalizes deviations of the profiles from their
    average values.

    This version (wham_3) supports G-shift for variable block handling.

    Args:
        analysis_dir: Path to analysis directory containing Lambda/, Energy/.
        nf: Number of simulation files to analyze.
        temp: Temperature in Kelvin.
        nts0: First terminal site index (for intersite coupling).
        nts1: Second terminal site index (for intersite profiles).
        use_gshift: If True, apply G_imp shifts from G_imp_shifts/ directory.
            If False (default), use zero shifts (legacy ALF behavior).
        nsubs: Array of subsites per site. If None, reads from prep/nsubs file.
        g_imp_path: Path to G_imp directory. If None, uses "G_imp" in analysis_dir.
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Raises:
        FileNotFoundError: If analysis directory or WHAM library not found.
        RuntimeError: If WHAM analysis fails.

    Note:
        WHAM expects the following files in the analysis directory:
        - Lambda/ directory with lambda trajectory files
        - Energy/ directory with cross-simulation energies
        - G_imp/ directory with entropy reference profiles (or custom g_imp_path)
        - G_imp_shifts/ directory with shift tables (required if use_gshift=True)

        WHAM produces:
        - multisite/C.dat: Hessian matrix
        - multisite/V.dat: Gradient vector
        - f.dat: Free energy weights

    Example:
        >>> from cphmd.wham import run_wham
        >>> run_wham("./analysis/5", nf=10, temp=298.15)
        >>> # With explicit nsubs and G_imp path:
        >>> run_wham("./analysis/5", nf=10, temp=298.15, nsubs=[2, 2], g_imp_path="/path/to/G_imp")
    """
    # Validate cutlsum (G12 conditional threshold) before any I/O
    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(
            f"cutlsum must be in (0, 1), got {cutlsum}. "
            "Typical value is 0.8 (exclude frames where λ_i + λ_j < 0.8)."
        )

    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    # Load library (cached after first call)
    lib = _get_wham_lib()

    # Create multisite output directory
    multisite_dir = analysis_dir / "multisite"
    multisite_dir.mkdir(exist_ok=True)

    # Prepare ctypes arguments
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    logger.info(f"Running WHAM with nf={nf}, temp={temp}, use_gshift={use_gshift}, fnex={fnex}")
    if nsubs_arr is not None:
        logger.info(f"  nsubs={list(nsubs_arr)}, g_imp_path={g_imp_path}")

    with _chdir_context(analysis_dir):
        with _redirect_c_output(log_path):
            result = lib.wham(
                nf, temp, nts0, nts1, int(use_gshift),
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.chi_offset, constants.omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
            )
        if result != 0:
            raise RuntimeError(f"WHAM returned error code: {result}")

    logger.info("WHAM analysis completed successfully")


def check_wham_available() -> bool:
    """Check if WHAM library is available and compiled.

    Returns:
        True if WHAM library exists and can be loaded, False otherwise.
    """
    try:
        _get_wham_lib()
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def get_wham_lib_path() -> Path:
    """Get path to the WHAM shared library.

    Returns:
        Path to libwham.so
    """
    return _WHAM_LIB_PATH


def prepare_g_imp_for_wham(
    analysis_dir: str | Path,
    alf_info: "ALFInfo | dict",
    force_recompute: bool = False,
) -> Path:
    """Prepare G_imp files for WHAM analysis.

    Ensures G_imp files exist for all sites based on constraint configuration,
    computing them if needed. Creates symlinks in the analysis directory
    pointing to the cached G_imp files.

    Args:
        analysis_dir: Path to analysis directory.
        alf_info: ALF configuration with constraint parameters.
        force_recompute: If True, recompute G_imp even if cached.

    Returns:
        Path to G_imp directory in analysis_dir.
    """
    from cphmd.core.alf_utils import ensure_alf_info
    from cphmd.core.entropy import compute_g1_cross, compute_g12, compute_g_imp, get_cache_path

    analysis_dir = Path(analysis_dir)
    alf_info = ensure_alf_info(alf_info)

    # Get constraint parameters
    constraint_type = alf_info.constraint_type
    fnex = alf_info.fnex
    fpie_width = alf_info.fpie_width
    fpie_force = alf_info.fpie_force
    bins = alf_info.g_imp_bins
    cutlsum = alf_info.cutlsum
    nsubs = alf_info.nsubs

    # G1/G2 are cutlsum-independent (base cache); G12 uses cutlsum-aware cache
    base_cache = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)
    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force, cutlsum)

    # CUDA indexes G_imp files by dimension: G1_{nsubs}.dat, G12_{nsubs}.dat, etc.
    unique_ndims = sorted(set(nsubs)) if len(nsubs) > 0 else []

    for ndim in unique_ndims:
        g1_file = base_cache / f"G1_{ndim}.dat"
        g2_file = base_cache / f"G2_{ndim}.dat"
        g12_file = cache_dir / f"G12_{ndim}.dat"

        if force_recompute or not g1_file.exists() or not g2_file.exists():
            logger.info(f"Computing G_imp for ndim={ndim}")
            compute_g_imp(
                constraint_type=constraint_type,
                ndim=ndim,
                bins=bins,
                fnex=fnex,
                fpie_width=fpie_width,
                fpie_force=fpie_force,
                use_cache=True,
            )

        if force_recompute or not g12_file.exists():
            logger.info(f"Computing G12 for ndim={ndim}")
            compute_g12(
                constraint_type=constraint_type,
                ndim=ndim,
                bins=bins,
                fnex=fnex,
                fpie_width=fpie_width,
                fpie_force=fpie_force,
                cutlsum=cutlsum,
                use_cache=True,
            )

    # Cross-site profiles (cutlsum-independent, base cache)
    for ndim_i in unique_ndims:
        for ndim_j in unique_ndims:
            cross_file = base_cache / f"G1_{ndim_i}_{ndim_j}.dat"
            if force_recompute or not cross_file.exists():
                logger.info(f"Computing G1_cross for ndim_i={ndim_i}, ndim_j={ndim_j}")
                compute_g1_cross(
                    constraint_type=constraint_type,
                    ndim_i=ndim_i,
                    ndim_j=ndim_j,
                    bins=bins,
                    fnex=fnex,
                    fpie_width=fpie_width,
                    fpie_force=fpie_force,
                    use_cache=True,
                )

    # Create G_imp directory in analysis_dir and copy files from both caches
    g_imp_dir = analysis_dir / "G_imp"
    g_imp_dir.mkdir(exist_ok=True)

    for ndim in unique_ndims:
        # G1/G2 from base cache, G12 from cutlsum cache
        for prefix, src_dir in [("G1", base_cache), ("G2", base_cache), ("G12", cache_dir)]:
            src = src_dir / f"{prefix}_{ndim}.dat"
            dst = g_imp_dir / f"{prefix}_{ndim}.dat"
            if src.exists() and (force_recompute or not dst.exists()):
                shutil.copy2(src, dst)

    # Copy cross-site files from base cache
    for ndim_i in unique_ndims:
        for ndim_j in unique_ndims:
            src = base_cache / f"G1_{ndim_i}_{ndim_j}.dat"
            dst = g_imp_dir / f"G1_{ndim_i}_{ndim_j}.dat"
            if src.exists() and (force_recompute or not dst.exists()):
                shutil.copy2(src, dst)

    logger.info(f"G_imp files prepared in {g_imp_dir}")
    return g_imp_dir


def run_wham_with_g_imp(
    analysis_dir: str | Path,
    alf_info: "ALFInfo | dict",
    nf: int,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    force_recompute_g_imp: bool = False,
) -> None:
    """Run WHAM analysis with automatic G_imp preparation.

    This is a convenience function that:
    1. Prepares G_imp files based on constraint configuration
    2. Runs WHAM analysis

    Args:
        analysis_dir: Path to analysis directory.
        alf_info: ALF configuration with constraint and temperature parameters.
        nf: Number of simulation files to analyze.
        nts0: First terminal site index.
        nts1: Second terminal site index.
        use_gshift: If True, apply G_imp shifts.
        force_recompute_g_imp: If True, recompute G_imp even if cached.

    Example:
        >>> from cphmd.wham import run_wham_with_g_imp
        >>> from cphmd.core.alf_utils import ALFInfo
        >>> alf_info = ALFInfo(
        ...     name="test",
        ...     nsubs=np.array([2, 3]),
        ...     constraint_type="fpie",
        ...     temp=298.15,
        ... )
        >>> run_wham_with_g_imp("./analysis/5", alf_info, nf=10)
    """
    from cphmd.core.alf_utils import ensure_alf_info

    alf_info = ensure_alf_info(alf_info)

    # Prepare G_imp files
    g_imp_dir = prepare_g_imp_for_wham(analysis_dir, alf_info, force_recompute_g_imp)

    # Run WHAM with nsubs and g_imp_path
    run_wham(
        analysis_dir=analysis_dir,
        nf=nf,
        temp=alf_info.temp,
        nts0=nts0,
        nts1=nts1,
        use_gshift=use_gshift,
        nsubs=alf_info.nsubs,
        g_imp_path=g_imp_dir,
        fnex=alf_info.fnex,
        cutlsum=alf_info.cutlsum,
    )


def run_lmalf(
    analysis_dir: str | Path,
    nf: int,
    temp: float,
    ms: int = 0,
    msprof: int = 0,
    max_iter: int = 0,
    tolerance: float = 0.0,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run LMALF (Likelihood Maximization ALF) analysis.

    LMALF is an alternative to WHAM that uses L-BFGS optimization to find
    optimal bias parameters by maximizing likelihood. Key differences:
    - Direct optimization instead of iterative histogram reweighting
    - Profile-based fitting with built-in L2 regularization
    - May converge faster for some systems

    Args:
        analysis_dir: Path to analysis directory containing Lambda.dat.
        nf: Number of simulation files (used for logging, data read from files).
        temp: Temperature in Kelvin.
        ms: Multisite coupling flag (0=none, 1=full coupling, 2=c-only).
        msprof: Multisite profiles flag (0=disabled, 1=enabled).
        max_iter: Maximum L-BFGS iterations (0 = use default 250).
        tolerance: Convergence tolerance (0 = use default 1.25e-3).
        nsubs: Array of subsites per site. If None, reads from prep/nsubs file.
        g_imp_path: Path to G_imp directory. If None, uses "G_imp" in analysis_dir.
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Raises:
        FileNotFoundError: If analysis directory or library not found.
        RuntimeError: If LMALF analysis fails.

    Note:
        LMALF expects the following files in the analysis directory:
        - Lambda.dat: Combined lambda trajectory [frames x nblocks]
        - ensweight.dat: Ensemble weights [frames] (optional, defaults to 1.0)
        - x_prev.dat, s_prev.dat: Previous parameters (optional, for ms=1)

        LMALF produces:
        - OUT.dat: Optimized bias parameters in flat format

        The OUT.dat must be post-processed with GetFreeEnergyLM() to
        generate the b.dat, c.dat, x.dat, s.dat files.

    Example:
        >>> from cphmd.wham import run_lmalf
        >>> run_lmalf("./analysis/5", nf=10, temp=298.15)
        >>> # With explicit nsubs:
        >>> run_lmalf("./analysis/5", nf=10, temp=298.15, nsubs=[2, 2], g_imp_path="/path/to/G_imp")
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    # Load library (cached after first call)
    lib = _get_wham_lib()

    # Verify Lambda.dat exists
    lambda_file = analysis_dir / "Lambda.dat"
    if not lambda_file.exists():
        raise FileNotFoundError(
            f"Lambda.dat not found in {analysis_dir}. "
            "LMALF requires a combined Lambda.dat file."
        )

    # Prepare ctypes arguments
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    logger.info(f"Running LMALF with nf={nf}, temp={temp}, ms={ms}, msprof={msprof}, fnex={fnex}")
    if nsubs_arr is not None:
        logger.info(f"  nsubs={list(nsubs_arr)}, g_imp_path={g_imp_path}")

    with _chdir_context(analysis_dir):
        with _redirect_c_output(log_path):
            result = lib.lmalf(
                nf, temp, ms, msprof, max_iter, tolerance,
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.fnex, constants.chi_offset, constants.omega_scale,
                chi_offset_t, chi_offset_u, ntriangle,
            )
        if result != 0:
            raise RuntimeError(f"LMALF returned error code: {result}")

    logger.info("LMALF analysis completed successfully")


def check_lmalf_available() -> bool:
    """Check if LMALF is available in the compiled library.

    Returns:
        True if LMALF function exists and can be called, False otherwise.
    """
    try:
        lib = _get_wham_lib()
        _ = lib.lmalf
        return True
    except (FileNotFoundError, RuntimeError, AttributeError):
        return False


def _pack_wham_data(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pack lambda trajectories and cross-energies into flat D_h layout for CUDA.

    Matches the readdata() packing in wham.cu (lines 420-456):
    - D[t*ndim + 0]           = E_self (energy_matrix[nf-1][i] for sim i)
    - D[t*ndim + 1..NL]       = lambda values
    - D[t*ndim + NL+1..NL+nf] = cross-energies (energy_matrix[j][i] for all j)
    - D[t*ndim + NL+nf+1]     = 0.0  (reserved bin_1D)
    - D[t*ndim + NL+nf+2]     = 0.0  (reserved bin_2D)

    ndim = nblocks + nf + 3  (NL + NF + 1 (E_self) + 2 (bins))

    Args:
        lambda_arrays: Per-simulation lambda trajectories. lambda_arrays[i] has
            shape (n_frames_i, nblocks).
        energy_matrix: Cross-simulation energies. energy_matrix[j][i] has shape
            (n_frames_i, 1): bias energy of simulation i's frames under
            simulation j's parameters.
        nblocks: Total number of lambda blocks (NL).
        nf: Number of simulations (NF).

    Returns:
        D_flat: float64 array of shape (total_frames * ndim,).
        sim_indices: int32 array of shape (total_frames,) — simulation index per frame.
        frame_counts: int32 array of shape (nf,) — frames per simulation.
        total_frames: Total number of frames across all simulations.
    """
    NL = nblocks
    ndim = NL + nf + 3

    # Count total frames and per-sim frame counts
    frame_counts = np.array([lambda_arrays[i].shape[0] for i in range(nf)], dtype=np.int32)
    total_frames = int(frame_counts.sum())

    # Pre-allocate contiguous output arrays
    D_flat = np.zeros(total_frames * ndim, dtype=np.float64)
    sim_indices = np.empty(total_frames, dtype=np.int32)

    offset = 0
    for i in range(nf):
        n_i = frame_counts[i]
        # Slice for this simulation's block in D_flat
        idx = np.arange(offset, offset + n_i)

        # sim_indices[offset:offset+n_i] = i
        sim_indices[offset:offset + n_i] = i

        # Column 0: E_self = energy_matrix[nf-1][i][:, 0]
        D_flat[idx * ndim] = energy_matrix[nf - 1][i][:n_i, 0]

        # Columns 1..NL: lambda values
        for b in range(NL):
            D_flat[idx * ndim + 1 + b] = lambda_arrays[i][:n_i, b]

        # Columns NL+1..NL+nf: cross energies
        for j in range(nf):
            D_flat[idx * ndim + NL + 1 + j] = energy_matrix[j][i][:n_i, 0]

        # Columns NL+nf+1, NL+nf+2 are already 0.0 (reserved bins)

        offset += n_i

    return D_flat, sim_indices, frame_counts, total_frames


def run_wham_from_memory(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run WHAM analysis from in-memory numpy arrays (no file I/O for input).

    Packs lambda trajectories and cross-simulation energies into the flat D_h
    layout expected by the CUDA kernel, then calls wham_from_memory().

    Output files (C.dat, V.dat, f.dat) are still written by CUDA to output_dir.

    Args:
        lambda_arrays: Per-simulation lambda trajectories. lambda_arrays[i] has
            shape (n_frames_i, nblocks).
        energy_matrix: Cross-simulation energies. energy_matrix[j][i] has shape
            (n_frames_i, 1).
        nblocks: Total number of lambda blocks.
        nf: Number of simulations.
        temp: Temperature in Kelvin.
        nts0: First terminal site index.
        nts1: Second terminal site index.
        use_gshift: If True, apply G_imp shifts.
        nsubs: Array of subsites per site.
        g_imp_path: Path to G_imp directory.
        gshift_data: G_imp shift array, or None for zero shifts.
        output_dir: Directory for C.dat/V.dat/f.dat output. Defaults to cwd.
        log_file: Optional log file for CUDA stdout/stderr.
        fnex: FNEX parameter for bias constants.
        cutlsum: G12 conditional threshold (must be in (0, 1)).

    Raises:
        ValueError: If cutlsum is out of range.
        RuntimeError: If CUDA wham_from_memory returns non-zero.
    """
    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(
            f"cutlsum must be in (0, 1), got {cutlsum}. "
            "Typical value is 0.8 (exclude frames where lambda_i + lambda_j < 0.8)."
        )

    # Pack data into flat layout
    D_flat, sim_indices, frame_counts, total_frames = _pack_wham_data(
        lambda_arrays, energy_matrix, nblocks, nf,
    )

    # Prepare gshift
    if gshift_data is not None:
        gshift_flat = np.ascontiguousarray(gshift_data.ravel(), dtype=np.float64)
    else:
        gshift_flat = np.zeros(1, dtype=np.float64)

    # Load library
    lib = _get_wham_lib()

    # Prepare ctypes arguments
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    # Prepare output directory
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "multisite").mkdir(exist_ok=True)

    # ctypes data pointers — keep arrays alive in local scope
    D_ptr = D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    logger.info(
        f"Running WHAM (in-memory) with nf={nf}, temp={temp}, "
        f"total_frames={total_frames}, fnex={fnex}"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = lib.wham_from_memory(
                nf, temp, nts0, nts1, int(use_gshift),
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.chi_offset, constants.omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
                D_ptr, si_ptr, fc_ptr,
                total_frames, gs_ptr,
            )
        if result != 0:
            raise RuntimeError(f"WHAM (in-memory) returned error code: {result}")

    logger.info("WHAM (in-memory) analysis completed successfully")


def run_wham_from_packed(
    D_flat: np.ndarray,
    sim_indices: np.ndarray,
    frame_counts: np.ndarray,
    total_frames: int,
    nf: int,
    nblocks: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run WHAM from pre-packed D_h data (no intermediate energy_matrix).

    Same as run_wham_from_memory but skips _pack_wham_data() — the caller
    (compute_packed_wham_data) has already built D_flat in CUDA layout.

    Args:
        D_flat: Pre-packed float64 array of shape (total_frames * ndim,).
        sim_indices: int32 array of shape (total_frames,).
        frame_counts: int32 array of shape (nf,).
        total_frames: Total number of frames.
        nf: Number of simulations.
        nblocks: Total number of lambda blocks.
        temp: Temperature in Kelvin.
        (other args: same as run_wham_from_memory)
    """
    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(
            f"cutlsum must be in (0, 1), got {cutlsum}. "
            "Typical value is 0.8 (exclude frames where lambda_i + lambda_j < 0.8)."
        )

    # Ensure contiguous arrays
    D_flat = np.ascontiguousarray(D_flat, dtype=np.float64)
    sim_indices = np.ascontiguousarray(sim_indices, dtype=np.int32)
    frame_counts = np.ascontiguousarray(frame_counts, dtype=np.int32)

    if gshift_data is not None:
        gshift_flat = np.ascontiguousarray(gshift_data.ravel(), dtype=np.float64)
    else:
        gshift_flat = np.zeros(1, dtype=np.float64)

    lib = _get_wham_lib()
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "multisite").mkdir(exist_ok=True)

    D_ptr = D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    logger.info(
        f"Running WHAM (packed) with nf={nf}, temp={temp}, "
        f"total_frames={total_frames}, fnex={fnex}"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = lib.wham_from_memory(
                nf, temp, nts0, nts1, int(use_gshift),
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.chi_offset, constants.omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
                D_ptr, si_ptr, fc_ptr,
                total_frames, gs_ptr,
            )
        if result != 0:
            raise RuntimeError(f"WHAM (packed) returned error code: {result}")

    logger.info("WHAM (packed) analysis completed successfully")


def compute_weights_from_packed(
    D_flat: np.ndarray,
    sim_indices: np.ndarray,
    frame_counts: np.ndarray,
    total_frames: int,
    nf: int,
    nblocks: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WHAM per-frame weights from pre-packed D_h data (GPU).

    Runs f-value convergence then computes w(t) = 1/Σ_j n_j·exp(f_j - β·E_j(t))
    for each frame. Returns weights and converged f-values.

    Args:
        D_flat: Pre-packed float64 array of shape (total_frames * ndim,).
        sim_indices: int32 array of shape (total_frames,).
        frame_counts: int32 array of shape (nf,).
        total_frames: Total number of frames.
        nf: Number of simulations.
        nblocks: Total number of lambda blocks.
        temp: Temperature in Kelvin.
        (other args: same as run_wham_from_packed)

    Returns:
        Tuple of (weights, f_values):
            weights: float64 array of shape (total_frames,) — per-frame WHAM weights.
            f_values: float64 array of shape (nf,) — converged free energies.
    """
    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(
            f"cutlsum must be in (0, 1), got {cutlsum}. "
            "Typical value is 0.8 (exclude frames where lambda_i + lambda_j < 0.8)."
        )

    D_flat = np.ascontiguousarray(D_flat, dtype=np.float64)
    sim_indices = np.ascontiguousarray(sim_indices, dtype=np.int32)
    frame_counts = np.ascontiguousarray(frame_counts, dtype=np.int32)

    if gshift_data is not None:
        gshift_flat = np.ascontiguousarray(gshift_data.ravel(), dtype=np.float64)
    else:
        gshift_flat = np.zeros(1, dtype=np.float64)

    lib = _get_wham_lib()
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)

    D_ptr = D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Pre-allocate output arrays
    weights_out = np.empty(total_frames, dtype=np.float64)
    f_out = np.empty(nf, dtype=np.float64)
    nf_out = ctypes.c_int(0)

    w_ptr = weights_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f_ptr = f_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    logger.info(
        f"Computing WHAM weights (packed) with nf={nf}, temp={temp}, "
        f"total_frames={total_frames}, fnex={fnex}"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = lib.wham_compute_weights_from_memory(
                nf, temp, nts0, nts1, int(use_gshift),
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.chi_offset, constants.omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
                D_ptr, si_ptr, fc_ptr,
                total_frames, gs_ptr,
                w_ptr, f_ptr, ctypes.byref(nf_out),
            )
        if result != 0:
            raise RuntimeError(f"WHAM weight computation returned error code: {result}")

    actual_nf = nf_out.value
    logger.info(f"WHAM weight computation completed (nf={actual_nf})")
    return weights_out, f_out[:actual_nf]


def compute_weights_numpy(
    D_flat: np.ndarray,
    frame_counts: np.ndarray,
    nf: int,
    nblocks: int,
    temp: float,
    f_values: np.ndarray,
) -> np.ndarray:
    """Compute WHAM weights from packed D_h data using numpy (CPU fallback).

    Pure numpy implementation for testing without GPU. Uses the log-sum-exp
    trick for numerical stability.

    Args:
        D_flat: Pre-packed float64 array of shape (total_frames * ndim,).
        frame_counts: int32 array of shape (nf,) — frames per simulation.
        nf: Number of simulations.
        nblocks: Total number of lambda blocks (NL).
        temp: Temperature in Kelvin.
        f_values: float64 array of shape (nf,) — converged f-values.

    Returns:
        float64 array of shape (total_frames,) — per-frame WHAM weights.
    """
    NL = nblocks
    ndim = NL + nf + 3
    kB = 0.001987204  # kcal/(mol·K)
    beta = 1.0 / (kB * temp)
    total_frames = int(frame_counts.sum())
    D = D_flat.reshape(total_frames, ndim)

    # Cross-energies: columns NL+1 .. NL+nf
    cross_E = D[:, NL + 1 : NL + 1 + nf]  # (total_frames, nf)
    n_j = frame_counts.astype(np.float64)  # (nf,)

    # log-space: log_denom[t,j] = log(n_j) + f_j - β·E_j(t)
    log_terms = np.log(n_j[np.newaxis, :]) + f_values[np.newaxis, :] - beta * cross_E
    # log-sum-exp for numerical stability
    max_log = np.max(log_terms, axis=1, keepdims=True)
    log_denom = max_log.ravel() + np.log(np.sum(np.exp(log_terms - max_log), axis=1))
    weights = np.exp(-log_denom)

    return weights


def run_wham_distributed_from_packed(
    D_flat: np.ndarray,
    sim_indices: np.ndarray,
    frame_counts: np.ndarray,
    total_frames: int,
    nf: int,
    nblocks: int,
    temp: float,
    comm,
    rank: int,
    nranks: int,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run distributed WHAM from pre-packed D_h data.

    Same workflow as run_wham_distributed but the D_h data is already packed
    on rank 0 (from compute_packed_wham_data_distributed). Skips the
    _prepare_wham_common_args/_pack_wham_data call.

    Falls back to run_wham_from_packed for nranks <= 1.
    """
    from mpi4py import MPI

    if nranks <= 1:
        if rank == 0:
            run_wham_from_packed(
                D_flat=D_flat,
                sim_indices=sim_indices,
                frame_counts=frame_counts,
                total_frames=total_frames,
                nf=nf,
                nblocks=nblocks,
                temp=temp,
                nts0=nts0,
                nts1=nts1,
                use_gshift=use_gshift,
                nsubs=nsubs,
                g_imp_path=g_imp_path,
                gshift_data=gshift_data,
                output_dir=output_dir,
                log_file=log_file,
                fnex=fnex,
                cutlsum=cutlsum,
                chi_offset=chi_offset,
                omega_decay=omega_decay,
                chi_offset_t=chi_offset_t,
                chi_offset_u=chi_offset_u,
                ntriangle=ntriangle,
            )
        return

    out_path = Path(output_dir) if output_dir is not None else Path.cwd()
    log_path = Path(log_file).resolve() if log_file is not None else None

    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(f"cutlsum must be in (0, 1), got {cutlsum}.")

    # Prepare gshift
    if gshift_data is not None:
        gshift_flat = np.ascontiguousarray(gshift_data.ravel(), dtype=np.float64)
    else:
        gshift_flat = np.zeros(1, dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase A: rank 0 runs f-value convergence on pre-packed data
    # ------------------------------------------------------------------
    iterate_ok = True
    f_values = None
    n_profiles = 0

    if rank == 0:
        try:
            D_flat = np.ascontiguousarray(D_flat, dtype=np.float64)
            sim_indices = np.ascontiguousarray(sim_indices, dtype=np.int32)
            frame_counts = np.ascontiguousarray(frame_counts, dtype=np.int32)

            lib = _get_wham_lib()
            nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
            g_imp_path_bytes = _to_bytes(g_imp_path)
            constants = derive_bias_constants(
                fnex, chi_offset=chi_offset, omega_decay=omega_decay
            )

            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "multisite").mkdir(exist_ok=True)

            f_out = np.zeros(nf, dtype=np.float64)
            nf_out = ctypes.c_int(0)

            D_ptr = D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            f_ptr = f_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            with _chdir_context(out_path):
                with _redirect_c_output(log_path):
                    result = lib.wham_iterate_from_memory(
                        nf, temp, nts0, nts1, int(use_gshift),
                        nsubs_ptr, nsites, g_imp_path_bytes,
                        constants.chi_offset, constants.omega_scale, cutlsum,
                        chi_offset_t, chi_offset_u, ntriangle,
                        D_ptr, si_ptr, fc_ptr,
                        total_frames, gs_ptr,
                        f_ptr, ctypes.byref(nf_out),
                    )
                if result != 0:
                    raise RuntimeError(f"WHAM iterate returned error code: {result}")

            f_values = f_out[: nf_out.value]
            n_profiles = _compute_n_profiles(nsubs, nts1)
            logger.info(
                f"Phase A done: {nf_out.value} f-values, {n_profiles} profiles"
            )
        except Exception as e:
            iterate_ok = False
            logger.error(f"Phase A (packed) failed: {e}")

    # Broadcast iterate status
    iterate_ok = comm.bcast(iterate_ok, root=0)
    if not iterate_ok:
        raise RuntimeError("WHAM Phase A (f-convergence) failed on rank 0")

    # ------------------------------------------------------------------
    # Build slim D + pre-compute lnDenom on rank 0, then broadcast
    # ------------------------------------------------------------------
    # Slim D layout: [E_self | lambda[0..NL-1] | bin1D | bin2D]
    #   ndim = NL + 3  (vs full ndim = NL + nf + 3)
    # lnDenom: one scalar per frame (WHAM reweighting denominator)
    nsubs_arr_local = np.asarray(nsubs, dtype=np.int32)
    NL = int(nsubs_arr_local.sum())
    slim_ndim = NL + 3

    if rank == 0:
        full_ndim = NL + nf + 3
        D_2d = D_flat.reshape(total_frames, full_ndim)

        # Compute lnDenom = logsumexp_j(log(n_j) + f_j - beta * E_cross[t,j])
        from scipy.special import logsumexp as _logsumexp

        kB = 0.001987204  # kcal/mol/K, matches CUDA
        beta = 1.0 / (kB * temp)
        E_cross = D_2d[:, 1 + NL : 1 + NL + nf]  # (total_frames, nf)
        log_n = np.log(frame_counts.astype(np.float64))  # (nf,)
        terms = log_n[np.newaxis, :] + f_values[np.newaxis, :] - beta * E_cross
        lnDenom = _logsumexp(terms, axis=1).astype(np.float64)  # (total_frames,)

        # Build slim D (strip cross-energy columns)
        slim_D = np.zeros((total_frames, slim_ndim), dtype=np.float64)
        slim_D[:, 0] = D_2d[:, 0]  # E_self
        slim_D[:, 1 : 1 + NL] = D_2d[:, 1 : 1 + NL]  # lambda columns
        # bin1D/bin2D = zeros (CUDA writes them)
        slim_D_flat = np.ascontiguousarray(slim_D.ravel())

        # Free full D — the big memory win
        del D_flat, D_2d, E_cross, terms, slim_D

        sizes = (
            slim_D_flat.shape[0], sim_indices.shape[0],
            frame_counts.shape[0], gshift_flat.shape[0], total_frames,
        )
        logger.info(
            f"Slim D: {slim_D_flat.nbytes / 1e6:.0f} MB "
            f"(was {total_frames * full_ndim * 8 / 1e6:.0f} MB)"
        )
    else:
        slim_D_flat = None
        lnDenom = None
        sizes = None

    sizes = comm.bcast(sizes, root=0)
    sd_sz, si_sz, fc_sz, gs_sz, total_frames = sizes

    if rank != 0:
        slim_D_flat = np.empty(sd_sz, dtype=np.float64)
        lnDenom = np.empty(total_frames, dtype=np.float64)
        sim_indices = np.empty(si_sz, dtype=np.int32)
        frame_counts = np.empty(fc_sz, dtype=np.int32)
        gshift_flat = np.empty(gs_sz, dtype=np.float64)

    comm.Bcast(slim_D_flat, root=0)
    comm.Bcast(lnDenom, root=0)
    comm.Bcast(sim_indices, root=0)
    comm.Bcast(frame_counts, root=0)
    comm.Bcast(gshift_flat, root=0)

    f_values = comm.bcast(f_values, root=0)
    n_profiles = comm.bcast(n_profiles, root=0)

    # ------------------------------------------------------------------
    # Phase B: each rank computes its profile subset (slim D path)
    # ------------------------------------------------------------------
    my_start = rank * n_profiles // nranks
    my_end = (rank + 1) * n_profiles // nranks

    logger.info(
        f"Phase B (slim): rank {rank} profiles [{my_start}, {my_end}) "
        f"of {n_profiles}, ndim={slim_ndim}"
    )

    profile_ok = True
    dim_out = ctypes.c_int(0)
    max_dim = 1000
    C_out = np.zeros(max_dim * max_dim, dtype=np.float64)
    V_out = np.zeros(max_dim, dtype=np.float64)

    if my_start < my_end:
        lib = _get_wham_lib()
        nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
        g_imp_path_bytes = _to_bytes(g_imp_path)
        constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)
        f_in = np.ascontiguousarray(f_values, dtype=np.float64)
        lnDenom_c = np.ascontiguousarray(lnDenom, dtype=np.float64)

        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "multisite").mkdir(exist_ok=True)

        D_ptr = slim_D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        f_ptr = f_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ld_ptr = lnDenom_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        C_ptr = C_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        V_ptr = V_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        try:
            with _chdir_context(out_path):
                with _redirect_c_output(log_path):
                    result = lib.wham_profiles_slim_from_memory(
                        nf, temp, nts0, nts1, int(use_gshift),
                        nsubs_ptr, nsites, g_imp_path_bytes,
                        constants.chi_offset, constants.omega_scale, cutlsum,
                        chi_offset_t, chi_offset_u, ntriangle,
                        D_ptr, slim_ndim,
                        si_ptr, fc_ptr,
                        total_frames, gs_ptr,
                        f_ptr, len(f_in),
                        ld_ptr,
                        my_start, my_end,
                        C_ptr, V_ptr, ctypes.byref(dim_out),
                    )
                if result != 0:
                    raise RuntimeError(f"WHAM profiles returned error code: {result}")
        except Exception as e:
            profile_ok = False
            logger.error(f"Phase B (slim) failed on rank {rank}: {e}")


    all_ok = comm.allreduce(int(profile_ok), op=MPI.MIN)
    if not all_ok:
        raise RuntimeError(
            "WHAM Phase B (profile computation) failed on one or more ranks"
        )

    # Ranks that skipped CUDA have dim_out=0; get the real dim via allreduce MAX.
    dim = comm.allreduce(dim_out.value, op=MPI.MAX)
    if dim > max_dim:
        raise RuntimeError(f"C/V dimension {dim} exceeds max_dim={max_dim}")

    C_local = C_out[: dim * dim].copy()
    V_local = V_out[:dim].copy()

    if rank == 0:
        C_global = np.zeros_like(C_local)
        V_global = np.zeros_like(V_local)
    else:
        C_global = None
        V_global = None

    comm.Reduce(C_local, C_global, op=MPI.SUM, root=0)
    comm.Reduce(V_local, V_global, op=MPI.SUM, root=0)

    if rank == 0:
        _finalize_cv(C_global, V_global, dim, out_path)
        logger.info(f"WHAM distributed (packed) completed: C/V dim={dim}")


def _prepare_wham_common_args(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
    temp: float,
    nts0: int,
    nts1: int,
    use_gshift: bool,
    nsubs: np.ndarray | list[int] | None,
    g_imp_path: str | Path | None,
    gshift_data: np.ndarray | None,
    fnex: float,
    cutlsum: float,
    chi_offset: float | None,
    omega_decay: float | None,
    chi_offset_t: float,
    chi_offset_u: float,
    ntriangle: int,
) -> dict:
    """Prepare common arguments for distributed WHAM calls.

    Packs data, resolves constants, prepares ctypes pointers.
    Returns a dict of all prepared arguments.
    """
    if cutlsum <= 0.0 or cutlsum >= 1.0:
        raise ValueError(f"cutlsum must be in (0, 1), got {cutlsum}.")

    D_flat, sim_indices, frame_counts, total_frames = _pack_wham_data(
        lambda_arrays, energy_matrix, nblocks, nf,
    )

    if gshift_data is not None:
        gshift_flat = np.ascontiguousarray(gshift_data.ravel(), dtype=np.float64)
    else:
        gshift_flat = np.zeros(1, dtype=np.float64)

    lib = _get_wham_lib()
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    return {
        "lib": lib,
        "D_flat": D_flat,
        "sim_indices": sim_indices,
        "frame_counts": frame_counts,
        "total_frames": total_frames,
        "gshift_flat": gshift_flat,
        "nsubs_arr": nsubs_arr,
        "nsubs_ptr": nsubs_ptr,
        "nsites": nsites,
        "g_imp_path_bytes": g_imp_path_bytes,
        "constants": constants,
        "nf": nf,
        "temp": temp,
        "nts0": nts0,
        "nts1": nts1,
        "use_gshift": use_gshift,
        "chi_offset_t": chi_offset_t,
        "chi_offset_u": chi_offset_u,
        "ntriangle": ntriangle,
        "cutlsum": cutlsum,
    }


def run_wham_iterate(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> np.ndarray:
    """Run WHAM Phase A: f-value convergence only.

    Returns the converged f-values array for broadcasting to other ranks.
    Does NOT compute profiles (C.dat/V.dat).

    Returns:
        f_values: float64 array of shape (NF,) with converged f-values.
    """
    args = _prepare_wham_common_args(
        lambda_arrays, energy_matrix, nblocks, nf, temp, nts0, nts1,
        use_gshift, nsubs, g_imp_path, gshift_data,
        fnex, cutlsum, chi_offset, omega_decay, chi_offset_t, chi_offset_u, ntriangle,
    )

    # Prepare output directory (for f.dat written by iteratedata)
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "multisite").mkdir(exist_ok=True)

    # Output buffers for f-values
    f_out = np.zeros(nf, dtype=np.float64)
    nf_out = ctypes.c_int(0)

    D_ptr = args["D_flat"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    si_ptr = args["sim_indices"].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fc_ptr = args["frame_counts"].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gs_ptr = args["gshift_flat"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f_ptr = f_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    log_path = Path(log_file).resolve() if log_file is not None else None

    logger.info(f"Running WHAM iterate (Phase A) with nf={nf}, temp={temp}")

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = args["lib"].wham_iterate_from_memory(
                nf, temp, nts0, nts1, int(use_gshift),
                args["nsubs_ptr"], args["nsites"], args["g_imp_path_bytes"],
                args["constants"].chi_offset, args["constants"].omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
                D_ptr, si_ptr, fc_ptr,
                args["total_frames"], gs_ptr,
                f_ptr, ctypes.byref(nf_out),
            )
        if result != 0:
            raise RuntimeError(f"WHAM iterate returned error code: {result}")

    actual_nf = nf_out.value
    logger.info(f"WHAM iterate completed, {actual_nf} f-values converged")
    return f_out[:actual_nf]


def run_wham_profiles(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
    temp: float,
    f_values: np.ndarray,
    profile_start: int,
    profile_end: int,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run WHAM Phase B: compute profiles for a subset.

    Uses pre-converged f-values from run_wham_iterate().
    Returns raw (unregularized) C/V contributions for MPI_Reduce.

    Args:
        f_values: Converged f-values from Phase A.
        profile_start: First profile index (inclusive).
        profile_end: Last profile index (exclusive).

    Returns:
        C_partial: float64 array of shape (dim*dim,) — raw Hessian contributions.
        V_partial: float64 array of shape (dim,) — raw gradient contributions.
        dim: Dimension of the C/V system (jN + iN).
    """
    args = _prepare_wham_common_args(
        lambda_arrays, energy_matrix, nblocks, nf, temp, nts0, nts1,
        use_gshift, nsubs, g_imp_path, gshift_data,
        fnex, cutlsum, chi_offset, omega_decay, chi_offset_t, chi_offset_u, ntriangle,
    )

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "multisite").mkdir(exist_ok=True)

    # We don't know the exact dim until CUDA computes it, but we can estimate:
    # dim = jN + iN, which depends on nsubs, ntriangle, ms, msprof.
    # Allocate generously — CUDA will tell us the actual dim.
    # Upper bound: for nsubs with N subsites each, jN ~ N + ntriangle*N*(N-1)/2 per site
    # and iN ~ N + 2*N*(N-1)/2 per site. 1000 is a safe upper bound for any system.
    max_dim = 1000
    C_out = np.zeros(max_dim * max_dim, dtype=np.float64)
    V_out = np.zeros(max_dim, dtype=np.float64)
    dim_out = ctypes.c_int(0)

    f_in = np.ascontiguousarray(f_values, dtype=np.float64)

    D_ptr = args["D_flat"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    si_ptr = args["sim_indices"].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fc_ptr = args["frame_counts"].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gs_ptr = args["gshift_flat"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f_ptr = f_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    C_ptr = C_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    V_ptr = V_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    log_path = Path(log_file).resolve() if log_file is not None else None

    logger.info(
        f"Running WHAM profiles (Phase B) for profiles [{profile_start}, {profile_end})"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = args["lib"].wham_profiles_from_memory(
                nf, temp, nts0, nts1, int(use_gshift),
                args["nsubs_ptr"], args["nsites"], args["g_imp_path_bytes"],
                args["constants"].chi_offset, args["constants"].omega_scale, cutlsum,
                chi_offset_t, chi_offset_u, ntriangle,
                D_ptr, si_ptr, fc_ptr,
                args["total_frames"], gs_ptr,
                f_ptr, len(f_in),
                profile_start, profile_end,
                C_ptr, V_ptr, ctypes.byref(dim_out),
            )
        if result != 0:
            raise RuntimeError(f"WHAM profiles returned error code: {result}")

    dim = dim_out.value
    logger.info(f"WHAM profiles completed, dim={dim}")

    # Reshape to actual dimensions
    C_partial = C_out[:dim * dim].copy()
    V_partial = V_out[:dim].copy()
    return C_partial, V_partial, dim


def _finalize_cv(C: np.ndarray, V: np.ndarray, dim: int, output_dir: Path) -> None:
    """Apply regularization and write C.dat/V.dat after MPI_Reduce.

    Mirrors the finalization logic in getfofq() (wham.cu):
    - Add regularization to empty rows and diagonal
    - Write multisite/C.dat and multisite/V.dat
    """
    C_mat = C.reshape(dim, dim)
    lambda_reg = 1e-9
    big_lambda = 1.0

    for j1 in range(dim):
        row_weight = np.sum(np.abs(C_mat[j1, :]))
        if row_weight < 1e-12:
            C_mat[j1, j1] += big_lambda
            V[j1] = 0.0
        C_mat[j1, j1] += lambda_reg

    multisite_dir = output_dir / "multisite"
    multisite_dir.mkdir(exist_ok=True)

    with open(multisite_dir / "C.dat", "w") as f:
        for j1 in range(dim):
            f.write(" ".join(f"{C_mat[j1, j2]:.12g}" for j2 in range(dim)))
            f.write("\n")

    with open(multisite_dir / "V.dat", "w") as f:
        for j1 in range(dim):
            f.write(f" {V[j1]:.12g}\n")


def _compute_n_profiles(nsubs, nts1: int) -> int:
    """Compute total number of WHAM profiles (iN) from nsubs.

    Mirrors the profile counting logic in wham.cu readdata_from_memory.
    """
    nsubs_arr = np.asarray(nsubs, dtype=np.int32) if nsubs is not None else np.array([], dtype=np.int32)
    iN = 0
    nsites_val = len(nsubs_arr)
    for s1 in range(nsites_val):
        for s2 in range(s1, nsites_val):
            if s1 == s2:
                ns = int(nsubs_arr[s1])
                if ns == 2:
                    iN += ns + ns * (ns - 1) // 2
                else:
                    iN += ns + 2 * ns * (ns - 1) // 2
            elif nts1 > 0:  # msprof
                iN += int(nsubs_arr[s1]) * int(nsubs_arr[s2])
    return iN


def run_wham_distributed(
    lambda_arrays: list[np.ndarray],
    energy_matrix: list[list[np.ndarray]],
    nblocks: int,
    nf: int,
    temp: float,
    comm,
    rank: int,
    nranks: int,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    gshift_data: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    cutlsum: float = 0.8,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run WHAM with profile computation distributed across MPI ranks.

    Flow:
    1. Rank 0: pack data + f-value convergence (iteratedata on 1 GPU)
    2. Broadcast packed data arrays + f-values to all ranks
    3. All ranks: compute assigned profile subset on own GPU
    4. MPI_Reduce(SUM) partial C/V on rank 0
    5. Rank 0: regularize + write C.dat/V.dat

    Non-rank-0 ranks need not have lambda_arrays/energy_matrix — the packed
    data is broadcast from rank 0. Falls back to run_wham_from_memory() when
    nranks=1.

    Args:
        comm: MPI communicator (e.g., MPI.COMM_WORLD).
        rank: This rank's index.
        nranks: Total number of ranks.
        (other args: same as run_wham_from_memory)
    """
    from mpi4py import MPI

    # Single-rank fallback: use the standard non-distributed path
    if nranks <= 1:
        if rank == 0:
            run_wham_from_memory(
                lambda_arrays=lambda_arrays,
                energy_matrix=energy_matrix,
                nblocks=nblocks,
                nf=nf,
                temp=temp,
                nts0=nts0,
                nts1=nts1,
                use_gshift=use_gshift,
                nsubs=nsubs,
                g_imp_path=g_imp_path,
                gshift_data=gshift_data,
                output_dir=output_dir,
                log_file=log_file,
                fnex=fnex,
                cutlsum=cutlsum,
                chi_offset=chi_offset,
                omega_decay=omega_decay,
                chi_offset_t=chi_offset_t,
                chi_offset_u=chi_offset_u,
                ntriangle=ntriangle,
            )
        return

    out_path = Path(output_dir) if output_dir is not None else Path.cwd()
    log_path = Path(log_file).resolve() if log_file is not None else None

    # ------------------------------------------------------------------
    # Phase A: rank 0 packs data + runs f-value convergence
    # ------------------------------------------------------------------
    iterate_ok = True
    f_values = None
    n_profiles = 0
    # Packed arrays (only meaningful on rank 0 initially)
    D_flat = sim_indices = frame_counts = gshift_flat = None
    total_frames = 0

    if rank == 0:
        try:
            packed = _prepare_wham_common_args(
                lambda_arrays, energy_matrix, nblocks, nf, temp, nts0, nts1,
                use_gshift, nsubs, g_imp_path, gshift_data,
                fnex, cutlsum, chi_offset, omega_decay,
                chi_offset_t, chi_offset_u, ntriangle,
            )

            D_flat = packed["D_flat"]
            sim_indices = packed["sim_indices"]
            frame_counts = packed["frame_counts"]
            total_frames = packed["total_frames"]
            gshift_flat = packed["gshift_flat"]

            # Prepare output directory
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "multisite").mkdir(exist_ok=True)

            # Run iterate from packed data
            f_out = np.zeros(nf, dtype=np.float64)
            nf_out = ctypes.c_int(0)

            D_ptr = D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            f_ptr = f_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            with _chdir_context(out_path):
                with _redirect_c_output(log_path):
                    result = packed["lib"].wham_iterate_from_memory(
                        nf, temp, nts0, nts1, int(use_gshift),
                        packed["nsubs_ptr"], packed["nsites"],
                        packed["g_imp_path_bytes"],
                        packed["constants"].chi_offset,
                        packed["constants"].omega_scale, cutlsum,
                        chi_offset_t, chi_offset_u, ntriangle,
                        D_ptr, si_ptr, fc_ptr,
                        total_frames, gs_ptr,
                        f_ptr, ctypes.byref(nf_out),
                    )
                if result != 0:
                    raise RuntimeError(f"WHAM iterate returned error code: {result}")

            f_values = f_out[: nf_out.value]
            n_profiles = _compute_n_profiles(nsubs, nts1)
            logger.info(
                f"Phase A done: {nf_out.value} f-values, {n_profiles} profiles"
            )
        except Exception as e:
            iterate_ok = False
            logger.error(f"Phase A failed: {e}")

    # Broadcast iterate status — MUST happen so non-rank-0 doesn't deadlock
    iterate_ok = comm.bcast(iterate_ok, root=0)
    if not iterate_ok:
        raise RuntimeError("WHAM Phase A (f-convergence) failed on rank 0")

    # ------------------------------------------------------------------
    # Build slim D + pre-compute lnDenom on rank 0, then broadcast
    # ------------------------------------------------------------------
    # Slim D layout: [E_self | lambda[0..NL-1] | bin1D | bin2D]
    #   ndim = NL + 3  (vs full ndim = NL + nf + 3)
    # lnDenom: one scalar per frame (WHAM reweighting denominator)
    nsubs_arr_local = np.asarray(nsubs, dtype=np.int32)
    NL = int(nsubs_arr_local.sum())
    slim_ndim = NL + 3

    if rank == 0:
        full_ndim = NL + nf + 3
        D_2d = D_flat.reshape(total_frames, full_ndim)

        # Compute lnDenom = logsumexp_j(log(n_j) + f_j - beta * E_cross[t,j])
        from scipy.special import logsumexp as _logsumexp

        kB = 0.001987204  # kcal/mol/K, matches CUDA
        beta = 1.0 / (kB * temp)
        E_cross = D_2d[:, 1 + NL : 1 + NL + nf]  # (total_frames, nf)
        log_n = np.log(frame_counts.astype(np.float64))  # (nf,)
        terms = log_n[np.newaxis, :] + f_values[np.newaxis, :] - beta * E_cross
        lnDenom = _logsumexp(terms, axis=1).astype(np.float64)  # (total_frames,)

        # Build slim D (strip cross-energy columns)
        slim_D = np.zeros((total_frames, slim_ndim), dtype=np.float64)
        slim_D[:, 0] = D_2d[:, 0]  # E_self
        slim_D[:, 1 : 1 + NL] = D_2d[:, 1 : 1 + NL]  # lambda columns
        # bin1D/bin2D = zeros (CUDA writes them)
        slim_D_flat = np.ascontiguousarray(slim_D.ravel())

        # Free full D — the big memory win
        del D_flat, D_2d, E_cross, terms, slim_D

        sizes = (
            slim_D_flat.shape[0], sim_indices.shape[0],
            frame_counts.shape[0], gshift_flat.shape[0], total_frames,
        )
        logger.info(
            f"Slim D: {slim_D_flat.nbytes / 1e6:.0f} MB "
            f"(was {total_frames * full_ndim * 8 / 1e6:.0f} MB)"
        )
    else:
        slim_D_flat = None
        lnDenom = None
        sizes = None

    sizes = comm.bcast(sizes, root=0)
    sd_sz, si_sz, fc_sz, gs_sz, total_frames = sizes

    if rank != 0:
        slim_D_flat = np.empty(sd_sz, dtype=np.float64)
        lnDenom = np.empty(total_frames, dtype=np.float64)
        sim_indices = np.empty(si_sz, dtype=np.int32)
        frame_counts = np.empty(fc_sz, dtype=np.int32)
        gshift_flat = np.empty(gs_sz, dtype=np.float64)

    comm.Bcast(slim_D_flat, root=0)
    comm.Bcast(lnDenom, root=0)
    comm.Bcast(sim_indices, root=0)
    comm.Bcast(frame_counts, root=0)
    comm.Bcast(gshift_flat, root=0)

    f_values = comm.bcast(f_values, root=0)
    n_profiles = comm.bcast(n_profiles, root=0)

    # ------------------------------------------------------------------
    # Phase B: each rank computes its profile subset (slim D path)
    # ------------------------------------------------------------------
    my_start = rank * n_profiles // nranks
    my_end = (rank + 1) * n_profiles // nranks

    logger.info(
        f"Phase B (slim): rank {rank} profiles [{my_start}, {my_end}) "
        f"of {n_profiles}, ndim={slim_ndim}"
    )

    profile_ok = True
    dim_out = ctypes.c_int(0)
    max_dim = 1000
    C_out = np.zeros(max_dim * max_dim, dtype=np.float64)
    V_out = np.zeros(max_dim, dtype=np.float64)

    if my_start < my_end:
        lib = _get_wham_lib()
        nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
        g_imp_path_bytes = _to_bytes(g_imp_path)
        constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)
        f_in = np.ascontiguousarray(f_values, dtype=np.float64)
        lnDenom_c = np.ascontiguousarray(lnDenom, dtype=np.float64)

        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "multisite").mkdir(exist_ok=True)

        D_ptr = slim_D_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        si_ptr = sim_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        fc_ptr = frame_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        gs_ptr = gshift_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        f_ptr = f_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ld_ptr = lnDenom_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        C_ptr = C_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        V_ptr = V_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        try:
            with _chdir_context(out_path):
                with _redirect_c_output(log_path):
                    result = lib.wham_profiles_slim_from_memory(
                        nf, temp, nts0, nts1, int(use_gshift),
                        nsubs_ptr, nsites, g_imp_path_bytes,
                        constants.chi_offset, constants.omega_scale, cutlsum,
                        chi_offset_t, chi_offset_u, ntriangle,
                        D_ptr, slim_ndim,
                        si_ptr, fc_ptr,
                        total_frames, gs_ptr,
                        f_ptr, len(f_in),
                        ld_ptr,
                        my_start, my_end,
                        C_ptr, V_ptr, ctypes.byref(dim_out),
                    )
                if result != 0:
                    raise RuntimeError(f"WHAM profiles returned error code: {result}")
        except Exception as e:
            profile_ok = False
            logger.error(f"Phase B (slim) failed on rank {rank}: {e}")

    # Sync profile status — ensures all ranks reach Reduce together
    all_ok = comm.allreduce(int(profile_ok), op=MPI.MIN)
    if not all_ok:
        raise RuntimeError("WHAM Phase B (profile computation) failed on one or more ranks")

    # Ranks that skipped CUDA have dim_out=0; get the real dim via allreduce MAX.
    dim = comm.allreduce(dim_out.value, op=MPI.MAX)
    if dim > max_dim:
        raise RuntimeError(
            f"WHAM dimension {dim} exceeds buffer size {max_dim} — "
            f"increase max_dim in run_wham_distributed"
        )
    C_partial = C_out[: dim * dim].copy()
    V_partial = V_out[:dim].copy()

    # ------------------------------------------------------------------
    # MPI_Reduce: sum partial C/V on rank 0
    # ------------------------------------------------------------------
    # Use empty arrays (not None) on non-root for portability across mpi4py versions
    C_total = np.zeros_like(C_partial) if rank == 0 else np.empty_like(C_partial)
    V_total = np.zeros_like(V_partial) if rank == 0 else np.empty_like(V_partial)

    comm.Reduce(C_partial, C_total, op=MPI.SUM, root=0)
    comm.Reduce(V_partial, V_total, op=MPI.SUM, root=0)

    # --- Rank 0: finalize and write output ---
    if rank == 0:
        _finalize_cv(C_total, V_total, dim, out_path)
        logger.info("Distributed WHAM completed successfully")


def run_lmalf_from_memory(
    lambda_combined: np.ndarray,
    ensweight: np.ndarray | None,
    nf: int,
    temp: float,
    ms: int = 0,
    msprof: int = 0,
    max_iter: int = 0,
    tolerance: float = 0.0,
    nsubs: np.ndarray | list[int] | None = None,
    g_imp_path: str | Path | None = None,
    x_prev: np.ndarray | None = None,
    s_prev: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run LMALF analysis from in-memory numpy arrays (no file I/O for input).

    Passes lambda trajectories and optional ensemble weights directly to the
    CUDA lmalf_from_memory() kernel.

    Output file (OUT.dat) is still written by CUDA to output_dir.

    Args:
        lambda_combined: Combined lambda trajectory, shape (n_frames, nblocks).
        ensweight: Ensemble weights, shape (n_frames,), or None for uniform.
        nf: Number of simulations.
        temp: Temperature in Kelvin.
        ms: Multisite coupling flag.
        msprof: Multisite profiles flag.
        max_iter: Maximum L-BFGS iterations (0 = default).
        tolerance: Convergence tolerance (0 = default).
        nsubs: Array of subsites per site.
        g_imp_path: Path to G_imp directory.
        x_prev: Previous x parameters, shape (nblocks, nblocks), or None.
        s_prev: Previous s parameters, shape (nblocks, nblocks), or None.
        output_dir: Directory for OUT.dat output. Defaults to cwd.
        log_file: Optional log file for CUDA stdout/stderr.
        fnex: FNEX parameter for bias constants.
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Raises:
        RuntimeError: If CUDA lmalf_from_memory returns non-zero.
    """
    # Ensure contiguous float64
    lambda_flat = np.ascontiguousarray(lambda_combined, dtype=np.float64)
    n_frames = lambda_flat.shape[0]

    # Ensemble weights: contiguous or NULL
    if ensweight is not None:
        ens_flat = np.ascontiguousarray(ensweight.ravel(), dtype=np.float64)
        ens_ptr = ens_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        ens_flat = None  # noqa: F841 — keep name for clarity
        ens_ptr = ctypes.POINTER(ctypes.c_double)()  # NULL

    # Previous coupling parameters
    if x_prev is not None:
        x_flat = np.ascontiguousarray(x_prev.ravel(), dtype=np.float64)
        x_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nblocks_sq = x_flat.size
    else:
        x_flat = None  # noqa: F841
        x_ptr = ctypes.POINTER(ctypes.c_double)()  # NULL
        nblocks_sq = 0

    if s_prev is not None:
        s_flat = np.ascontiguousarray(s_prev.ravel(), dtype=np.float64)
        s_ptr = s_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        s_flat = None  # noqa: F841
        s_ptr = ctypes.POINTER(ctypes.c_double)()  # NULL

    # Load library
    lib = _get_wham_lib()

    # Prepare ctypes arguments
    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    g_imp_path_bytes = _to_bytes(g_imp_path)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    # Prepare output directory
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)

    # Lambda data pointer
    lam_ptr = lambda_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    logger.info(
        f"Running LMALF (in-memory) with nf={nf}, temp={temp}, "
        f"n_frames={n_frames}, ms={ms}, msprof={msprof}, fnex={fnex}"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = lib.lmalf_from_memory(
                nf, temp, ms, msprof, max_iter, tolerance,
                nsubs_ptr, nsites, g_imp_path_bytes,
                constants.fnex, constants.chi_offset, constants.omega_scale,
                chi_offset_t, chi_offset_u, ntriangle,
                lam_ptr, ens_ptr, n_frames,
                x_ptr, s_ptr, nblocks_sq,
            )
        if result != 0:
            raise RuntimeError(f"LMALF (in-memory) returned error code: {result}")

    logger.info("LMALF (in-memory) analysis completed successfully")


# ═══════════════════════════════════════════════════════════════════════════
# nonlinear solver (now merged into libwham.so)
# ═══════════════════════════════════════════════════════════════════════════


def run_nonlinear_from_memory(
    lambda_combined: np.ndarray,
    ensweight: np.ndarray | None,
    nf: int,
    temp: float,
    ms: int = 0,
    msprof: int = 0,
    max_iter: int = 0,
    tolerance: float = 0.0,
    nsubs: np.ndarray | list[int] | None = None,
    x_prev: np.ndarray | None = None,
    s_prev: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    fnex: float = 5.5,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
    chi_offset_t: float = 0.012,
    chi_offset_u: float = 0.012,
    ntriangle: int = 5,
) -> None:
    """Run nonlinear L-BFGS analysis from in-memory numpy arrays.

    Same interface as run_lmalf_from_memory but uses the nonlinear CUDA solver.
    Output file (OUT.dat) is written by CUDA to output_dir.

    Args:
        lambda_combined: Combined lambda trajectory, shape (n_frames, nblocks).
        ensweight: Ensemble weights, shape (n_frames,), or None for uniform.
        nf: Number of simulations.
        temp: Temperature in Kelvin.
        ms: Multisite coupling flag.
        msprof: Multisite profiles flag.
        max_iter: Maximum L-BFGS iterations (0 = default 250).
        tolerance: Convergence tolerance (0 = default 1.25e-3).
        nsubs: Array of subsites per site.
        x_prev: Previous x parameters, shape (nblocks, nblocks), or None.
        s_prev: Previous s parameters, shape (nblocks, nblocks), or None.
        output_dir: Directory for OUT.dat output. Defaults to cwd.
        log_file: Optional log file for CUDA stdout/stderr.
        fnex: FNEX parameter for bias constants.
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).
        chi_offset_t: t-term sigmoid offset.
        chi_offset_u: u-term Hill sigmoid offset.
        ntriangle: Pair params per unique pair (1, 3, 5, 7, or 9).

    Raises:
        RuntimeError: If CUDA nonlinear_from_memory returns non-zero.
    """
    lambda_flat = np.ascontiguousarray(lambda_combined, dtype=np.float64)
    n_frames = lambda_flat.shape[0]

    if ensweight is not None:
        ens_flat = np.ascontiguousarray(ensweight.ravel(), dtype=np.float64)
    else:
        ens_flat = np.ones(n_frames, dtype=np.float64)
    ens_ptr = ens_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if x_prev is not None:
        x_flat = np.ascontiguousarray(x_prev.ravel(), dtype=np.float64)
        x_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nblocks_sq = x_flat.size
    else:
        x_flat = None  # noqa: F841
        x_ptr = ctypes.POINTER(ctypes.c_double)()
        nblocks_sq = 0

    if s_prev is not None:
        s_flat = np.ascontiguousarray(s_prev.ravel(), dtype=np.float64)
        s_ptr = s_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        s_flat = None  # noqa: F841
        s_ptr = ctypes.POINTER(ctypes.c_double)()

    lib = _get_wham_lib()

    nsubs_arr, nsubs_ptr, nsites = _prepare_nsubs(nsubs)
    log_path = Path(log_file).resolve() if log_file is not None else None

    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd()
    out_path.mkdir(parents=True, exist_ok=True)

    lam_ptr = lambda_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    logger.info(
        f"Running nonlinear (in-memory) with nf={nf}, temp={temp}, "
        f"n_frames={n_frames}, ms={ms}, msprof={msprof}, fnex={fnex}"
    )

    with _chdir_context(out_path):
        with _redirect_c_output(log_path):
            result = lib.nonlinear_from_memory(
                nf, temp, ms, msprof, max_iter, tolerance,
                nsubs_ptr, nsites,
                constants.fnex, constants.chi_offset, constants.omega_scale,
                chi_offset_t, chi_offset_u, ntriangle,
                lam_ptr, ens_ptr, n_frames,
                x_ptr, s_ptr, nblocks_sq,
            )
        if result != 0:
            raise RuntimeError(f"nonlinear returned error code: {result}")

    logger.info("nonlinear analysis completed successfully")
