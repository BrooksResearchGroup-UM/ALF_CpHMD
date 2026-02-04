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

Usage:
    from cphmd.wham import run_wham, run_lmalf

    # WHAM analysis (default)
    run_wham(analysis_dir, nf=10, temp=298.15, nts0=1, nts1=1)

    # LMALF analysis (alternative)
    run_lmalf(analysis_dir, nf=10, temp=298.15)

With automatic G_imp computation:
    from cphmd.wham import run_wham_with_g_imp
    run_wham_with_g_imp(analysis_dir, alf_info, nf=10)
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

if TYPE_CHECKING:
    from cphmd.core.alf_utils import ALFInfo

logger = logging.getLogger(__name__)

# Path to the compiled WHAM shared library
_WHAM_LIB_PATH = Path(__file__).parent / "libwham.so"


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


def run_wham(
    analysis_dir: str | Path,
    nf: int,
    temp: float,
    nts0: int = 1,
    nts1: int = 1,
    use_gshift: bool = False,
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

    Raises:
        FileNotFoundError: If analysis directory or WHAM library not found.
        RuntimeError: If WHAM analysis fails.

    Note:
        WHAM expects the following files in the analysis directory:
        - Lambda/ directory with lambda trajectory files
        - Energy/ directory with cross-simulation energies
        - ../prep/nsubs file with system info
        - G_imp/ directory with entropy reference profiles
        - G_imp_shifts/ directory with shift tables (required if use_gshift=True)

        WHAM produces:
        - multisite/C.dat: Hessian matrix
        - multisite/V.dat: Gradient vector
        - f.dat: Free energy weights

    Example:
        >>> from cphmd.wham import run_wham
        >>> run_wham("./analysis/5", nf=10, temp=298.15)
        >>> # With G_imp shifts:
        >>> run_wham("./analysis/5", nf=10, temp=298.15, use_gshift=True)
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    # Check for compiled library
    if not _WHAM_LIB_PATH.exists():
        raise FileNotFoundError(
            f"WHAM library not found at {_WHAM_LIB_PATH}. "
            "Please compile: cd cphmd/wham/src && "
            "nvcc -shared -Xcompiler -fPIC -o ../libwham.so wham.cu"
        )

    # Create multisite output directory
    multisite_dir = analysis_dir / "multisite"
    multisite_dir.mkdir(exist_ok=True)

    # Run WHAM in analysis directory context
    # WHAM expects files in current directory, so we must chdir temporarily
    with _chdir_context(analysis_dir):
        try:
            logger.info(f"Running WHAM with nf={nf}, temp={temp}, use_gshift={use_gshift}")
            whamlib = ctypes.CDLL(str(_WHAM_LIB_PATH))
            pywham = whamlib.wham
            pywham.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            pywham.restype = ctypes.c_int

            result = pywham(nf, temp, nts0, nts1, int(use_gshift))
            if result != 0:
                raise RuntimeError(f"WHAM returned error code: {result}")
                
            logger.info("WHAM analysis completed successfully")
            
        except OSError as e:
            raise RuntimeError(
                f"Failed to load WHAM library: {e}. "
                "Ensure CUDA is available and library is compiled correctly."
            )


def check_wham_available() -> bool:
    """Check if WHAM library is available and compiled.

    Returns:
        True if WHAM library exists and can be loaded, False otherwise.
    """
    if not _WHAM_LIB_PATH.exists():
        return False
    
    try:
        ctypes.CDLL(str(_WHAM_LIB_PATH))
        return True
    except OSError:
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
    from cphmd.core.entropy import compute_g_imp, get_cache_path

    analysis_dir = Path(analysis_dir)
    alf_info = ensure_alf_info(alf_info)

    # Get constraint parameters
    constraint_type = alf_info.constraint_type
    fnex = alf_info.fnex
    fpie_width = alf_info.fpie_width
    fpie_force = alf_info.fpie_force
    bins = alf_info.g_imp_bins
    nsubs = alf_info.nsubs

    # Ensure G_imp files exist in cache
    cache_dir = get_cache_path(constraint_type, bins, fnex, fpie_width, fpie_force)

    for ndim in set(nsubs):
        if ndim < 2:
            continue
        g1_file = cache_dir / f"G1_{ndim}.dat"
        g2_file = cache_dir / f"G2_{ndim}.dat"

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

    # Create G_imp directory in analysis_dir and copy/link files
    g_imp_dir = analysis_dir / "G_imp"
    g_imp_dir.mkdir(exist_ok=True)

    for ndim in set(nsubs):
        if ndim < 2:
            continue

        # Copy files from cache to analysis directory
        # (Using copy instead of symlink for portability across filesystems)
        src_g1 = cache_dir / f"G1_{ndim}.dat"
        src_g2 = cache_dir / f"G2_{ndim}.dat"
        dst_g1 = g_imp_dir / f"G1_{ndim}.dat"
        dst_g2 = g_imp_dir / f"G2_{ndim}.dat"

        if src_g1.exists() and (force_recompute or not dst_g1.exists()):
            shutil.copy2(src_g1, dst_g1)
        if src_g2.exists() and (force_recompute or not dst_g2.exists()):
            shutil.copy2(src_g2, dst_g2)

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
    prepare_g_imp_for_wham(analysis_dir, alf_info, force_recompute_g_imp)

    # Run WHAM
    run_wham(
        analysis_dir=analysis_dir,
        nf=nf,
        temp=alf_info.temp,
        nts0=nts0,
        nts1=nts1,
        use_gshift=use_gshift,
    )


def run_lmalf(
    analysis_dir: str | Path,
    nf: int,
    temp: float,
    ms: int = 0,
    msprof: int = 0,
    max_iter: int = 0,
    tolerance: float = 0.0,
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

    Raises:
        FileNotFoundError: If analysis directory or library not found.
        RuntimeError: If LMALF analysis fails.

    Note:
        LMALF expects the following files in the analysis directory:
        - Lambda.dat: Combined lambda trajectory [frames x nblocks]
        - ensweight.dat: Ensemble weights [frames] (optional, defaults to 1.0)
        - ../../prep/nsubs: Number of subsites per site
        - x_prev.dat, s_prev.dat: Previous parameters (optional, for ms=1)

        LMALF produces:
        - OUT.dat: Optimized bias parameters in flat format

        The OUT.dat must be post-processed with GetFreeEnergyLM() to
        generate the b.dat, c.dat, x.dat, s.dat files.

    Example:
        >>> from cphmd.wham import run_lmalf
        >>> run_lmalf("./analysis/5", nf=10, temp=298.15)
        >>> # With custom convergence settings:
        >>> run_lmalf("./analysis/5", nf=10, temp=298.15, max_iter=500, tolerance=1e-4)
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    # Check for compiled library
    if not _WHAM_LIB_PATH.exists():
        raise FileNotFoundError(
            f"WHAM/LMALF library not found at {_WHAM_LIB_PATH}. "
            "Please compile: cd cphmd/wham && ./build.sh"
        )

    # Verify Lambda.dat exists
    lambda_file = analysis_dir / "Lambda.dat"
    if not lambda_file.exists():
        raise FileNotFoundError(
            f"Lambda.dat not found in {analysis_dir}. "
            "LMALF requires a combined Lambda.dat file."
        )

    # Run LMALF in analysis directory context
    with _chdir_context(analysis_dir):
        try:
            logger.info(f"Running LMALF with nf={nf}, temp={temp}, ms={ms}, msprof={msprof}")
            whamlib = ctypes.CDLL(str(_WHAM_LIB_PATH))
            pylmalf = whamlib.lmalf
            pylmalf.argtypes = [
                ctypes.c_int,     # nf
                ctypes.c_double,  # temp
                ctypes.c_int,     # ms
                ctypes.c_int,     # msprof
                ctypes.c_int,     # max_iter
                ctypes.c_double,  # tolerance
            ]
            pylmalf.restype = ctypes.c_int

            result = pylmalf(nf, temp, ms, msprof, max_iter, tolerance)
            if result != 0:
                raise RuntimeError(f"LMALF returned error code: {result}")

            logger.info("LMALF analysis completed successfully")

        except OSError as e:
            raise RuntimeError(
                f"Failed to load WHAM/LMALF library: {e}. "
                "Ensure CUDA is available and library is compiled correctly."
            )


def check_lmalf_available() -> bool:
    """Check if LMALF is available in the compiled library.

    Returns:
        True if LMALF function exists and can be called, False otherwise.
    """
    if not _WHAM_LIB_PATH.exists():
        return False

    try:
        whamlib = ctypes.CDLL(str(_WHAM_LIB_PATH))
        _ = whamlib.lmalf
        return True
    except (OSError, AttributeError):
        return False


def prepare_lmalf_input(
    analysis_dir: str | Path,
    lambda_files: list[str | Path],
    weight_files: list[str | Path] | None = None,
) -> None:
    """Prepare input files for LMALF analysis.

    Concatenates multiple lambda trajectory files into a single Lambda.dat
    and optionally prepares ensemble weights.

    Args:
        analysis_dir: Target analysis directory.
        lambda_files: List of lambda trajectory files to concatenate.
        weight_files: Optional list of weight files (same length as lambda_files).
            If not provided, all frames are weighted equally.

    Example:
        >>> from cphmd.wham import prepare_lmalf_input
        >>> lambda_files = [f"data/Lambda.{k}.{r}.dat" for k in range(3) for r in range(4)]
        >>> prepare_lmalf_input("./analysis/5", lambda_files)
    """
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate lambda files
    all_lambda = []
    for lf in lambda_files:
        lf = Path(lf)
        if lf.exists():
            data = np.loadtxt(lf)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            all_lambda.append(data)

    if not all_lambda:
        raise ValueError("No valid lambda files found")

    combined_lambda = np.vstack(all_lambda)
    np.savetxt(analysis_dir / "Lambda.dat", combined_lambda, fmt="%.8f")
    logger.info(f"Created Lambda.dat with {combined_lambda.shape[0]} frames, {combined_lambda.shape[1]} blocks")

    # Handle weights
    if weight_files:
        all_weights = []
        for wf in weight_files:
            wf = Path(wf)
            if wf.exists():
                weights = np.loadtxt(wf)
                if weights.ndim == 0:
                    weights = np.array([weights])
                all_weights.extend(weights.flatten())

        if all_weights:
            np.savetxt(analysis_dir / "ensweight.dat", all_weights, fmt="%.8f")
            logger.info(f"Created ensweight.dat with {len(all_weights)} weights")
    else:
        # Default uniform weights
        n_frames = combined_lambda.shape[0]
        np.savetxt(analysis_dir / "ensweight.dat", np.ones(n_frames), fmt="%.8f")
        logger.info(f"Created ensweight.dat with {n_frames} uniform weights")
