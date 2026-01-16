"""WHAM (Weighted Histogram Analysis Method) module.

This module provides GPU-accelerated WHAM analysis for ALF simulations.
Bundled from wham_3 variant with G-shift support for variable blocks.

The WHAM library must be compiled before use:
    cd cphmd/wham/src
    nvcc -shared -Xcompiler -fPIC -o ../libwham.so wham.cu

Usage:
    from cphmd.wham import run_wham
    run_wham(analysis_dir, nf=10, temp=298.15, nts0=1, nts1=1)

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
