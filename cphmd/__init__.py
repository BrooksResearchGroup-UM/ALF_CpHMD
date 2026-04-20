"""
CpHMD - Constant pH Molecular Dynamics with ALF

A Python package for running ALF-based constant pH molecular dynamics
simulations using CHARMM/pyCHARMM.
"""

from importlib import resources
from pathlib import Path

__version__ = "0.1.0"
__author__ = "Stanislav Cherepanov"

# Package root directory
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _resolve_toppar_dir() -> Path:
    fallback = PROJECT_ROOT / "toppar"
    try:
        packaged = Path(str(resources.files("toppar")))
    except ModuleNotFoundError:
        return fallback
    if packaged.exists():
        return packaged
    return fallback


# Default paths
TOPPAR_DIR = _resolve_toppar_dir()
