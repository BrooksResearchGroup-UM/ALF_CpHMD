"""
CpHMD - Constant pH Molecular Dynamics with ALF

A Python package for running ALF-based constant pH molecular dynamics
simulations using CHARMM/pyCHARMM.
"""

__version__ = "0.1.0"
__author__ = "Stanislav Cherepanov"

from pathlib import Path

# Package root directory
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Default paths
TOPPAR_DIR = PROJECT_ROOT / "toppar"
