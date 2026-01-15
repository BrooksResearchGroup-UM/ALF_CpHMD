"""Analysis module - post-simulation analysis tools.

This module provides tools for analyzing ALF/CpHMD simulation results:
- Energy profile visualization and convergence tracking
- Volume analysis for hydration studies
"""

from .energy_profiles import (
    EnergyProfileConfig,
    EnergyProfileResult,
    analyze_energy_profiles,
    generate_simplex_grid,
    total_energy,
)

from .volume import (
    VolumeConfig,
    VolumeResult,
    calculate_volume,
)

__all__ = [
    # Energy profiles
    "EnergyProfileConfig",
    "EnergyProfileResult",
    "analyze_energy_profiles",
    "generate_simplex_grid",
    "total_energy",
    # Volume
    "VolumeConfig",
    "VolumeResult",
    "calculate_volume",
]
