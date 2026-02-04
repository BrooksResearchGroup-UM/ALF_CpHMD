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

from .dca import (
    DCAResult,
    get_model_dca,
    get_variance_dca,
    bootstrap_moments_dca,
)
from .population_convergence import (
    read_populations_from_runs,
    plot_population_convergence,
    generate_population_plots,
)
from .henderson_hasselbalch import (
    HHFitResult,
    SubstatePopulation,
    SiteHHResult,
    logistic,
    three_state_hh,
    two_state_basic_hh,
    two_state_acidic_hh,
    compute_block_weights,
    compute_theoretical_populations,
    fit_hh_curve,
    plot_hh_curves,
    plot_site_substates,
    write_hh_csv,
    generate_hh_analysis,
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
    # DCA / Potts Model
    "DCAResult",
    "get_model_dca",
    "get_variance_dca",
    "bootstrap_moments_dca",
    # Population convergence
    "read_populations_from_runs",
    "plot_population_convergence",
    "generate_population_plots",
    # Henderson-Hasselbalch
    "HHFitResult",
    "SubstatePopulation",
    "SiteHHResult",
    "logistic",
    "three_state_hh",
    "two_state_basic_hh",
    "two_state_acidic_hh",
    "compute_block_weights",
    "compute_theoretical_populations",
    "fit_hh_curve",
    "plot_hh_curves",
    "plot_site_substates",
    "write_hh_csv",
    "generate_hh_analysis",
]
