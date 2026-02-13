"""Analysis module - post-simulation analysis tools.

This module provides tools for analyzing ALF/CpHMD simulation results:
- Energy profile visualization and convergence tracking
- Volume analysis for hydration studies
"""

from .dca import (
    DCAResult,
    bootstrap_moments_dca,
    get_model_dca,
    get_variance_dca,
)
from .energy_profiles import (
    EnergyProfileConfig,
    EnergyProfileResult,
    analyze_energy_profiles,
    generate_simplex_grid,
    total_energy,
)
from .henderson_hasselbalch import (
    HHFitResult,
    SiteHHResult,
    SubstatePopulation,
    compute_block_weights,
    compute_theoretical_populations,
    fit_hh_curve,
    generate_hh_analysis,
    logistic,
    plot_hh_curves,
    plot_site_substates,
    three_state_hh,
    two_state_acidic_hh,
    two_state_basic_hh,
    write_hh_csv,
)
from .plot_style import (
    apply_pub_style,
    clean_axes,
    get_state_colors,
    savefig,
)
from .population_convergence import (
    generate_population_plots,
    plot_population_convergence,
    read_populations_from_runs,
)
from .rmsd_convergence_plot import (
    generate_rmsd_convergence_plots,
    plot_b_bias_convergence,
    plot_pairwise_rmsd_convergence,
    plot_rmsd_convergence,
)
from .volume import (
    VolumeConfig,
    VolumeResult,
    calculate_volume,
)
from .wham_profiles import (
    plot_wham_profiles,
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
    # RMSD convergence
    "plot_rmsd_convergence",
    "plot_pairwise_rmsd_convergence",
    "plot_b_bias_convergence",
    "generate_rmsd_convergence_plots",
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
    # WHAM free energy profiles
    "plot_wham_profiles",
    # Plot styling
    "apply_pub_style",
    "clean_axes",
    "get_state_colors",
    "savefig",
]
