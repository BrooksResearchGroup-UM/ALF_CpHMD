"""Analysis module - post-simulation analysis tools.

Exports are resolved lazily so importing ``cphmd.analysis`` does not load
pyCHARMM-bound optional analysis modules.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    # Energy profiles
    "EnergyProfileConfig": "cphmd.analysis.energy_profiles",
    "EnergyProfileResult": "cphmd.analysis.energy_profiles",
    "analyze_energy_profiles": "cphmd.analysis.energy_profiles",
    "generate_simplex_grid": "cphmd.analysis.energy_profiles",
    "total_energy": "cphmd.analysis.energy_profiles",
    # Volume
    "VolumeConfig": "cphmd.analysis.volume",
    "VolumeResult": "cphmd.analysis.volume",
    "calculate_volume": "cphmd.analysis.volume",
    # DCA / Potts Model
    "DCAResult": "cphmd.analysis.dca",
    "get_model_dca": "cphmd.analysis.dca",
    "get_variance_dca": "cphmd.analysis.dca",
    "bootstrap_moments_dca": "cphmd.analysis.dca",
    # Population convergence
    "read_populations_from_runs": "cphmd.analysis.population_convergence",
    "plot_population_convergence": "cphmd.analysis.population_convergence",
    "generate_population_plots": "cphmd.analysis.population_convergence",
    # RMSD convergence
    "plot_rmsd_convergence": "cphmd.analysis.rmsd_convergence_plot",
    "plot_pairwise_rmsd_convergence": "cphmd.analysis.rmsd_convergence_plot",
    "plot_b_bias_convergence": "cphmd.analysis.rmsd_convergence_plot",
    "generate_rmsd_convergence_plots": "cphmd.analysis.rmsd_convergence_plot",
    # Henderson-Hasselbalch
    "HHFitResult": "cphmd.analysis.henderson_hasselbalch",
    "SubstatePopulation": "cphmd.analysis.henderson_hasselbalch",
    "SiteHHResult": "cphmd.analysis.henderson_hasselbalch",
    "logistic": "cphmd.analysis.henderson_hasselbalch",
    "three_state_hh": "cphmd.analysis.henderson_hasselbalch",
    "two_state_basic_hh": "cphmd.analysis.henderson_hasselbalch",
    "two_state_acidic_hh": "cphmd.analysis.henderson_hasselbalch",
    "compute_block_weights": "cphmd.analysis.henderson_hasselbalch",
    "compute_theoretical_populations": "cphmd.analysis.henderson_hasselbalch",
    "fit_hh_curve": "cphmd.analysis.henderson_hasselbalch",
    "plot_hh_curves": "cphmd.analysis.henderson_hasselbalch",
    "plot_site_substates": "cphmd.analysis.henderson_hasselbalch",
    "write_hh_csv": "cphmd.analysis.henderson_hasselbalch",
    "generate_hh_analysis": "cphmd.analysis.henderson_hasselbalch",
    # LDIN parser
    "StateInfo": "cphmd.analysis.ldin_parser",
    "SiteInfo": "cphmd.analysis.ldin_parser",
    "parse_block_str": "cphmd.analysis.ldin_parser",
    # pKa analysis
    "PKaAnalysisConfig": "cphmd.analysis.pka_analyzer",
    "PKaAnalyzer": "cphmd.analysis.pka_analyzer",
    "PKaResults": "cphmd.analysis.pka_analyzer",
    "SitePKaResult": "cphmd.analysis.pka_analyzer",
    # pKa data loading
    "discover_parquets": "cphmd.analysis.pka_data",
    "build_site_map": "cphmd.analysis.pka_data",
    "get_site_columns": "cphmd.analysis.pka_data",
    "load_lambda_data": "cphmd.analysis.pka_data",
    "apply_cutoff": "cphmd.analysis.pka_data",
    "skip_equilibration": "cphmd.analysis.pka_data",
    "compute_populations": "cphmd.analysis.pka_data",
    "compute_total_population": "cphmd.analysis.pka_data",
    "prepare_fit_data": "cphmd.analysis.pka_data",
    # pKa fitting
    "FitResult": "cphmd.analysis.pka_fitting",
    "MultiStateFitResult": "cphmd.analysis.pka_fitting",
    "sigmoid": "cphmd.analysis.pka_fitting",
    "make_multi_sigmoid": "cphmd.analysis.pka_fitting",
    "build_2state_guess": "cphmd.analysis.pka_fitting",
    "build_multistate_guess": "cphmd.analysis.pka_fitting",
    "quick_prefit": "cphmd.analysis.pka_fitting",
    "correct_pka": "cphmd.analysis.pka_fitting",
    "identify_transition_region": "cphmd.analysis.pka_fitting",
    "bootstrap_fit_2state": "cphmd.analysis.pka_fitting",
    "bootstrap_fit_multistate": "cphmd.analysis.pka_fitting",
    # pKa plots
    "RESNAME_LABELS": "cphmd.analysis.pka_plots",
    "plot_pka": "cphmd.analysis.pka_plots",
    "plot_pka_convergence": "cphmd.analysis.pka_plots",
    # WHAM free energy profiles
    "plot_wham_profiles": "cphmd.analysis.wham_profiles",
    # Plot styling
    "apply_pub_style": "cphmd.analysis.plot_style",
    "clean_axes": "cphmd.analysis.plot_style",
    "get_state_colors": "cphmd.analysis.plot_style",
    "savefig": "cphmd.analysis.plot_style",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
