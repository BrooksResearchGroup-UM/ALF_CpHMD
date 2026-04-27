"""Core module - main CpHMD workflow components."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Literal

PhaseType = Literal[1, 2, 3]
ElecType = Literal["pmeex", "pmeon", "pmenn", "fshift", "fswitch"]
VdwType = Literal["vswitch", "vfswitch"]
RestrainType = Literal["SCAT", "NOE", "none"]
PrepFormat = Literal["default", "legacy", "auto"]


def _warn(name: str, replacement: str) -> None:
    warnings.warn(
        f"cphmd.core.{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _stub_class(name: str, replacement: str):
    def _init(self, *args, **kwargs):
        raise RuntimeError(f"{name} is deprecated; use {replacement} instead.")

    return type(
        name,
        (),
        {
            "__module__": __name__,
            "__doc__": f"Deprecated compatibility stub for {replacement}.",
            "__init__": _init,
        },
    )


def _stub_function(name: str, replacement: str):
    def _stub(*args, **kwargs):
        raise RuntimeError(f"{name} is deprecated; use {replacement} instead.")

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__module__ = __name__
    _stub.__doc__ = f"Deprecated compatibility stub for {replacement}."
    return _stub


_EXPORTS: dict[str, tuple[str, str, str]] = {
    "ALFConfig": ("cphmd.training.config", "ALFConfig", "cphmd.training.config.ALFConfig"),
    "ProductionConfig": (
        "cphmd.training.production_hooks",
        "ProductionConfig",
        "cphmd.training.production_hooks.ProductionConfig",
    ),
    "ALFInfo": ("cphmd.core.alf_utils", "ALFInfo", "cphmd.core.alf_utils.ALFInfo"),
    "PatchSpec": ("cphmd.core.patch_spec", "PatchSpec", "cphmd.core.patch_spec.PatchSpec"),
    "PATCH_REGISTRY": (
        "cphmd.core.patch_spec",
        "PATCH_REGISTRY",
        "cphmd.core.patch_spec.PATCH_REGISTRY",
    ),
    "PatchApplier": (
        "cphmd.core.patch_applier",
        "PatchApplier",
        "cphmd.core.patch_applier.PatchApplier",
    ),
    "compute_bias_energy": (
        "cphmd.core.alf_utils",
        "compute_bias_energy",
        "cphmd.core.alf_utils.compute_bias_energy",
    ),
    "compute_wham_inputs": (
        "cphmd.core.alf_utils",
        "compute_wham_inputs",
        "cphmd.core.alf_utils.compute_wham_inputs",
    ),
    "convert_lambda_binary_to_parquet": (
        "cphmd.core.alf_utils",
        "convert_lambda_binary_to_parquet",
        "cphmd.core.alf_utils.convert_lambda_binary_to_parquet",
    ),
    "get_energy_from_analysis_dir": (
        "cphmd.core.alf_utils",
        "get_energy_from_analysis_dir",
        "cphmd.core.alf_utils.get_energy_from_analysis_dir",
    ),
    "init_vars": ("cphmd.core.alf_utils", "init_vars", "cphmd.core.alf_utils.init_vars"),
    "set_vars_from_analysis_dir": (
        "cphmd.core.alf_utils",
        "set_vars_from_analysis_dir",
        "cphmd.core.alf_utils.set_vars_from_analysis_dir",
    ),
    "BiasAnalyzer": (
        "cphmd.core.bias_analyzer",
        "BiasAnalyzer",
        "cphmd.core.bias_analyzer.BiasAnalyzer",
    ),
    "BiasConstants": (
        "cphmd.core.bias_constants",
        "BiasConstants",
        "cphmd.core.bias_constants.BiasConstants",
    ),
    "derive_bias_constants": (
        "cphmd.core.bias_constants",
        "derive_bias_constants",
        "cphmd.core.bias_constants.derive_bias_constants",
    ),
    "DEFAULT_FNEX": (
        "cphmd.core.bias_constants",
        "DEFAULT_FNEX",
        "cphmd.core.bias_constants.DEFAULT_FNEX",
    ),
    "OMEGA_DECAY": (
        "cphmd.core.bias_constants",
        "OMEGA_DECAY",
        "cphmd.core.bias_constants.OMEGA_DECAY",
    ),
    "CHI_OFFSET": (
        "cphmd.core.bias_constants",
        "CHI_OFFSET",
        "cphmd.core.bias_constants.CHI_OFFSET",
    ),
    "OMEGA_SCALE": (
        "cphmd.core.bias_constants",
        "OMEGA_SCALE",
        "cphmd.core.bias_constants.OMEGA_SCALE",
    ),
    "guess_initial_biases": (
        "cphmd.core.bias_guesser",
        "guess_initial_biases",
        "cphmd.core.bias_guesser.guess_initial_biases",
    ),
    "guess_initial_biases_combined": (
        "cphmd.core.bias_guesser",
        "guess_initial_biases_combined",
        "cphmd.core.bias_guesser.guess_initial_biases_combined",
    ),
    "guess_initial_biases_vacuum": (
        "cphmd.core.bias_guesser",
        "guess_initial_biases_vacuum",
        "cphmd.core.bias_guesser.guess_initial_biases_vacuum",
    ),
    "BiasSearchConfig": (
        "cphmd.core.bias_search",
        "BiasSearchConfig",
        "cphmd.core.bias_search.BiasSearchConfig",
    ),
    "BiasSearchResult": (
        "cphmd.core.bias_search",
        "BiasSearchResult",
        "cphmd.core.bias_search.BiasSearchResult",
    ),
    "run_bias_search": (
        "cphmd.core.bias_search",
        "run_bias_search",
        "cphmd.core.bias_search.run_bias_search",
    ),
    "ConvergenceTracker": (
        "cphmd.core.convergence_tracker",
        "ConvergenceTracker",
        "cphmd.core.convergence_tracker.ConvergenceTracker",
    ),
    "CpHMDParameters": (
        "cphmd.core.cphmd_params",
        "CpHMDParameters",
        "cphmd.core.cphmd_params.CpHMDParameters",
    ),
    "SiteParameters": (
        "cphmd.core.cphmd_params",
        "SiteParameters",
        "cphmd.core.cphmd_params.SiteParameters",
    ),
    "adjust_tags_for_effective_ph": (
        "cphmd.core.cphmd_params",
        "adjust_tags_for_effective_ph",
        "cphmd.core.cphmd_params.adjust_tags_for_effective_ph",
    ),
    "compute_all_site_parameters": (
        "cphmd.core.cphmd_params",
        "compute_all_site_parameters",
        "cphmd.core.cphmd_params.compute_all_site_parameters",
    ),
    "compute_per_unit_shift": (
        "cphmd.core.cphmd_params",
        "compute_per_unit_shift",
        "cphmd.core.cphmd_params.compute_per_unit_shift",
    ),
    "get_delta_pKa_for_phase": (
        "cphmd.core.cphmd_params",
        "get_delta_pKa_for_phase",
        "cphmd.core.cphmd_params.get_delta_pKa_for_phase",
    ),
    "replica_pH": (
        "cphmd.core.cphmd_params",
        "replica_pH",
        "cphmd.core.cphmd_params.replica_pH",
    ),
    "clear_cache": ("cphmd.core.entropy", "clear_cache", "cphmd.core.entropy.clear_cache"),
    "compute_g_imp": ("cphmd.core.entropy", "compute_g_imp", "cphmd.core.entropy.compute_g_imp"),
    "ensure_g_imp_available": (
        "cphmd.core.entropy",
        "ensure_g_imp_available",
        "cphmd.core.entropy.ensure_g_imp_available",
    ),
    "get_cache_dir": ("cphmd.core.entropy", "get_cache_dir", "cphmd.core.entropy.get_cache_dir"),
    "get_cache_path": ("cphmd.core.entropy", "get_cache_path", "cphmd.core.entropy.get_cache_path"),
    "ExpectedPopulations": (
        "cphmd.core.expected_populations",
        "ExpectedPopulations",
        "cphmd.core.expected_populations.ExpectedPopulations",
    ),
    "compute_expected_populations": (
        "cphmd.core.expected_populations",
        "compute_expected_populations",
        "cphmd.core.expected_populations.compute_expected_populations",
    ),
    "compute_model_deviation": (
        "cphmd.core.expected_populations",
        "compute_model_deviation",
        "cphmd.core.expected_populations.compute_model_deviation",
    ),
    "FreeEnergyResult": (
        "cphmd.core.free_energy",
        "FreeEnergyResult",
        "cphmd.core.free_energy.FreeEnergyResult",
    ),
    "fallback_bias_update": (
        "cphmd.core.free_energy",
        "fallback_bias_update",
        "cphmd.core.free_energy.fallback_bias_update",
    ),
    "get_free_energy5": (
        "cphmd.core.free_energy",
        "get_free_energy5",
        "cphmd.core.free_energy.get_free_energy5",
    ),
    "get_populations_from_lambda": (
        "cphmd.core.free_energy",
        "get_populations_from_lambda",
        "cphmd.core.free_energy.get_populations_from_lambda",
    ),
    "GImpProvisioner": (
        "cphmd.core.g_imp_provisioner",
        "GImpProvisioner",
        "cphmd.core.g_imp_provisioner.GImpProvisioner",
    ),
    "PhaseTransitionConfig": (
        "cphmd.core.phase_switcher",
        "PhaseTransitionConfig",
        "cphmd.core.phase_switcher.PhaseTransitionConfig",
    ),
    "PKaFitResult": (
        "cphmd.core.phase_switcher",
        "PKaFitResult",
        "cphmd.core.phase_switcher.PKaFitResult",
    ),
    "ReplicaLambdaData": (
        "cphmd.core.phase_switcher",
        "ReplicaLambdaData",
        "cphmd.core.phase_switcher.ReplicaLambdaData",
    ),
    "StopCriteriaConfig": (
        "cphmd.core.phase_switcher",
        "StopCriteriaConfig",
        "cphmd.core.phase_switcher.StopCriteriaConfig",
    ),
    "StopCriteriaResult": (
        "cphmd.core.phase_switcher",
        "StopCriteriaResult",
        "cphmd.core.phase_switcher.StopCriteriaResult",
    ),
    "EWBSState": ("cphmd.core.phase_switcher", "EWBSState", "cphmd.core.phase_switcher.EWBSState"),
    "calculate_populations": (
        "cphmd.core.phase_switcher",
        "calculate_populations",
        "cphmd.core.phase_switcher.calculate_populations",
    ),
    "check_phase3_stop": (
        "cphmd.core.phase_switcher",
        "check_phase3_stop",
        "cphmd.core.phase_switcher.check_phase3_stop",
    ),
    "check_phase_transition": (
        "cphmd.core.phase_switcher",
        "check_phase_transition",
        "cphmd.core.phase_switcher.check_phase_transition",
    ),
    "check_pka_convergence": (
        "cphmd.core.phase_switcher",
        "check_pka_convergence",
        "cphmd.core.phase_switcher.check_pka_convergence",
    ),
    "check_pka_convergence_simple": (
        "cphmd.core.phase_switcher",
        "check_pka_convergence_simple",
        "cphmd.core.phase_switcher.check_pka_convergence_simple",
    ),
    "check_stop_criteria": (
        "cphmd.core.phase_switcher",
        "check_stop_criteria",
        "cphmd.core.phase_switcher.check_stop_criteria",
    ),
    "compute_block_variance": (
        "cphmd.core.phase_switcher",
        "compute_block_variance",
        "cphmd.core.phase_switcher.compute_block_variance",
    ),
    "compute_entropy": (
        "cphmd.core.phase_switcher",
        "compute_entropy",
        "cphmd.core.phase_switcher.compute_entropy",
    ),
    "compute_replica_populations": (
        "cphmd.core.phase_switcher",
        "compute_replica_populations",
        "cphmd.core.phase_switcher.compute_replica_populations",
    ),
    "compute_rms_changes": (
        "cphmd.core.phase_switcher",
        "compute_rms_changes",
        "cphmd.core.phase_switcher.compute_rms_changes",
    ),
    "compute_spread": (
        "cphmd.core.phase_switcher",
        "compute_spread",
        "cphmd.core.phase_switcher.compute_spread",
    ),
    "count_transition_events": (
        "cphmd.core.phase_switcher",
        "count_transition_events",
        "cphmd.core.phase_switcher.count_transition_events",
    ),
    "enough_transitions": (
        "cphmd.core.phase_switcher",
        "enough_transitions",
        "cphmd.core.phase_switcher.enough_transitions",
    ),
    "ewbs_bottleneck_type": (
        "cphmd.core.phase_switcher",
        "ewbs_bottleneck_type",
        "cphmd.core.phase_switcher.ewbs_bottleneck_type",
    ),
    "fit_pka_from_replicas": (
        "cphmd.core.phase_switcher",
        "fit_pka_from_replicas",
        "cphmd.core.phase_switcher.fit_pka_from_replicas",
    ),
    "load_lambda_data": (
        "cphmd.core.phase_switcher",
        "load_lambda_data",
        "cphmd.core.phase_switcher.load_lambda_data",
    ),
    "load_lambda_data_per_replica": (
        "cphmd.core.phase_switcher",
        "load_lambda_data_per_replica",
        "cphmd.core.phase_switcher.load_lambda_data_per_replica",
    ),
    "update_ewbs_state": (
        "cphmd.core.phase_switcher",
        "update_ewbs_state",
        "cphmd.core.phase_switcher.update_ewbs_state",
    ),
    "write_populations_file": (
        "cphmd.core.phase_switcher",
        "write_populations_file",
        "cphmd.core.phase_switcher.write_populations_file",
    ),
    "PairwiseRMSD": (
        "cphmd.core.rmsd_convergence",
        "PairwiseRMSD",
        "cphmd.core.rmsd_convergence.PairwiseRMSD",
    ),
    "RMSDConvergenceConfig": (
        "cphmd.core.rmsd_convergence",
        "RMSDConvergenceConfig",
        "cphmd.core.rmsd_convergence.RMSDConvergenceConfig",
    ),
    "RMSDState": (
        "cphmd.core.rmsd_convergence",
        "RMSDState",
        "cphmd.core.rmsd_convergence.RMSDState",
    ),
    "check_rmsd_phase_transition": (
        "cphmd.core.rmsd_convergence",
        "check_rmsd_phase_transition",
        "cphmd.core.rmsd_convergence.check_rmsd_phase_transition",
    ),
    "check_rmsd_stop": (
        "cphmd.core.rmsd_convergence",
        "check_rmsd_stop",
        "cphmd.core.rmsd_convergence.check_rmsd_stop",
    ),
    "compute_pairwise_rmsd": (
        "cphmd.core.rmsd_convergence",
        "compute_pairwise_rmsd",
        "cphmd.core.rmsd_convergence.compute_pairwise_rmsd",
    ),
    "compute_site_rmsd": (
        "cphmd.core.rmsd_convergence",
        "compute_site_rmsd",
        "cphmd.core.rmsd_convergence.compute_site_rmsd",
    ),
    "TransitionResult": (
        "cphmd.core.transitions",
        "TransitionResult",
        "cphmd.core.transitions.TransitionResult",
    ),
    "compute_connectivity_metric": (
        "cphmd.core.transitions",
        "compute_connectivity_metric",
        "cphmd.core.transitions.compute_connectivity_metric",
    ),
    "compute_transition_matrix": (
        "cphmd.core.transitions",
        "compute_transition_matrix",
        "cphmd.core.transitions.compute_transition_matrix",
    ),
    "find_weakest_transitions": (
        "cphmd.core.transitions",
        "find_weakest_transitions",
        "cphmd.core.transitions.find_weakest_transitions",
    ),
    "get_transitions": (
        "cphmd.core.transitions",
        "get_transitions",
        "cphmd.core.transitions.get_transitions",
    ),
    "save_transition_matrix": (
        "cphmd.core.transitions",
        "save_transition_matrix",
        "cphmd.core.transitions.save_transition_matrix",
    ),
    "summarize_transitions": (
        "cphmd.core.transitions",
        "summarize_transitions",
        "cphmd.core.transitions.summarize_transitions",
    ),
    "transition_matrix_to_coupling_weights": (
        "cphmd.core.transitions",
        "transition_matrix_to_coupling_weights",
        "cphmd.core.transitions.transition_matrix_to_coupling_weights",
    ),
    "VarianceResult": (
        "cphmd.core.variance",
        "VarianceResult",
        "cphmd.core.variance.VarianceResult",
    ),
    "get_variance": ("cphmd.core.variance", "get_variance", "cphmd.core.variance.get_variance"),
    "LigandPatchDef": (
        "cphmd.core.patching",
        "LigandPatchDef",
        "cphmd.core.patching.LigandPatchDef",
    ),
    "LigandSiteDef": (
        "cphmd.core.patching",
        "LigandSiteDef",
        "cphmd.core.patching.LigandSiteDef",
    ),
    "PatchConfig": ("cphmd.core.patching", "PatchConfig", "cphmd.core.patching.PatchConfig"),
    "PatchParser": ("cphmd.core.patching", "PatchParser", "cphmd.core.patching.PatchParser"),
    "patch_system": ("cphmd.core.patching", "patch_system", "cphmd.core.patching.patch_system"),
}


_STUB_EXPORTS: dict[str, tuple[str, str]] = {
    "BlockConfig": ("cphmd.native.block", "class"),
    "build_block_command": ("cphmd.native.block", "function"),
    "read_variable_file": ("cphmd.native.block", "function"),
    "BoxParameters": ("cphmd.native.system", "class"),
    "FFTParameters": ("cphmd.native.system", "class"),
    "NonBondedConfig": ("cphmd.native.system", "class"),
    "CHARMMSession": ("cphmd.native.system", "class"),
    "read_topology_files": ("cphmd.native.system", "function"),
    "read_structure": ("cphmd.native.system", "function"),
    "setup_crystal": ("cphmd.native.system", "function"),
    "setup_nonbonded": ("cphmd.native.system", "function"),
    "define_selections": ("cphmd.native.system", "function"),
    "execute_block_command": ("cphmd.native.block", "function"),
    "BlockGeneratorConfig": ("cphmd.native.block", "class"),
    "BlockGeneratorResult": ("cphmd.native.block", "class"),
    "generate_block_files": ("cphmd.native.block", "function"),
    "build_nsubsites_str": ("cphmd.training.production_hooks.ProductionHooks", "function"),
    "build_parquet_metadata": ("cphmd.training.production_hooks.ProductionHooks", "function"),
    "find_resume_point": ("cphmd.training.production_hooks.ProductionHooks", "function"),
    "find_restart_for_chunk": ("cphmd.training.production_hooks.ProductionHooks", "function"),
    "generate_scat_restraints": ("cphmd.native.system.add_restraint", "function"),
    "generate_noe_restraints": ("cphmd.native.system.add_restraint", "function"),
    "write_restraint_file": ("cphmd.native.system.add_restraint", "function"),
}

_DEPRECATED_EXPORT_NAMES = {
    "ALFConfig",
    "ProductionConfig",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name, replacement = _EXPORTS[name]
        if name in _DEPRECATED_EXPORT_NAMES:
            _warn(name, replacement)
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name in _STUB_EXPORTS:
        replacement, kind = _STUB_EXPORTS[name]
        _warn(name, replacement)
        value = (
            _stub_class(name, replacement) if kind == "class" else _stub_function(name, replacement)
        )
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(
    dict.fromkeys(
        [
            "PhaseType",
            "ElecType",
            "VdwType",
            "RestrainType",
            "PrepFormat",
            *_EXPORTS,
            *_STUB_EXPORTS,
        ]
    )
)
