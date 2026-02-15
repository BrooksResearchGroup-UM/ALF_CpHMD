"""Core module - main CpHMD workflow components."""

# Modules with pyCHARMM top-level imports are lazy-loaded via __getattr__
# so that mpi4py can initialize MPI before pyCHARMM does.

from typing import Literal

# Shared type aliases — defined here before sub-module imports so that
# alf_runner, charmm_utils, rmsd_convergence, etc. can import them without
# circular-import issues (the module object already has these attributes by
# the time the sub-modules execute their top-level imports).
PhaseType = Literal[1, 2, 3]
ElecType = Literal["pmeex", "pmeon", "pmenn", "fshift", "fswitch"]
VdwType = Literal["vswitch", "vfswitch"]

from .alf_runner import ALFConfig, ALFSimulation, run_alf_simulation
from .alf_utils import (
    ALFInfo,
    compute_bias_energy,
    compute_wham_inputs,
    convert_lambda_binary_to_parquet,
    get_energy_from_analysis_dir,
    init_vars,
    set_vars,
    set_vars_from_analysis_dir,
)
from .bias_analyzer import BiasAnalyzer
from .bias_constants import (
    CHI_OFFSET,
    DEFAULT_FNEX,
    OMEGA_DECAY,
    OMEGA_SCALE,
    BiasConstants,
    derive_bias_constants,
)
from .bias_guesser import (
    guess_initial_biases,
    guess_initial_biases_combined,
    guess_initial_biases_vacuum,
)
from .bias_search import (
    BiasSearchConfig,
    BiasSearchResult,
    run_bias_search,
)
from .block_builder import (
    BlockConfig,
    build_block_command,
    read_variable_file,
)
from .charmm_utils import (
    BoxParameters,
    CHARMMSession,
    FFTParameters,
    NonBondedConfig,
    define_selections,
    execute_block_command,
    read_structure,
    read_topology_files,
    setup_crystal,
    setup_nonbonded,
)
from .convergence_tracker import ConvergenceTracker
from .cphmd_params import (
    CpHMDParameters,
    SiteParameters,
    adjust_tags_for_effective_ph,
    compute_all_site_parameters,
    compute_per_unit_shift,
    get_delta_pKa_for_phase,
    replica_pH,
)
from .dynamics_runner import DynamicsRunner
from .entropy import (
    clear_cache,
    compute_g_imp,
    ensure_g_imp_available,
    get_cache_dir,
    get_cache_path,
)
from .expected_populations import (
    ExpectedPopulations,
    compute_expected_populations,
    compute_model_deviation,
)
from .free_energy import (
    FreeEnergyResult,
    fallback_bias_update,
    get_free_energy5,
    get_populations_from_lambda,
)
from .g_imp_provisioner import GImpProvisioner
from .generate_block import (
    BlockGeneratorConfig,
    BlockGeneratorResult,
    generate_block_files,
)
from .phase_switcher import (
    EWBSState,
    PhaseTransitionConfig,
    PKaFitResult,
    ReplicaLambdaData,
    StopCriteriaConfig,
    StopCriteriaResult,
    calculate_populations,
    check_phase3_stop,
    check_phase_transition,
    check_pka_convergence,
    check_pka_convergence_simple,
    check_stop_criteria,
    compute_block_variance,
    compute_entropy,
    compute_replica_populations,
    compute_rms_changes,
    compute_spread,
    ewbs_bottleneck_type,
    fit_pka_from_replicas,
    load_lambda_data,
    load_lambda_data_per_replica,
    update_ewbs_state,
    write_populations_file,
)
from .restraints import (
    generate_noe_restraints,
    generate_scat_restraints,
    write_restraint_file,
)
from .rmsd_convergence import (
    PairwiseRMSD,
    RMSDConvergenceConfig,
    RMSDState,
    check_rmsd_phase_transition,
    check_rmsd_stop,
    compute_pairwise_rmsd,
    compute_site_rmsd,
)
from .transitions import (
    TransitionResult,
    compute_connectivity_metric,
    compute_transition_matrix,
    find_weakest_transitions,
    get_transitions,
    save_transition_matrix,
    summarize_transitions,
    transition_matrix_to_coupling_weights,
)
from .variance import (
    VarianceResult,
    get_variance,
)

# Lazy-loaded names from .patching (has top-level pyCHARMM imports)
_PATCHING_NAMES = {
    "LigandPatchDef", "PatchConfig", "PatchParser", "patch_system",
}


def __getattr__(name):
    if name in _PATCHING_NAMES:
        from . import patching
        for attr in _PATCHING_NAMES:
            globals()[attr] = getattr(patching, attr)
        return globals()[name]
    raise AttributeError(f"module 'cphmd.core' has no attribute {name}")


__all__ = [
    # Shared type aliases
    "PhaseType",
    "ElecType",
    "VdwType",
    # Patching (lazy)
    "LigandPatchDef",
    "PatchConfig",
    "PatchParser",
    "patch_system",
    # ALF Runner + extracted sub-classes
    "ALFConfig",
    "ALFSimulation",
    "run_alf_simulation",
    "BiasAnalyzer",
    "guess_initial_biases",
    "guess_initial_biases_combined",
    "guess_initial_biases_vacuum",
    "ConvergenceTracker",
    "DynamicsRunner",
    "GImpProvisioner",
    # CpHMD Parameters
    "CpHMDParameters",
    "SiteParameters",
    "adjust_tags_for_effective_ph",
    "compute_all_site_parameters",
    "compute_per_unit_shift",
    "get_delta_pKa_for_phase",
    "replica_pH",
    # Block Builder
    "BlockConfig",
    "build_block_command",
    "read_variable_file",
    # Restraints
    "generate_scat_restraints",
    "generate_noe_restraints",
    "write_restraint_file",
    # CHARMM Utilities
    "BoxParameters",
    "FFTParameters",
    "NonBondedConfig",
    "CHARMMSession",
    "read_topology_files",
    "read_structure",
    "setup_crystal",
    "setup_nonbonded",
    "define_selections",
    "execute_block_command",
    # Bias Search
    "BiasSearchConfig",
    "BiasSearchResult",
    "run_bias_search",
    # Block Generator
    "BlockGeneratorConfig",
    "BlockGeneratorResult",
    "generate_block_files",
    # Bias Constants
    "BiasConstants",
    "derive_bias_constants",
    "DEFAULT_FNEX",
    "OMEGA_DECAY",
    "CHI_OFFSET",
    "OMEGA_SCALE",
    # ALF Utilities
    "ALFInfo",
    "init_vars",
    "set_vars",
    "set_vars_from_analysis_dir",
    "get_energy_from_analysis_dir",
    "compute_wham_inputs",
    "compute_bias_energy",
    "convert_lambda_binary_to_parquet",
    # Expected Populations (pH-aware targets)
    "ExpectedPopulations",
    "compute_expected_populations",
    "compute_model_deviation",
    # Free Energy Optimization
    "FreeEnergyResult",
    "get_free_energy5",
    "fallback_bias_update",
    "get_populations_from_lambda",
    # Entropy / G_imp
    "compute_g_imp",
    "ensure_g_imp_available",
    "get_cache_path",
    "get_cache_dir",
    "clear_cache",
    # Variance Estimation
    "VarianceResult",
    "get_variance",
    # Transition Analysis
    "TransitionResult",
    "get_transitions",
    "summarize_transitions",
    # RMSD Convergence
    "RMSDConvergenceConfig",
    "RMSDState",
    "PairwiseRMSD",
    "compute_site_rmsd",
    "compute_pairwise_rmsd",
    "check_rmsd_phase_transition",
    "check_rmsd_stop",
    # Phase Switching
    "EWBSState",
    "PhaseTransitionConfig",
    "StopCriteriaConfig",
    "ReplicaLambdaData",
    "PKaFitResult",
    "StopCriteriaResult",
    "check_phase_transition",
    "check_phase3_stop",
    "check_stop_criteria",
    "check_pka_convergence",
    "check_pka_convergence_simple",
    "load_lambda_data",
    "load_lambda_data_per_replica",
    "compute_block_variance",
    "compute_entropy",
    "compute_rms_changes",
    "compute_replica_populations",
    "compute_spread",
    "ewbs_bottleneck_type",
    "fit_pka_from_replicas",
    "update_ewbs_state",
    "calculate_populations",
    "write_populations_file",
]
