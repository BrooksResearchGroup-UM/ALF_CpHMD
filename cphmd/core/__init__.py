"""Core module - main CpHMD workflow components."""

# Modules with pyCHARMM top-level imports are lazy-loaded via __getattr__
# so that mpi4py can initialize MPI before pyCHARMM does.

from .alf_runner import ALFConfig, ALFSimulation, run_alf_simulation
from .cphmd_params import (
    CpHMDParameters,
    SiteParameters,
    compute_all_site_parameters,
    compute_bias_shifts,
    get_delta_pKa_for_phase,
)
from .block_builder import (
    BlockConfig,
    build_block_command,
    read_variable_file,
)
from .restraints import (
    generate_scat_restraints,
    generate_noe_restraints,
    write_restraint_file,
)
from .charmm_utils import (
    BoxParameters,
    FFTParameters,
    NonBondedConfig,
    CHARMMSession,
    read_topology_files,
    read_structure,
    setup_crystal,
    setup_nonbonded,
    define_selections,
    execute_block_command,
)
from .bias_search import (
    BiasSearchConfig,
    BiasSearchResult,
    run_bias_search,
)
from .generate_block import (
    BlockGeneratorConfig,
    BlockGeneratorResult,
    generate_block_files,
)
from .alf_utils import (
    ALFInfo,
    init_vars,
    set_vars,
    set_vars_from_analysis_dir,
    get_energy_from_analysis_dir,
    compute_bias_energy,
    write_lambda_text,
    convert_lambda_binary_to_text,
)
from .free_energy import (
    FreeEnergyResult,
    get_free_energy5,
    fallback_bias_update,
    get_populations_from_lambda,
)
from .entropy import (
    compute_g_imp,
    ensure_g_imp_available,
    get_cache_path,
    get_cache_dir,
    clear_cache,
)
from .variance import (
    VarianceResult,
    get_variance,
)
from .transitions import (
    TransitionResult,
    get_transitions,
    summarize_transitions,
    compute_transition_matrix,
    transition_matrix_to_coupling_weights,
    save_transition_matrix,
    compute_connectivity_metric,
    find_weakest_transitions,
)
from .rmsd_convergence import (
    RMSDConvergenceConfig,
    RMSDState,
    PairwiseRMSD,
    compute_site_rmsd,
    compute_pairwise_rmsd,
    check_rmsd_phase_transition,
    check_rmsd_stop,
)
from .phase_switcher import (
    PhaseTransitionConfig,
    StopCriteriaConfig,
    ReplicaLambdaData,
    PKaFitResult,
    StopCriteriaResult,
    check_phase_transition,
    check_phase3_stop,
    check_stop_criteria,
    check_pka_convergence,
    check_pka_convergence_simple,
    load_lambda_data,
    load_lambda_data_per_replica,
    compute_replica_populations,
    compute_spread,
    fit_pka_from_replicas,
    calculate_populations,
    write_populations_file,
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
    # Patching (lazy)
    "LigandPatchDef",
    "PatchConfig",
    "PatchParser",
    "patch_system",
    # ALF Runner
    "ALFConfig",
    "ALFSimulation",
    "run_alf_simulation",
    # CpHMD Parameters
    "CpHMDParameters",
    "SiteParameters",
    "compute_all_site_parameters",
    "compute_bias_shifts",
    "get_delta_pKa_for_phase",
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
    # ALF Utilities
    "ALFInfo",
    "init_vars",
    "set_vars",
    "set_vars_from_analysis_dir",
    "get_energy_from_analysis_dir",
    "compute_bias_energy",
    "write_lambda_text",
    "convert_lambda_binary_to_text",
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
    "compute_replica_populations",
    "compute_spread",
    "fit_pka_from_replicas",
    "calculate_populations",
    "write_populations_file",
]
