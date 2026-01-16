"""Core module - main CpHMD workflow components."""

from .patching import LigandPatchDef, PatchConfig, PatchParser, patch_system
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
    get_energy,
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
)

__all__ = [
    # Patching
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
    "get_energy",
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
]
