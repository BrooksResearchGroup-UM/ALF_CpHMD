"""YAML configuration loading and merging for CpHMD workflows.

Config chain priority (last wins):
    package defaults → local cphmd_config.yaml → --config flag → CLI flags

The loader reads YAML files, merges them following the priority chain,
and converts the merged dict into the appropriate dataclass
(ALFConfig, PatchConfig, SolvationConfig).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cphmd import TOPPAR_DIR

logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_LOCAL_CONFIG_NAME = "cphmd_config.yaml"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents. Returns empty dict for empty files.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple config dicts with later values taking priority.

    Performs a shallow merge — nested dicts are replaced, not deep-merged,
    which matches the flat structure of the config YAML files.

    Args:
        *configs: Config dicts in priority order (last wins).

    Returns:
        Merged config dict.
    """
    merged: dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            merged.update(cfg)
    return merged


def _load_defaults(section: str) -> dict[str, Any]:
    """Load package-default YAML for a given section.

    Args:
        section: One of "alf", "patch", "solvation".

    Returns:
        Default config dict. Empty dict if no defaults file exists.
    """
    path = _DEFAULTS_DIR / f"{section}.yaml"
    if path.exists():
        return load_yaml_config(path)
    return {}


def _load_local_config() -> dict[str, Any]:
    """Load local cphmd_config.yaml from the current working directory.

    Returns:
        Config dict, or empty dict if no local config exists.
    """
    path = Path.cwd() / _LOCAL_CONFIG_NAME
    if path.exists():
        logger.info("Loading local config: %s", path)
        return load_yaml_config(path)
    return {}


def _resolve_config_chain(
    section: str,
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full config dict for a section following the priority chain.

    Priority (last wins):
        1. Package defaults (cphmd/config/defaults/<section>.yaml)
        2. Local cphmd_config.yaml[section]
        3. --config file[section]
        4. CLI flag overrides

    Args:
        section: Config section name ("alf", "patch", "solvation", "build").
        config_path: Optional path to a --config YAML file.
        cli_overrides: Optional dict of CLI flag overrides (None values excluded).

    Returns:
        Merged config dict for the section.
    """
    # 1. Package defaults
    defaults = _load_defaults(section)

    # 2. Local config
    local_full = _load_local_config()
    local_section = local_full.get(section, {}) or {}

    # 3. --config file
    file_section: dict[str, Any] = {}
    if config_path is not None:
        file_full = load_yaml_config(config_path)
        file_section = file_full.get(section, {}) or {}

    # 4. CLI overrides (strip None values — those are unset flags)
    overrides: dict[str, Any] = {}
    if cli_overrides:
        overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    return merge_configs(defaults, local_section, file_section, overrides)


def config_to_alf(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> "ALFConfig":
    """Build an ALFConfig from the config chain.

    Args:
        config_path: Optional --config YAML file.
        cli_overrides: Optional CLI flag overrides.

    Returns:
        Populated ALFConfig dataclass.
    """
    from cphmd.core.alf_runner import ALFConfig

    cfg = _resolve_config_chain("alf", config_path, cli_overrides)

    # Map YAML key aliases to ALFConfig field names
    if "cleanup" in cfg and "cleanup_old_analysis" not in cfg:
        cfg["cleanup_old_analysis"] = cfg.pop("cleanup")
    elif "cleanup" in cfg:
        cfg.pop("cleanup")

    if "hh_plots" in cfg and "generate_hh_plots" not in cfg:
        cfg["generate_hh_plots"] = cfg.pop("hh_plots")
    elif "hh_plots" in cfg:
        cfg.pop("hh_plots")

    if "auto_phase" in cfg and "auto_phase_switch" not in cfg:
        cfg["auto_phase_switch"] = cfg.pop("auto_phase")
    elif "auto_phase" in cfg:
        cfg.pop("auto_phase")

    # Ensure toppar_dir is set
    cfg.setdefault("toppar_dir", str(TOPPAR_DIR))

    # Handle g_imp_bins string parsing (from YAML it could be int or list)
    g_imp_bins = cfg.get("g_imp_bins")
    if isinstance(g_imp_bins, str):
        if "," in g_imp_bins:
            cfg["g_imp_bins"] = [int(x.strip()) for x in g_imp_bins.split(",")]
        else:
            cfg["g_imp_bins"] = int(g_imp_bins)

    # Handle endpoint_weight string parsing (from YAML it could be float or list)
    ew = cfg.get("endpoint_weight")
    if isinstance(ew, str):
        if "," in ew:
            cfg["endpoint_weight"] = [float(x.strip()) for x in ew.split(",")]
        else:
            cfg["endpoint_weight"] = float(ew)

    # Handle analysis_window string parsing (int or per-phase list[int])
    aw = cfg.get("analysis_window")
    if isinstance(aw, str):
        if "," in aw:
            cfg["analysis_window"] = [int(x.strip()) for x in aw.split(",")]
        else:
            cfg["analysis_window"] = int(aw)

    # Handle analysis_skip string parsing (int or per-phase list[int])
    ask = cfg.get("analysis_skip")
    if isinstance(ask, str):
        if "," in ask:
            cfg["analysis_skip"] = [int(x.strip()) for x in ask.split(",")]
        else:
            cfg["analysis_skip"] = int(ask)

    # Handle endpoint_decay string parsing (same per-phase pattern)
    ed = cfg.get("endpoint_decay")
    if isinstance(ed, str):
        if "," in ed:
            cfg["endpoint_decay"] = [float(x.strip()) for x in ed.split(",")]
        else:
            cfg["endpoint_decay"] = float(ed)

    # Warn if bias_type and individual no_*_bias flags are both specified
    _bias_flags = {"no_b_bias", "no_c_bias", "no_x_bias", "no_s_bias", "no_t_bias", "no_u_bias"}
    if "bias_type" in cfg and any(k in cfg for k in _bias_flags):
        import warnings

        warnings.warn(
            "Both 'bias_type' and individual no_*_bias flags are set. "
            "'bias_type' takes precedence; remove the individual flags to silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Handle replica_exchange: nested dict → ReplicaExchangeConfig, bool → enabled flag
    if "replica_exchange" in cfg:
        re_cfg = cfg["replica_exchange"]
        if isinstance(re_cfg, bool):
            from cphmd.core.replica_exchange import ReplicaExchangeConfig

            cfg["replica_exchange"] = ReplicaExchangeConfig(enabled=re_cfg)
        elif isinstance(re_cfg, dict):
            from cphmd.core.replica_exchange import ReplicaExchangeConfig

            cfg["replica_exchange"] = ReplicaExchangeConfig(**re_cfg)

    # Deprecated: preset_config is now always auto-derived from elec_type + vdw_type
    if "preset_config" in cfg:
        import warnings

        warnings.warn(
            "preset_config is deprecated and will be ignored. "
            "Preset configuration is now auto-derived from elec_type and vdw_type.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg.pop("preset_config")

    # Handle phase_transition: nested dict → PhaseTransitionConfig
    if "phase_transition" in cfg:
        pt_cfg = cfg["phase_transition"]
        if isinstance(pt_cfg, dict):
            from cphmd.core.phase_switcher import PhaseTransitionConfig

            cfg["phase_transition"] = PhaseTransitionConfig(**pt_cfg)

    return ALFConfig(**cfg)


def config_to_patch(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> "PatchConfig":
    """Build a PatchConfig from the config chain.

    Args:
        config_path: Optional --config YAML file.
        cli_overrides: Optional CLI flag overrides.

    Returns:
        Populated PatchConfig dataclass.
    """
    from cphmd.core.patching import PatchConfig

    cfg = _resolve_config_chain("patch", config_path, cli_overrides)

    # Handle ligand_patches list of dicts → LigandPatchDef objects
    if "ligand_patches" in cfg:
        from cphmd.core.patching import LigandPatchDef

        cfg["ligand_patches"] = [
            LigandPatchDef(**lp) if isinstance(lp, dict) else lp for lp in cfg["ligand_patches"]
        ]

    return PatchConfig(**cfg)


def config_to_solvation(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> "SolvationConfig":
    """Build a SolvationConfig from the config chain.

    Args:
        config_path: Optional --config YAML file.
        cli_overrides: Optional CLI flag overrides.

    Returns:
        Populated SolvationConfig dataclass.
    """
    from cphmd.setup.solvate import SolvationConfig

    cfg = _resolve_config_chain("solvation", config_path, cli_overrides)

    # Map YAML alias
    if "salt" in cfg and "salt_concentration" not in cfg:
        cfg["salt_concentration"] = cfg.pop("salt")
    elif "salt" in cfg:
        cfg.pop("salt")

    return SolvationConfig(**cfg)


def run_workflow(
    config_path: str | Path,
    step: str = "all",
) -> None:
    """Run one or more CpHMD workflow steps from a config file.

    Dispatches to the appropriate functions based on the step parameter.
    Steps are executed in order: build → solvate → patch → alf.

    Args:
        config_path: Path to the YAML config file.
        step: Which step to run. One of:
            "build"    — Create amino acid template structures
            "solvate"  — Solvate the system
            "patch"    — Apply CpHMD patches
            "alf"      — Run ALF simulation
            "all"      — Run all steps in order (skips steps without config)

    Raises:
        ValueError: If step is not recognized.
        FileNotFoundError: If config_path does not exist.
    """
    valid_steps = {"build", "solvate", "patch", "alf", "all"}
    if step not in valid_steps:
        raise ValueError(f"Unknown step {step!r}. Must be one of: {valid_steps}")

    config_path = Path(config_path)
    full_config = load_yaml_config(config_path)

    steps = ["build", "solvate", "patch", "alf"] if step == "all" else [step]

    # Map step names to YAML section names (step "solvate" → section "solvation")
    step_to_section = {"solvate": "solvation"}

    for current_step in steps:
        section_key = step_to_section.get(current_step, current_step)
        section = full_config.get(section_key)
        if section is None:
            if step == "all":
                logger.info("Skipping %s (no config section)", current_step)
                continue
            raise ValueError(f"Config file has no '{current_step}' section: {config_path}")

        logger.info("Running step: %s", current_step)

        if current_step == "build":
            _run_build(section)
        elif current_step == "solvate":
            _run_solvate(config_path)
        elif current_step == "patch":
            _run_patch(config_path)
        elif current_step == "alf":
            _run_alf(config_path)


def _run_build(build_config: dict[str, Any]) -> None:
    """Run the build step (create amino acid templates)."""
    from cphmd.setup.create_aa import create_amino_acid

    residue = build_config.get("residue")
    if residue is None:
        raise ValueError("build section must specify 'residue'")

    output_dir = build_config.get("output_dir", "pdb")
    template = build_config.get("template", "ALA ALA {res} ALA ALA")

    logger.info("Building template for %s in %s/", residue, output_dir)
    result = create_amino_acid(
        residue=residue,
        output_dir=output_dir,
        template=template,
    )
    if result:
        logger.info("Created: %s", result)


def _run_solvate(config_path: Path) -> None:
    """Run the solvate step."""
    from cphmd.setup.solvate import solvate_system

    config = config_to_solvation(config_path)
    result_dir = solvate_system(config)
    logger.info("Solvation complete: %s", result_dir)


def _run_patch(config_path: Path) -> None:
    """Run the patch step."""
    from cphmd.core.patching import patch_system

    config = config_to_patch(config_path)
    result_dir = patch_system(config)
    logger.info("Patching complete: %s", result_dir)


def _run_alf(config_path: Path) -> None:
    """Run the ALF step.

    Ensures mpi4py initializes MPI before pyCHARMM's C library loads.
    """
    from mpi4py import MPI  # noqa: F401 — must precede pyCHARMM

    from cphmd.core.alf_runner import run_alf_simulation

    config = config_to_alf(config_path)
    run_alf_simulation(config)
    logger.info("ALF simulation complete")
