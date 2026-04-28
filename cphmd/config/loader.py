"""YAML configuration loading and merging for CpHMD workflows.

Config chain priority (last wins):
    package defaults → local cphmd_config.yaml → --config flag → CLI flags

The loader reads YAML files, merges them following the priority chain,
and converts the merged dict into the appropriate dataclass
(ALFConfig, PatchConfig, SolvationConfig).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from cphmd import TOPPAR_DIR
from cphmd.simulation.backends import (
    AnalysisBackend,
    DomdecConfig,
    DynamicsBackend,
    parse_analysis_backend,
    parse_dynamics_backend,
)

if TYPE_CHECKING:
    from cphmd.core.patching import PatchConfig
    from cphmd.setup.prepare_pdb import PreparePDBConfig
    from cphmd.setup.solvate import SolvationConfig
    from cphmd.training.config import ALFConfig

logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_LOCAL_CONFIG_NAME = "cphmd_config.yaml"
_DEFAULT_TOPOLOGY_FILES = (
    "top_all36_prot.rtf",
    "par_all36m_prot.prm",
    "top_all36_na.rtf",
    "par_all36_na.prm",
    "toppar_water_ions.str",
    "top_all36_cgenff.rtf",
    "par_all36_cgenff.prm",
    "my_files/titratable_residues.str",
    "my_files/nucleic_c36.str",
)
_ALF_RUNTIME_ONLY_KEYS = {
    "checkpoint_every_segments",
    "checkpoint_interval_ps",
    "checkpoint_interval_steps",
    "coordinate_save_interval_ps",
    "coordinate_save_steps",
    "lambda_save_interval_ps",
    "lambda_save_steps",
    "md_block_ps",
    "md_block_steps",
    "nsteps",
    "nsteps_per_segment",
    "nsavc",
    "nsavl",
    "pH_values",
    "ph_end",
    "ph_start",
    "ph_values",
    "time_step_ps",
    "timestep_fs",
}


@dataclass(frozen=True)
class ArchiveConfig:
    lambda_precision: str = "full"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class NativeRuntimeConfig:
    """Minimal native runtime configuration used by Phase-5 CLI commands."""

    config_path: Path
    raw: dict[str, Any]
    run_dir: Path
    input_folder: Path
    nreps: int
    master_seed: int
    start: int
    end: int
    phase: int
    cent_ncres: int | bool
    temperature: float
    nsteps_per_segment: int
    nsavl: int
    nsavc: int
    time_step_ps: float
    checkpoint_every_segments: int
    archive: ArchiveConfig
    logging: LoggingConfig
    dynamics_backend: DynamicsBackend = DynamicsBackend.BLADE
    analysis_backend: AnalysisBackend = AnalysisBackend.CUDA_WHAM
    domdec: DomdecConfig = DomdecConfig()
    ph: bool = False
    ph_values: tuple[float, ...] = ()
    replica_exchange: Any = None
    walltime_safety_factor: float = 2.0
    production: Any | None = None
    toppar_dir: Path = TOPPAR_DIR
    topology_files: tuple[str, ...] = _DEFAULT_TOPOLOGY_FILES
    extra_files: tuple[Path, ...] = ()
    cutnb: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: str = "pmeex"
    vdw_type: str = "vswitch"
    fnex: float = 5.5


def load_config(path: str | Path) -> NativeRuntimeConfig:
    """Load the native runtime config used by ``cphmd init/run/status``.

    This loader intentionally avoids constructing legacy runner dataclasses so
    CLI startup can validate config without importing pyCHARMM-bound code.
    """

    config_path = Path(path).resolve()
    raw = load_yaml_config(config_path)
    _reject_user_facing_use_blade(raw)

    alf = _optional_mapping(raw, "alf")
    simulation = _optional_mapping(raw, "simulation")
    native = dict(raw.get("native", {}) or {})
    archive = dict(raw.get("archive", {}) or {})
    logging_cfg = dict(raw.get("logging", {}) or {})
    if "production" in raw:
        if raw["production"] is None:
            production_section = {}
        elif isinstance(raw["production"], dict):
            production_section = dict(raw["production"])
        else:
            raise ValueError("production must be a mapping")
    else:
        production_section = None
    runtime = _runtime_section(
        raw=raw,
        alf=alf,
        simulation=simulation,
        production_section=production_section,
    )

    if "master_seed" in raw:
        master_seed = raw["master_seed"]
    elif "master_seed" in runtime:
        master_seed = runtime["master_seed"]
    else:
        raise ValueError("config must define required key 'master_seed'")

    try:
        master_seed = int(master_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("master_seed must be an integer") from exc

    input_folder = _resolve_config_path(
        config_path.parent,
        runtime.get("input_folder") or raw.get("input_folder") or raw.get("run_dir") or ".",
    )
    run_dir = _resolve_config_path(config_path.parent, raw.get("run_dir") or input_folder)

    nreps = int(runtime.get("nreps") or raw.get("nreps") or 1)
    if nreps < 1:
        raise ValueError("nreps must be >= 1")

    if production_section is not None:
        lambda_precision = archive.get(
            "lambda_precision",
            production_section.get("lambda_precision", "shrinker"),
        )
    else:
        lambda_precision = archive.get("lambda_precision", "full")
    if lambda_precision not in {"full", "shrinker"}:
        raise ValueError("archive.lambda_precision must be 'full' or 'shrinker'")
    time_step_ps = _time_step_ps(runtime)
    nsteps_per_segment = _nsteps_per_segment(runtime, time_step_ps=time_step_ps)
    nsavl = _steps_from_interval(
        runtime,
        key="lambda_save_interval_ps",
        step_key="lambda_save_steps",
        legacy_key="nsavl",
        default=10,
        time_step_ps=time_step_ps,
    )
    nsavc = _steps_from_interval(
        runtime,
        key="coordinate_save_interval_ps",
        step_key="coordinate_save_steps",
        legacy_key="nsavc",
        default=100,
        time_step_ps=time_step_ps,
    )
    checkpoint_every_segments = _segments_from_interval(
        runtime,
        key="checkpoint_interval_ps",
        step_key="checkpoint_interval_steps",
        legacy_key="checkpoint_every_segments",
        default=1,
        nsteps_per_segment=nsteps_per_segment,
        time_step_ps=time_step_ps,
    )
    production = (
        _parse_production_config(
            config_path,
            production_section,
            lambda_precision,
            runtime=runtime,
            nsteps_per_segment=nsteps_per_segment,
            time_step_ps=time_step_ps,
        )
        if production_section is not None
        else None
    )

    replica_exchange = runtime.get("replica_exchange")
    ph_enabled = _ph_enabled(runtime)
    ph_values = _ph_values(runtime, nreps)
    dynamics_backend = parse_dynamics_backend(native.get("dynamics_backend", "blade"))
    analysis_backend = parse_analysis_backend(native.get("analysis_backend", "cuda-wham"))
    domdec = DomdecConfig.from_mapping(native.get("domdec"))
    _validate_native_backend(
        dynamics_backend=dynamics_backend,
        production_enabled=production_section is not None,
        ph_enabled=ph_enabled,
    )
    extra_files = _dedupe_paths(
        [
            *(
                _resolve_config_path(config_path.parent, path)
                for path in runtime.get("extra_files", [])
            ),
            *_legacy_import_extra_files(input_folder),
        ]
    )

    return NativeRuntimeConfig(
        config_path=config_path,
        raw=raw,
        run_dir=run_dir,
        input_folder=input_folder,
        nreps=nreps,
        master_seed=master_seed,
        start=int(alf.get("start", 1)),
        end=int(alf.get("end", 200)),
        phase=int(alf.get("phase", 1)),
        cent_ncres=alf.get("cent_ncres", False),
        temperature=float(runtime.get("temperature", 298.15)),
        nsteps_per_segment=nsteps_per_segment,
        nsavl=nsavl,
        nsavc=nsavc,
        time_step_ps=time_step_ps,
        checkpoint_every_segments=checkpoint_every_segments,
        archive=ArchiveConfig(lambda_precision=lambda_precision),
        logging=LoggingConfig(level=str(logging_cfg.get("level", "INFO"))),
        dynamics_backend=dynamics_backend,
        analysis_backend=analysis_backend,
        domdec=domdec,
        ph=ph_enabled,
        ph_values=ph_values,
        replica_exchange=replica_exchange,
        walltime_safety_factor=float(raw.get("walltime_safety_factor", 2.0)),
        production=production,
        toppar_dir=_resolve_config_path(config_path.parent, runtime.get("toppar_dir", TOPPAR_DIR)),
        topology_files=tuple(runtime.get("topology_files", _DEFAULT_TOPOLOGY_FILES)),
        extra_files=tuple(extra_files),
        cutnb=float(runtime.get("cutnb", 14.0)),
        ctofnb=float(runtime.get("ctofnb", 12.0)),
        ctonnb=float(runtime.get("ctonnb", 10.0)),
        elec_type=str(runtime.get("elec_type", "pmeex")),
        vdw_type=str(runtime.get("vdw_type") or "vswitch"),
        fnex=float(runtime.get("fnex", 5.5)),
    )


def _optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def _legacy_import_extra_files(input_folder: Path) -> list[Path]:
    manifest_path = input_folder / "legacy_import.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid legacy import manifest: {manifest_path}") from exc

    extra_files: list[Path] = []
    for raw_path in manifest.get("extra_files", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = input_folder / path
        extra_files.append(path.resolve())
    return extra_files


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _runtime_section(
    *,
    raw: dict[str, Any],
    alf: dict[str, Any],
    simulation: dict[str, Any],
    production_section: dict[str, Any] | None,
) -> dict[str, Any]:
    if production_section is not None and "simulation" in raw:
        return simulation
    return alf


def _parse_production_config(
    config_path: Path,
    section: dict[str, Any],
    lambda_precision: str,
    *,
    runtime: dict[str, Any],
    nsteps_per_segment: int,
    time_step_ps: float,
):
    from cphmd.training.production_hooks import ProductionConfig

    cfg = dict(section)
    duration_ns = cfg.pop("duration_ns", None)
    duration_steps = _pop_step_alias(cfg, "duration_steps", "nsteps")
    if duration_ns is not None and duration_steps is not None:
        raise ValueError(
            "production.duration_ns conflicts with production.duration_steps/nsteps"
        )
    legacy_keys = {"n_chunks", "segments_per_chunk"} & set(cfg)
    if legacy_keys:
        legacy_str = ", ".join(f"production.{key}" for key in sorted(legacy_keys))
        raise ValueError(
            f"{legacy_str} are internal counters; use production.duration_ns or "
            "production.duration_steps"
        )
    if duration_ns is None and duration_steps is None:
        raise ValueError("production.duration_ns or production.duration_steps is required")
    if duration_ns is not None:
        total_segments = _segments_for_duration_ns(
            duration_ns,
            nsteps_per_segment=nsteps_per_segment,
            time_step_ps=time_step_ps,
        )
    else:
        total_segments = _segments_for_duration_steps(
            duration_steps,
            nsteps_per_segment=nsteps_per_segment,
        )
    cfg["n_chunks"] = 1
    cfg["segments_per_chunk"] = total_segments
    if cfg.get("bias_file") is not None:
        cfg["bias_file"] = _resolve_config_path(config_path.parent, cfg["bias_file"])
    cfg.setdefault("use_presets", cfg.get("bias_file") is None)
    if cfg.get("use_presets") and cfg.get("preset_config") is None:
        cfg["preset_config"] = _derive_preset_config(runtime)
    cfg.setdefault("temperature", float(runtime.get("temperature", 298.15)))
    cfg.setdefault("fnex", float(runtime.get("fnex", 5.5)))
    cfg["topology_files"] = [
        _resolve_config_path(config_path.parent, path) for path in cfg.get("topology_files", [])
    ]
    cfg["extra_files"] = [
        _resolve_config_path(config_path.parent, path) for path in cfg.get("extra_files", [])
    ]
    cfg["lambda_precision"] = lambda_precision
    return ProductionConfig(**cfg)


def _derive_preset_config(runtime: dict[str, Any]) -> str:
    elec = _preset_electrostatics(str(runtime.get("elec_type", "pmeex")))
    vdw = str(runtime.get("vdw_type") or "vswitch").strip().lower().replace("-", "_")
    restraint = _preset_restraint(str(runtime.get("restrains", "SCAT")))
    hydrogens = "h" if _bool_config(runtime.get("restrain_hydrogens", False)) else "nh"
    return f"{elec}_{vdw}_{restraint}_{hydrogens}"


def _preset_electrostatics(value: str) -> str:
    token = value.strip().lower().replace("-", "_")
    aliases = {
        "pmeex": "pme_ex",
        "pme_ex": "pme_ex",
        "pmeon": "pme_on",
        "pme_on": "pme_on",
        "pmenn": "pme_nn",
        "pme_nn": "pme_nn",
        "fshift": "fshift",
        "fswitch": "fswitch",
    }
    if token not in aliases:
        raise ValueError(f"cannot derive preset config from elec_type={value!r}")
    return aliases[token]


def _preset_restraint(value: str) -> str:
    token = value.strip().lower()
    aliases = {
        "scat": "sca",
        "sca": "sca",
        "noe": "noe",
    }
    if token not in aliases:
        raise ValueError(f"cannot derive preset config from restrains={value!r}")
    return aliases[token]


def _bool_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _time_step_ps(runtime: dict[str, Any]) -> float:
    if "time_step_ps" in runtime:
        value = float(runtime["time_step_ps"])
    elif "timestep_fs" in runtime:
        value = float(runtime["timestep_fs"]) / 1000.0
    else:
        value = 0.004 if runtime.get("hmr", True) else 0.002
    if value <= 0:
        raise ValueError("simulation timestep must be positive")
    return value


def _nsteps_per_segment(runtime: dict[str, Any], *, time_step_ps: float) -> int:
    _reject_conflicting_keys(
        runtime,
        "simulation MD block length",
        ("md_block_ps", "md_block_steps", "nsteps_per_segment", "nsteps"),
    )
    if "md_block_ps" in runtime:
        return _positive_int_from_ps_interval(
            runtime["md_block_ps"],
            time_step_ps=time_step_ps,
            name="simulation.md_block_ps",
        )
    if "md_block_steps" in runtime:
        return _positive_int(runtime["md_block_steps"], "simulation.md_block_steps")
    return int(runtime.get("nsteps_per_segment", runtime.get("nsteps", 1000)))


def _steps_from_interval(
    runtime: dict[str, Any],
    *,
    key: str,
    step_key: str,
    legacy_key: str,
    default: int,
    time_step_ps: float,
) -> int:
    _reject_conflicting_keys(runtime, f"simulation {key}", (key, step_key, legacy_key))
    if key in runtime:
        return _positive_int_from_ps_interval(
            runtime[key],
            time_step_ps=time_step_ps,
            name=f"simulation.{key}",
        )
    if step_key in runtime:
        return _positive_int(runtime[step_key], f"simulation.{step_key}")
    return int(runtime.get(legacy_key, default))


def _segments_from_interval(
    runtime: dict[str, Any],
    *,
    key: str,
    step_key: str,
    legacy_key: str,
    default: int,
    nsteps_per_segment: int,
    time_step_ps: float,
) -> int:
    _reject_conflicting_keys(runtime, f"simulation {key}", (key, step_key, legacy_key))
    if key in runtime:
        segment_ps = nsteps_per_segment * time_step_ps
        return _positive_int_ratio(
            float(runtime[key]),
            segment_ps,
            name=f"simulation.{key}",
            unit="ps",
        )
    if step_key in runtime:
        return _positive_int_ratio(
            float(runtime[step_key]),
            float(nsteps_per_segment),
            name=f"simulation.{step_key}",
            unit="steps",
        )
    return int(runtime.get(legacy_key, default))


def _segments_for_duration_ns(
    duration_ns: Any,
    *,
    nsteps_per_segment: int,
    time_step_ps: float,
) -> int:
    duration_ps = float(duration_ns) * 1000.0
    segment_ps = nsteps_per_segment * time_step_ps
    return _positive_int_ratio(
        duration_ps,
        segment_ps,
        name="production.duration_ns",
        unit="ns",
        value_for_message=float(duration_ns),
    )


def _segments_for_duration_steps(
    duration_steps: Any,
    *,
    nsteps_per_segment: int,
) -> int:
    return _positive_int_ratio(
        float(duration_steps),
        float(nsteps_per_segment),
        name="production.duration_steps",
        unit="steps",
    )


def _positive_int_from_ps_interval(value: Any, *, time_step_ps: float, name: str) -> int:
    return _positive_int_ratio(float(value), time_step_ps, name=name, unit="ps")


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_int_ratio(
    numerator: float,
    denominator: float,
    *,
    name: str,
    unit: str,
    value_for_message: float | None = None,
) -> int:
    if numerator <= 0:
        raise ValueError(f"{name} must be positive")
    ratio = numerator / denominator
    rounded = round(ratio)
    if rounded <= 0 or abs(ratio - rounded) > 1e-9:
        shown_value = numerator if value_for_message is None else value_for_message
        raise ValueError(
            f"{name}={shown_value:g} {unit} is not divisible by the MD step/block size"
        )
    return int(rounded)


def _reject_conflicting_keys(
    mapping: dict[str, Any],
    label: str,
    keys: tuple[str, ...],
) -> None:
    present = [key for key in keys if key in mapping]
    if len(present) > 1:
        qualified = ", ".join(f"simulation.{key}" for key in present)
        raise ValueError(f"conflicting {label} keys: {qualified}")


def _pop_step_alias(mapping: dict[str, Any], primary: str, alias: str) -> Any | None:
    primary_present = primary in mapping
    alias_present = alias in mapping
    if primary_present and alias_present:
        raise ValueError(f"production.{primary} conflicts with production.{alias}")
    if primary_present:
        return mapping.pop(primary)
    if alias_present:
        return mapping.pop(alias)
    return None


def _resolve_config_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _reject_user_facing_use_blade(raw: dict[str, Any]) -> None:
    if "use_blade" in raw:
        raise ValueError(
            "use_blade is not accepted in YAML; use native.dynamics_backend instead"
        )
    for section_name, section in raw.items():
        if isinstance(section, dict) and "use_blade" in section:
            raise ValueError(
                f"{section_name}.use_blade is not accepted in YAML; use "
                "native.dynamics_backend instead"
            )


def _ph_values(alf: dict[str, Any], nreps: int) -> tuple[float, ...]:
    if "ph_values" in alf:
        return tuple(float(value) for value in alf["ph_values"])
    if "pH_values" in alf:
        return tuple(float(value) for value in alf["pH_values"])
    if "ph_start" in alf and "ph_end" in alf and nreps > 1:
        start = float(alf["ph_start"])
        end = float(alf["ph_end"])
        step = (end - start) / (nreps - 1)
        return tuple(start + step * idx for idx in range(nreps))
    if "ph" in alf and isinstance(alf["ph"], (int, float)) and not isinstance(alf["ph"], bool):
        return (float(alf["ph"]),)
    if "pH" in alf and isinstance(alf["pH"], (int, float)) and not isinstance(alf["pH"], bool):
        return (float(alf["pH"]),)
    return tuple(float(7.0) for _ in range(nreps))


def _ph_enabled(alf: dict[str, Any]) -> bool:
    if "ph" in alf:
        value = alf["ph"]
    elif "pH" in alf:
        value = alf["pH"]
    else:
        return False
    if isinstance(value, bool):
        return value
    return isinstance(value, (int, float))


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


def _validate_native_backend(
    *,
    dynamics_backend: DynamicsBackend,
    production_enabled: bool,
    ph_enabled: bool,
) -> None:
    if production_enabled and ph_enabled and dynamics_backend is DynamicsBackend.BLADE:
        raise ValueError(
            "native.dynamics_backend=blade is incompatible with production CpHMD "
            "BLOCK softcore/PMEL state; use native.dynamics_backend: domdec-gpu"
        )


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
    cfg = _resolve_config_chain("alf", config_path, cli_overrides)
    return alf_config_from_mapping(cfg)


def alf_config_from_mapping(
    mapping: dict[str, Any], *, include_defaults: bool = False
) -> "ALFConfig":
    """Build an ALFConfig from an already-selected ``alf`` mapping."""
    from cphmd.training.config import ALFConfig

    cfg = dict(mapping)
    if include_defaults:
        cfg = merge_configs(_load_defaults("alf"), cfg)

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

    for key in _ALF_RUNTIME_ONLY_KEYS:
        cfg.pop(key, None)

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

    prepare_cfg = _resolve_config_chain("prepare", config_path, None)
    if "input_file" not in cfg and prepare_cfg.get("input_source"):
        output_dir = prepare_cfg.get("output_dir", "pdb")
        output_name = Path(prepare_cfg.get("output_name", "input")).with_suffix("").name
        cfg["input_file"] = str(Path(output_dir) / output_name)

    patch_cfg = _resolve_config_chain("patch", config_path, None)
    if "selected_residues" not in cfg and "selected_residues" in patch_cfg:
        cfg["selected_residues"] = patch_cfg["selected_residues"]
    if "ligand_patches" not in cfg and "ligand_patches" in patch_cfg:
        cfg["ligand_patches"] = patch_cfg["ligand_patches"]

    return SolvationConfig(**cfg)


def config_to_prepare_pdb(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> "PreparePDBConfig":
    """Build a PreparePDBConfig from the config chain."""
    from cphmd.setup.prepare_pdb import PreparePDBConfig

    cfg = _resolve_config_chain("prepare", config_path, cli_overrides)
    return PreparePDBConfig(**cfg)


def run_workflow(
    config_path: str | Path,
    step: str = "all",
) -> None:
    """Run one or more CpHMD workflow steps from a config file.

    Dispatches to the appropriate functions based on the step parameter.
    Steps are executed in order: build → prepare → solvate → patch → alf.

    Args:
        config_path: Path to the YAML config file.
        step: Which step to run. One of:
            "build"    — Create amino acid template structures
            "prepare"  — Prepare PSF/CRD/PDB from a PDB source
            "solvate"  — Solvate the system
            "patch"    — Apply CpHMD patches
            "alf"      — Run ALF simulation
            "all"      — Run all steps in order (skips steps without config)

    Raises:
        ValueError: If step is not recognized.
        FileNotFoundError: If config_path does not exist.
    """
    valid_steps = {"build", "prepare", "solvate", "patch", "alf", "all"}
    if step not in valid_steps:
        raise ValueError(f"Unknown step {step!r}. Must be one of: {valid_steps}")

    config_path = Path(config_path)
    full_config = load_yaml_config(config_path)

    steps = ["build", "prepare", "solvate", "patch", "alf"] if step == "all" else [step]

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
        elif current_step == "prepare":
            _run_prepare(config_path)
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


def _run_prepare(config_path: Path) -> None:
    """Run the PDB preparation step."""
    from cphmd.setup.prepare_pdb import prepare_pdb_system

    config = config_to_prepare_pdb(config_path)
    output_base = prepare_pdb_system(config)
    logger.info("Preparation complete: %s", output_base)


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
    """Run the ALF step through the native init/run command pair."""
    from cphmd.cli.init_cmd import run_init
    from cphmd.cli.run_cmd import run_simulation

    cfg = load_config(config_path)
    init_marker = cfg.run_dir / "state" / "initialized.json"
    if not init_marker.exists():
        run_init(config_path=config_path)
    run_simulation(config_path=config_path)
    logger.info("ALF simulation complete")
