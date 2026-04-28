"""Compatibility ALF configuration for the native training/runtime boundary."""

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

PhaseType = Literal[1, 2, 3]
AnalysisMethod = Literal["wham", "lmalf", "hybrid", "nonlinear"]
ConvergenceMode = Literal["population", "rmsd"]
PrepFormat = Literal["default", "legacy", "auto"]
RestrainType = Literal["SCAT", "NOE", "none"]
ElecType = Literal["pmeex", "pmeon", "pmenn", "fshift", "fswitch"]
VdwType = Literal["vswitch", "vfswitch"]

_ALF_CONFIG_ALIASES = {"pH": "ph"}


def _warn_deprecated_alias(alias: str, canonical: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{alias} is deprecated; use {canonical} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _init_dataclass_with_aliases(
    instance: object,
    cls: type,
    values: dict[str, Any],
    aliases: dict[str, str],
) -> None:
    values = dict(values)
    for alias, canonical in aliases.items():
        if alias not in values:
            continue
        if canonical in values:
            raise TypeError(f"Cannot pass both {alias!r} and {canonical!r}.")
        _warn_deprecated_alias(alias, canonical, stacklevel=4)
        values[canonical] = values.pop(alias)

    for data_field in fields(cls):
        if not data_field.init:
            continue
        name = data_field.name
        if name in values:
            value = values.pop(name)
        elif data_field.default is not MISSING:
            value = data_field.default
        elif data_field.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = data_field.default_factory()  # type: ignore[misc]
        else:
            raise TypeError(f"Missing required argument: {name!r}.")
        object.__setattr__(instance, name, value)

    if values:
        unexpected = next(iter(values))
        raise TypeError(f"Unexpected argument: {unexpected!r}.")


def _deprecated_getattr(instance: object, name: str, aliases: dict[str, str]) -> Any:
    if name in aliases:
        canonical = aliases[name]
        _warn_deprecated_alias(name, canonical, stacklevel=3)
        return getattr(instance, canonical)
    raise AttributeError(f"{type(instance).__name__!r} object has no attribute {name!r}")


def _deprecated_setattr(instance: object, name: str, value: Any, aliases: dict[str, str]) -> None:
    if name in aliases:
        canonical = aliases[name]
        _warn_deprecated_alias(name, canonical, stacklevel=3)
        object.__setattr__(instance, canonical, value)
        return
    object.__setattr__(instance, name, value)


def _dedupe_extra_files(paths: list[str | Path]) -> list[str | Path]:
    """Return paths with duplicate filesystem targets removed."""
    seen: set[Path] = set()
    result: list[str | Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition criteria."""

    lambda_threshold: float = 0.8
    spread_1to2: float = 0.5
    spread_2to3: float = 0.2
    min_hits_1to2: int = 100
    min_hits_2to3: int = 1000
    min_transitions_1to2: int = 10
    min_transitions_2to3: int = 20
    pka_tolerance_1to2: float = 1.5
    pka_tolerance_2to3: float = 0.3
    min_phase1_runs: int = 20
    min_visited_1to2: int = 999
    min_visited_frac_1to2: float = 0.01
    min_multistate_runs_1to2: int = 3
    min_connectivity_2to3: float = 0.2
    min_state_fraction_2to3: float = 0.01
    strict_threshold_2to3: float = 0.97
    min_phase2_runs: int = 15
    ewbs_2to3: float = 0.10
    ewbs_2to3_window: int = 5
    ewbs_2to3_b: float | None = 0.20
    ewbs_2to3_c: float | None = 2.00
    ewbs_2to3_x: float | None = 1.00
    ewbs_2to3_s: float | None = 0.50
    max_pop_diff_1to2: float = 0.9
    max_stuck_phase2_runs: int = 15
    stuck_diff_threshold: float = 0.95
    max_phase_regressions: int = 2


@dataclass
class ALFReplicaExchangeConfig:
    """Compatibility configuration for native pH replica exchange."""

    enabled: bool = False
    exchange_interval_ps: float | None = None
    exchange_interval_steps: int | None = None
    exchange_freq: int | None = None
    exchange_every_segments: int | None = None

    def __post_init__(self) -> None:
        interval_values = {
            "exchange_interval_ps": self.exchange_interval_ps,
            "exchange_interval_steps": self.exchange_interval_steps,
            "exchange_freq": self.exchange_freq,
            "exchange_every_segments": self.exchange_every_segments,
        }
        present = [name for name, value in interval_values.items() if value is not None]
        if len(present) > 1:
            keys = ", ".join(f"replica_exchange.{name}" for name in present)
            raise ValueError(f"conflicting replica exchange interval keys: {keys}")
        if self.exchange_interval_ps is not None:
            self.exchange_interval_ps = float(self.exchange_interval_ps)
            if self.exchange_interval_ps <= 0:
                raise ValueError("replica_exchange.exchange_interval_ps must be positive")
        if self.exchange_interval_steps is not None:
            self.exchange_interval_steps = int(self.exchange_interval_steps)
            if self.exchange_interval_steps <= 0:
                raise ValueError("replica_exchange.exchange_interval_steps must be positive")
        if self.exchange_freq is not None:
            self.exchange_freq = int(self.exchange_freq)
            if self.exchange_freq <= 0:
                raise ValueError("replica_exchange.exchange_freq must be positive")
        if self.exchange_every_segments is not None:
            self.exchange_every_segments = int(self.exchange_every_segments)
            if self.exchange_every_segments <= 0:
                raise ValueError("replica_exchange.exchange_every_segments must be positive")


@dataclass(init=False)
class ALFConfig:
    """Configuration for ALF simulation."""

    input_folder: str | Path
    toppar_dir: str | Path = "toppar"
    temperature: float = 298.15
    ph: bool = False
    hmr: bool | None = None
    start: int = 1
    end: int = 20
    phase: PhaseType = 1
    nreps: int | None = None
    restrains: RestrainType = "SCAT"
    restrain_hydrogens: bool = False
    scat_force_constant: float = 300.0
    bias_type: str | None = None
    no_b_bias: bool = False
    no_c_bias: bool = False
    no_x_bias: bool = False
    no_s_bias: bool = False
    no_t_bias: bool = True
    no_u_bias: bool = True
    auto_phase_switch: bool = False
    auto_stop: bool = False
    phase_transition: PhaseTransitionConfig = field(default_factory=PhaseTransitionConfig)
    convergence_mode: ConvergenceMode = "population"
    cleanup_old_analysis: bool = False
    generate_hh_plots: bool = True
    generate_dashboard_plots: bool = True
    generate_population_plots: bool = False
    generate_g_profiles_2d: bool = False
    generate_g_profiles_3d: bool = False
    cent_ncres: int | bool = False
    use_presets: bool = False
    bias_guess: bool = False
    debug: bool = False
    coupling: Literal[0, 1, 2] = 0
    coupling_profile: bool | None = None
    min_xs_coverage_runs: int = 3
    phase1_xs_cutoff: float = 2.0
    phase1_cutoffs: dict | None = None
    phase2_cutoffs: dict | None = None
    phase3_cutoffs: dict | None = None
    phase1_repeats: int | None = None
    phase2_repeats: int | None = None
    phase3_repeats: int | None = None
    phase1_iteration_ps: float | None = None
    phase2_iteration_ps: float | None = None
    phase3_iteration_ps: float | None = None
    phase1_iteration_steps: int | None = None
    phase2_iteration_steps: int | None = None
    phase3_iteration_steps: int | None = None
    phase1_cycles: int | None = None
    phase2_cycles: int | None = None
    phase1_iterations: int | None = None
    phase2_iterations: int | None = None
    analysis_window: int | list[int] | None = None
    analysis_skip: int | list[int] = 1
    fnex: float = 5.5
    chi_offset: float | None = None
    omega_decay: float | None = None
    chi_offset_t: float = 0.012
    chi_offset_u: float = 0.012
    lambda_mass: float | None = None
    lambda_fbeta: float | None = None
    g_imp_bins: int | list[int] | None = None
    cutlsum: float = 0.8
    endpoint_weight: float | list[float] = 100.0
    endpoint_decay: float | list[float] = 2.0
    use_gshift: bool = True
    analysis_method: AnalysisMethod = "wham"
    lmalf_max_iter: int = 0
    lmalf_tolerance: float = 0.0
    cutnb: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType | None = None
    gscale: float | None = None
    topology_files: list[str] = field(
        default_factory=lambda: [
            "top_all36_prot.rtf",
            "par_all36m_prot.prm",
            "top_all36_na.rtf",
            "par_all36_na.prm",
            "toppar_water_ions.str",
            "top_all36_cgenff.rtf",
            "par_all36_cgenff.prm",
            "my_files/titratable_residues.str",
            "my_files/nucleic_c36.str",
        ]
    )
    extra_files: list[str | Path] = field(default_factory=list)
    replica_exchange: ALFReplicaExchangeConfig | bool | dict[str, Any] | None = None
    prep_format: PrepFormat = "auto"
    legacy_setup_script: str | None = None
    legacy_auto_convert: bool = True
    legacy_convert_dir: str | Path | None = None
    legacy_force_convert: bool = False
    legacy_replace_toppar: bool = False

    def __init__(self, input_folder: str | Path, **kwargs: Any) -> None:
        if "input_folder" in kwargs:
            raise TypeError("ALFConfig got multiple values for 'input_folder'.")
        kwargs["input_folder"] = input_folder
        _init_dataclass_with_aliases(self, ALFConfig, kwargs, _ALF_CONFIG_ALIASES)
        self.__post_init__()

    def __getattr__(self, name: str) -> Any:
        return _deprecated_getattr(self, name, _ALF_CONFIG_ALIASES)

    def __setattr__(self, name: str, value: Any) -> None:
        _deprecated_setattr(self, name, value, _ALF_CONFIG_ALIASES)

    def __post_init__(self) -> None:
        self.input_folder = Path(self.input_folder).resolve()
        self.toppar_dir = Path(self.toppar_dir)

        prep_dir = self.input_folder / "prep"

        if self.prep_format == "auto":
            has_patches = (prep_dir / "patches.dat").exists()
            has_alf_info = (prep_dir / "alf_info.py").exists()
            if has_patches:
                self.prep_format = "default"
            elif has_alf_info:
                self.prep_format = "legacy"
            else:
                raise FileNotFoundError(
                    f"Cannot detect prep format in {prep_dir}: "
                    f"expected patches.dat (default) or alf_info.py (legacy)"
                )

        use_legacy_defaults = self.prep_format == "legacy"
        if self.prep_format == "legacy" and self.legacy_auto_convert:
            self._convert_legacy_input()
            prep_dir = self.input_folder / "prep"

        if self.prep_format == "default":
            for rel_path in [
                "prep/system.psf",
                "prep/system.crd",
                "prep/patches.dat",
                "prep/box.dat",
                "prep/fft.dat",
            ]:
                path = self.input_folder / rel_path
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
        elif self.prep_format == "legacy":
            if not (prep_dir / "alf_info.py").exists():
                raise FileNotFoundError(f"Required file not found: {prep_dir / 'alf_info.py'}")
            if self.legacy_setup_script is None:
                self.legacy_setup_script = self._find_legacy_setup_script(prep_dir)

        if self.hmr is None:
            self.hmr = not use_legacy_defaults
        if self.vdw_type is None:
            self.vdw_type = "vfswitch" if use_legacy_defaults else "vswitch"
        if self.gscale is None:
            self.gscale = 0.1 if use_legacy_defaults else 10.0

        if self.lambda_mass is None:
            self.lambda_mass = 7.0 if self.hmr else 4.0
        if self.lambda_fbeta is None:
            self.lambda_fbeta = 7.0 if self.hmr else 10.0

        for phase in (1, 2):
            cycles_attr = f"phase{phase}_cycles"
            iterations_attr = f"phase{phase}_iterations"
            cycles = getattr(self, cycles_attr)
            iterations = getattr(self, iterations_attr)
            if cycles is not None and iterations is not None:
                raise ValueError(
                    f"alf.{cycles_attr} conflicts with alf.{iterations_attr}; "
                    f"use alf.{iterations_attr}"
                )
            if cycles is None and iterations is not None:
                setattr(self, cycles_attr, int(iterations))

        if self.bias_type is not None:
            bt = self.bias_type.lower()
            valid = {"bc", "bcx", "bcxs", "bcxst", "bcxstu"}
            if bt not in valid:
                raise ValueError(
                    f"Invalid bias_type={bt!r}. Must be one of: {', '.join(sorted(valid))}"
                )
            self.no_b_bias = "b" not in bt
            self.no_c_bias = "c" not in bt
            self.no_x_bias = "x" not in bt
            self.no_s_bias = "s" not in bt
            self.no_t_bias = "t" not in bt
            self.no_u_bias = "u" not in bt

        if self.no_x_bias:
            self.no_s_bias = True
        if self.no_x_bias or self.no_s_bias:
            self.no_t_bias = True
            self.no_u_bias = True

        if not self.no_t_bias or not self.no_u_bias:
            warnings.warn(
                "t/u bias terms are experimental and not yet validated. "
                "Use bias_type='bcxstu' only for testing.",
                stacklevel=2,
            )

        self.phase_transition = self._normalize_phase_transition(self.phase_transition)
        self.replica_exchange = self._normalize_replica_exchange(self.replica_exchange)

    def _normalize_phase_transition(
        self, value: PhaseTransitionConfig | dict[str, Any]
    ) -> PhaseTransitionConfig:
        if isinstance(value, PhaseTransitionConfig):
            return value
        if not isinstance(value, dict):
            raise TypeError("phase_transition must be a mapping or PhaseTransitionConfig")
        allowed = {field.name for field in fields(PhaseTransitionConfig)}
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            unexpected_str = ", ".join(unexpected)
            raise ValueError(f"Unsupported phase_transition keys: {unexpected_str}")
        return PhaseTransitionConfig(**{k: value[k] for k in value if k in allowed})

    def _normalize_replica_exchange(
        self,
        value: ALFReplicaExchangeConfig | bool | dict[str, Any] | None,
    ) -> ALFReplicaExchangeConfig:
        if value is None:
            return ALFReplicaExchangeConfig()
        if isinstance(value, ALFReplicaExchangeConfig):
            return value
        if isinstance(value, bool):
            return ALFReplicaExchangeConfig(enabled=value)
        if not isinstance(value, dict):
            raise TypeError(
                "replica_exchange must be None, bool, mapping, or ALFReplicaExchangeConfig"
            )

        allowed = {field.name for field in fields(ALFReplicaExchangeConfig)}
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            unexpected_str = ", ".join(unexpected)
            raise ValueError(f"Unsupported replica_exchange keys: {unexpected_str}")
        return ALFReplicaExchangeConfig(**{k: value[k] for k in value if k in allowed})

    def _convert_legacy_input(self) -> None:
        from cphmd.setup.legacy_convert import LegacyConvertConfig, convert_legacy_system

        result = convert_legacy_system(
            LegacyConvertConfig(
                input_folder=self.input_folder,
                output_folder=self.legacy_convert_dir,
                setup_script=self.legacy_setup_script,
                force=self.legacy_force_convert,
                ph_enabled=bool(self.ph),
                temperature=self.temperature,
                toppar_dir=self.toppar_dir,
                topology_files=self.topology_files,
                replace_legacy_toppar=self.legacy_replace_toppar,
                debug=self.debug,
            )
        )
        self.input_folder = result.output_folder.resolve()
        self.prep_format = "default"
        self.legacy_setup_script = result.setup_script
        self.extra_files = _dedupe_extra_files([*self.extra_files, *result.extra_files])

    @property
    def ntriangle(self) -> int:
        return (
            1
            + (0 if self.no_x_bias else 2)
            + (0 if self.no_s_bias else 2)
            + (0 if self.no_t_bias else 2)
            + (0 if self.no_u_bias else 2)
        )

    @staticmethod
    def resolve_g_imp_bins(
        g_imp_bins: int | list[int] | None, phase: int
    ) -> int | None:
        if g_imp_bins is None:
            return None
        if isinstance(g_imp_bins, int):
            return g_imp_bins
        return g_imp_bins[phase - 1]

    @staticmethod
    def resolve_endpoint_weight(
        endpoint_weight: float | list[float], phase: int
    ) -> float:
        if isinstance(endpoint_weight, (int, float)):
            return float(endpoint_weight)
        return float(endpoint_weight[phase - 1])

    @staticmethod
    def _find_legacy_setup_script(prep_dir: Path) -> str:
        exclude = {"lpsites.inp", "toppar.str"}
        candidates = [f.name for f in prep_dir.glob("*.inp") if f.name not in exclude]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No .inp setup script found in {prep_dir}. "
                f"Set legacy_setup_script in ALFConfig."
            )
        raise FileNotFoundError(
            f"Multiple .inp files in {prep_dir}: {candidates}. "
            f"Set legacy_setup_script in ALFConfig to specify which one."
        )


__all__ = [
    "ALFConfig",
    "ALFReplicaExchangeConfig",
    "AnalysisMethod",
    "ConvergenceMode",
    "ElecType",
    "PhaseTransitionConfig",
    "PhaseType",
    "PrepFormat",
    "RestrainType",
    "VdwType",
]
