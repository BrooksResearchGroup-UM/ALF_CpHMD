from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.simulation.context import LoopState
from cphmd.training.bias_snapshot import BiasSnapshot
from cphmd.training.segment_cache import SegmentCache


@dataclass(frozen=True)
class ALFTrainingConfig:
    cycle_every_segments: int
    end_cycle: int
    cache_segments: int
    generate_dashboard_plots: bool = True
    generate_population_plots: bool = True
    generate_hh_plots: bool = True
    generate_g_profiles_2d: bool = False
    generate_g_profiles_3d: bool = False
    phase1_repeats: int | None = None
    phase2_repeats: int | None = None
    phase3_repeats: int | None = None
    phase1_cycles: int | None = None
    phase2_cycles: int | None = None

    def __post_init__(self) -> None:
        if self.cycle_every_segments <= 0:
            raise ValueError("cycle_every_segments must be positive")
        if self.end_cycle <= 0:
            raise ValueError("end_cycle must be positive")
        if self.cache_segments <= 0:
            raise ValueError("cache_segments must be positive")
        for name, value in (
            ("phase1_repeats", self.phase1_repeats),
            ("phase2_repeats", self.phase2_repeats),
            ("phase3_repeats", self.phase3_repeats),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("phase1_cycles", self.phase1_cycles),
            ("phase2_cycles", self.phase2_cycles),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def repeats_for_phase(self, phase: int) -> int:
        value = {
            1: self.phase1_repeats,
            2: self.phase2_repeats,
            3: self.phase3_repeats,
        }.get(int(phase), self.phase3_repeats)
        return int(value if value is not None else self.cycle_every_segments)


class ALFHooks:
    uses_training_sidecars = True

    def __init__(
        self,
        config: ALFTrainingConfig,
        *,
        nsubs: tuple[int, ...],
        cycle_runner: Any,
        native_block: Any | None = None,
        replica_ph_values: tuple[float, ...] = (),
        cache: SegmentCache | None = None,
        work_dir: Path | str | None = None,
    ):
        self.config = config
        self.nsubs = tuple(nsubs)
        self.cycle_runner = cycle_runner
        self.native_block = native_block
        self.replica_ph_values = tuple(replica_ph_values)
        self.cache = cache or SegmentCache(max_segments=config.cache_segments)
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.last_snapshot: BiasSnapshot | None = None
        self._segments_since_last_cycle = 0

    def on_system_loaded(self, ctx, state: LoopState | None = None) -> None:
        if not getattr(ctx, "ph_enabled", False):
            self._initialize_biases_if_fresh(state)
            return
        native = self._native_block()
        native.set_ph(_ph_for_state(ctx, state))
        native.sync_state()
        self._initialize_biases_if_fresh(state)

    def before_segment(self, state: LoopState) -> LoopState | None:
        return None

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> LoopState | None:
        self.cache = self.cache.append(state.segment_idx, lambda_matrix, bias_matrix)
        self._segments_since_last_cycle += 1
        return None

    def should_trigger_cycle(self, state: LoopState) -> bool:
        if state.segment_idx <= 0:
            return False
        repeats = self.config.repeats_for_phase(state.phase)
        if self._segments_since_last_cycle > 0:
            return self._segments_since_last_cycle >= repeats
        return state.segment_idx % repeats == 0

    def will_trigger_cycle_after_next_segment(self, state: LoopState) -> bool:
        next_state = state.advance_segment()
        if next_state.segment_idx <= 0:
            return False
        repeats = self.config.repeats_for_phase(next_state.phase)
        if self._segments_since_last_cycle > 0:
            return self._segments_since_last_cycle + 1 >= repeats
        return next_state.segment_idx % repeats == 0

    def run_cycle(self, state: LoopState) -> BiasSnapshot:
        snapshot = self.cycle_runner.run_cycle(state=state, cache=self.cache)
        self._segments_since_last_cycle = 0
        self.last_snapshot = snapshot
        self._generate_cycle_plots(state)
        return snapshot

    def after_cycle_result(
        self,
        state: LoopState,
        snapshot: BiasSnapshot,
    ) -> LoopState | None:
        phase = self._analyzer_phase(default=state.phase)
        if phase == state.phase and not self._analyzer_stop_requested():
            phase = self._scheduled_phase(state.cycle_idx, state.phase)
        stop_requested = self._analyzer_stop_requested()
        if phase == state.phase and not stop_requested:
            return None
        return replace(state, phase=phase, stop_requested=state.stop_requested or stop_requested)

    def after_rex_swap(
        self,
        state: LoopState,
        *,
        partner_rank: int | None,
        accepted: bool,
    ) -> None:
        return None

    def is_done(self, state: LoopState) -> bool:
        return state.cycle_idx >= self.config.end_cycle

    def _scheduled_phase(self, completed_cycles: int, current_phase: int) -> int:
        phase1_cycles = self.config.phase1_cycles
        if phase1_cycles is None:
            return current_phase
        if completed_cycles < phase1_cycles:
            return 1

        phase2_cycles = self.config.phase2_cycles
        if phase2_cycles is None:
            return 2
        if completed_cycles < phase1_cycles + phase2_cycles:
            return 2
        return 3

    def _initialize_biases_if_fresh(self, state: LoopState | None) -> None:
        if state is not None and (state.segment_idx > 0 or state.cycle_idx > 0):
            return
        initializer = getattr(self.cycle_runner, "initialize_biases", None)
        if initializer is None:
            return
        snapshot = initializer()
        if snapshot is not None:
            self.last_snapshot = snapshot

    def _analyzer_phase(self, *, default: int) -> int:
        analyzer = getattr(self.cycle_runner, "analyzer", None)
        phase = getattr(analyzer, "last_phase", None)
        return default if phase is None else int(phase)

    def _analyzer_stop_requested(self) -> bool:
        analyzer = getattr(self.cycle_runner, "analyzer", None)
        return bool(getattr(analyzer, "last_stop_requested", False))

    def _native_block(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block

    def _generate_cycle_plots(self, state: LoopState) -> None:
        if not (
            self.config.generate_dashboard_plots
            or self.config.generate_population_plots
            or self.config.generate_hh_plots
            or self.config.generate_g_profiles_2d
            or self.config.generate_g_profiles_3d
        ):
            return
        if not self._is_plot_rank():
            return
        work_dir = self._plot_work_dir()
        if work_dir is None:
            return

        analysis_idx = state.cycle_idx + 1
        plots_dir = work_dir / "plots"
        if self.config.generate_dashboard_plots:
            try:
                from cphmd.analysis.alf_dashboard import generate_alf_dashboard

                generate_alf_dashboard(
                    work_dir,
                    max_run=analysis_idx,
                    output_dir=plots_dir,
                    nsubs=self.nsubs,
                    title=work_dir.name,
                )
            except Exception as exc:
                print(f"Warning: ALF dashboard plot failed: {exc}")

        if self.config.generate_population_plots:
            try:
                from cphmd.analysis.population_convergence import generate_population_plots

                generate_population_plots(
                    input_folder=work_dir,
                    max_run=analysis_idx,
                    output_dir=plots_dir,
                    nsubs=list(self.nsubs),
                )
            except Exception as exc:
                print(f"Warning: population convergence plots failed: {exc}")

        if self.config.generate_hh_plots:
            try:
                analyzer = getattr(self.cycle_runner, "analyzer", None)
                generate_hh_plots = getattr(analyzer, "generate_hh_plots", None)
                if callable(generate_hh_plots):
                    generate_hh_plots(
                        analysis_idx=analysis_idx,
                        phase=state.phase,
                        output_dir=plots_dir,
                        nsubs=self.nsubs,
                    )
            except Exception as exc:
                print(f"Warning: HH pH plots failed: {exc}")

        if self.config.generate_g_profiles_2d or self.config.generate_g_profiles_3d:
            try:
                from cphmd.analysis.wham_profiles import plot_wham_profiles

                analysis_dir = work_dir / f"analysis{analysis_idx}"
                plot_wham_profiles(
                    analysis_dir=analysis_dir,
                    nsubs=list(self.nsubs),
                    msprof=self._plot_msprof(),
                    main_plots_dir=plots_dir,
                    include_1d=False,
                    include_2d=self.config.generate_g_profiles_2d,
                    include_cross=self.config.generate_g_profiles_3d,
                )
            except Exception as exc:
                print(f"Warning: WHAM profile plots failed: {exc}")

    def _is_plot_rank(self) -> bool:
        analyzer = getattr(self.cycle_runner, "analyzer", None)
        ctx = getattr(analyzer, "ctx", None)
        return int(getattr(ctx, "rank", 0)) == 0

    def _plot_work_dir(self) -> Path | None:
        if self.work_dir is not None:
            return self.work_dir
        analyzer = getattr(self.cycle_runner, "analyzer", None)
        work_dir = getattr(analyzer, "work_dir", None)
        return Path(work_dir) if work_dir is not None else None

    def _plot_msprof(self) -> int:
        analyzer = getattr(self.cycle_runner, "analyzer", None)
        alf_info = getattr(analyzer, "alf_info", {})
        ntersite = alf_info.get("ntersite") if isinstance(alf_info, dict) else None
        if ntersite is not None and len(ntersite) >= 2:
            return int(ntersite[1])
        return 1 if self.config.generate_g_profiles_3d else 0


def _ph_for_state(ctx, state: LoopState | None) -> float:
    values = tuple(getattr(ctx, "replica_ph_values", ()) or ())
    if not values:
        return float(getattr(ctx, "ph", 7.0))
    current = state or LoopState(replica_label=getattr(ctx, "replica_label", 0))
    label = current.replica_label
    if label is None:
        label = getattr(ctx, "replica_label", 0)
    return float(values[int(label)])


def _ph_for_state_from_values(state: LoopState, values: tuple[float, ...]) -> float:
    if not values:
        return 7.0
    label = state.replica_label if state.replica_label is not None else 0
    return float(values[int(label)])
