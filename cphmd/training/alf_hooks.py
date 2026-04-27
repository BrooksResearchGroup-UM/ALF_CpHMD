from __future__ import annotations

from dataclasses import dataclass
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
    generate_population_plots: bool = False
    generate_g_profiles_2d: bool = False
    generate_g_profiles_3d: bool = False

    def __post_init__(self) -> None:
        if self.cycle_every_segments <= 0:
            raise ValueError("cycle_every_segments must be positive")
        if self.end_cycle <= 0:
            raise ValueError("end_cycle must be positive")
        if self.cache_segments <= 0:
            raise ValueError("cache_segments must be positive")


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

    def on_system_loaded(self, ctx, state: LoopState | None = None) -> None:
        if not getattr(ctx, "ph_enabled", False):
            return
        native = self._native_block()
        native.set_ph(_ph_for_state(ctx, state))
        native.sync_state()

    def before_segment(self, state: LoopState) -> LoopState | None:
        return None

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> LoopState | None:
        self.cache = self.cache.append(state.segment_idx, lambda_matrix, bias_matrix)
        return None

    def should_trigger_cycle(self, state: LoopState) -> bool:
        return (
            state.segment_idx > 0
            and state.segment_idx % self.config.cycle_every_segments == 0
        )

    def run_cycle(self, state: LoopState) -> BiasSnapshot:
        snapshot = self.cycle_runner.run_cycle(state=state, cache=self.cache)
        self.last_snapshot = snapshot
        self._generate_cycle_plots(state)
        return snapshot

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

    def _native_block(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block

    def _generate_cycle_plots(self, state: LoopState) -> None:
        if not (
            self.config.generate_dashboard_plots
            or self.config.generate_population_plots
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
