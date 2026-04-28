from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

import numpy as np

from cphmd.simulation.archiver import Archiver
from cphmd.simulation.backends import DynamicsBackend
from cphmd.simulation.checkpoint import CheckpointManager
from cphmd.simulation.context import LoopHooks, LoopState, RunContext
from cphmd.simulation.walltime import WalltimeGuard, job_end_time_from_env

RunSegmentFn = Callable[..., tuple[np.ndarray, np.ndarray]]
logger = logging.getLogger(__name__)


def default_native_modules() -> tuple[ModuleType, ...]:
    from cphmd.native import block, dynamics, minimization, mpi, rex

    return block, dynamics, minimization, mpi, rex


class SimulationLoop:
    def __init__(
        self,
        ctx: RunContext,
        hooks: LoopHooks,
        *,
        checkpoint: CheckpointManager | None = None,
        archiver: Archiver | None = None,
        rex_driver=None,
        run_segment: RunSegmentFn | None = None,
        native_modules: Iterable[ModuleType] | None = None,
        native_dynamics: Any | None = None,
        native_minimizer: Any | None = None,
        bias_rebuilder: Any | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        self.ctx = ctx
        self.hooks = hooks
        modules = tuple(native_modules) if native_modules is not None else default_native_modules()
        self._native_modules = modules
        self.checkpoint = checkpoint or CheckpointManager(
            ctx,
            native_modules=modules,
            require_charmm_restart=True,
        )
        self.archiver = archiver or Archiver(ctx)
        self.rex_driver = rex_driver if rex_driver is not None else self._build_rex_driver()
        self._run_segment = run_segment or self._run_native_segment
        self._native_dynamics_override = native_dynamics
        self._native_minimizer_override = native_minimizer
        self._bias_rebuilder_override = bias_rebuilder
        self._time_fn = time_fn or time.time
        self._stop_requested = False
        self._segment_seconds: list[float] = []
        self._archive_seconds: list[float] = []
        self._rex_seconds: list[float] = []
        self._rex_hook_seconds: list[float] = []
        self._checkpoint_seconds: list[float] = []
        self._cycle_seconds: list[float] = []
        walltime_end = ctx.walltime_end_epoch
        if walltime_end is None:
            walltime_end = job_end_time_from_env()
        self._walltime_guard = WalltimeGuard(
            job_end_time=walltime_end,
            safety_factor=ctx.walltime_safety_factor,
        )
        if walltime_end is None:
            logger.warning("walltime guard disabled: SLURM_JOB_END_TIME unset")

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> LoopState:
        state, rng_state = self.checkpoint.resume_or_fresh()
        resumed = state.segment_idx > 0 or state.run_idx > 0 or state.cycle_idx > 0
        if self.ctx.rex_enabled and state.replica_label is None:
            state = state.with_initial_label(self.ctx.replica_label)
        on_system_loaded = getattr(self.hooks, "on_system_loaded", None)
        if on_system_loaded is not None:
            on_system_loaded(self.ctx, state)
        if resumed:
            self._restore_training_state(state)
        if not resumed:
            self._prepare_startup(state)
        else:
            self._initialize_native_dynamics()
            self._preflight_native_dynamics()
        on_native_ready = getattr(self.hooks, "on_native_ready", None)
        if on_native_ready is not None:
            on_native_ready(self.ctx, state)
        stop_token = self._install_signal_handlers()
        try:
            while not self.hooks.is_done(state):
                wrote_checkpoint = False
                if stop_token is not None and stop_token.stop_requested:
                    self.request_stop()
                if self._should_stop_before_next_segment(state):
                    state = state.with_stop_requested()
                    break
                before_segment = getattr(self.hooks, "before_segment", None)
                if before_segment is not None:
                    next_state = before_segment(state)
                    if next_state is not None:
                        state = next_state
                segment_started = self._time_fn()
                segment_seed = state.integrator_seed
                lambda_matrix, bias_matrix = self._run_segment(
                    nsteps=self.ctx.nsteps_per_segment,
                    nsavl=self.ctx.nsavl,
                    nsavc=self.ctx.nsavc,
                    timestep=self.ctx.time_step_ps,
                    temperature=self.ctx.temperature,
                    gpu_id=self.ctx.gpu_id,
                    dynamics_backend=self.ctx.dynamics_backend,
                    start=(not resumed and state.segment_idx == 0),
                    lambda_headers=self.ctx.lambda_headers,
                    iseed=segment_seed,
                    lambda_parquet_path=self.ctx.native_lambda_scratch_path(state.segment_idx),
                )
                if segment_seed is not None:
                    state = state.with_integrator_seed(None)
                self._walltime_guard.record_segment_duration(
                    self._record_duration(self._segment_seconds, segment_started)
                )
                archive_started = self._time_fn()
                self.archiver.write_segment(lambda_matrix, bias_matrix, state)
                self._record_duration(self._archive_seconds, archive_started)
                next_state = self.hooks.after_segment(state, lambda_matrix, bias_matrix)
                if next_state is not None:
                    state = next_state
                state = state.advance_segment()
                if self.rex_driver is not None and self.rex_driver.should_attempt(state):
                    previous_accepted = state.rex_accepted
                    rex_started = self._time_fn()
                    rex_result = self.rex_driver.attempt(state)
                    self._record_duration(self._rex_seconds, rex_started)
                    state = state.with_rex_result(
                        replica_label=rex_result.replica_label,
                        attempted=rex_result.attempted,
                        accepted=rex_result.accepted,
                    )
                    after_rex_swap = getattr(self.hooks, "after_rex_swap", None)
                    if after_rex_swap is not None:
                        rex_hook_started = self._time_fn()
                        after_rex_swap(
                            state,
                            partner_rank=_partner_rank(rex_result.exchange_result),
                            accepted=sum(state.rex_accepted) > sum(previous_accepted),
                        )
                        self._record_duration(self._rex_hook_seconds, rex_hook_started)
                if self._should_trigger_cycle(state):
                    cycle_started = self._time_fn()
                    snapshot = self.hooks.run_cycle(state)
                    self._walltime_guard.record_cycle_duration(
                        self._record_duration(self._cycle_seconds, cycle_started)
                    )
                    next_state = state.with_cycle_result()
                    after_cycle_result = getattr(self.hooks, "after_cycle_result", None)
                    if after_cycle_result is not None:
                        adjusted_state = after_cycle_result(next_state, snapshot)
                        if adjusted_state is not None:
                            next_state = adjusted_state
                    self.checkpoint.write_training_sidecars(
                        cache=getattr(self.hooks, "cache", None),
                        bias_snapshot=snapshot,
                        state=next_state,
                    )
                    state = next_state
                    checkpoint_started = self._time_fn()
                    rng_state = _with_production_rng_state(rng_state, state)
                    self.checkpoint.write(state, rng_state=rng_state)
                    self._write_status_summary(state)
                    self._walltime_guard.record_checkpoint_duration(
                        self._record_duration(self._checkpoint_seconds, checkpoint_started)
                    )
                    wrote_checkpoint = True
                if (
                    not wrote_checkpoint
                    and state.segment_idx % self.ctx.checkpoint_every_segments == 0
                ):
                    checkpoint_started = self._time_fn()
                    rng_state = _with_production_rng_state(rng_state, state)
                    self.checkpoint.write(state, rng_state=rng_state)
                    self._write_status_summary(state)
                    self._walltime_guard.record_checkpoint_duration(
                        self._record_duration(self._checkpoint_seconds, checkpoint_started)
                    )
        finally:
            self._restore_signal_handlers(stop_token)
        self.checkpoint.write_final(state, rng_state=_with_production_rng_state(rng_state, state))
        final_run_state = "stopped" if state.stop_requested else "completed"
        self._write_status_summary(state, run_state=final_run_state)
        self._write_timing_summary(state, run_state=final_run_state)
        return state

    def _build_rex_driver(self):
        if not self.ctx.rex_enabled:
            return None

        from cphmd.simulation.rex_driver import REXDriver

        return REXDriver(self.ctx)

    def _should_trigger_cycle(self, state: LoopState) -> bool:
        trigger = getattr(self.hooks, "should_trigger_cycle", None)
        if trigger is None:
            return False
        return bool(trigger(state))

    def _initialize_native_dynamics(self) -> None:
        native_dynamics = self._native_dynamics()
        if self.ctx.dynamics_backend is DynamicsBackend.BLADE:
            native_dynamics.use_blade(self.ctx.gpu_id)
            native_dynamics.enable_fast_routines()
            return
        if self.ctx.dynamics_backend.uses_domdec:
            native_dynamics.use_domdec(
                gpu=self.ctx.dynamics_backend is DynamicsBackend.DOMDEC_GPU,
                gpu_id=self.ctx.gpu_id,
                config=self.ctx.domdec,
            )

    def _preflight_native_dynamics(self) -> None:
        native_dynamics = self._native_dynamics()
        if self.ctx.dynamics_backend is DynamicsBackend.BLADE:
            preflight = getattr(native_dynamics, "preflight_blade_energy", None)
            if preflight is not None:
                preflight()
            return
        if self.ctx.dynamics_backend.uses_domdec:
            preflight = getattr(native_dynamics, "preflight_domdec_energy", None)
            if preflight is not None:
                preflight(gpu=self.ctx.dynamics_backend is DynamicsBackend.DOMDEC_GPU)

    def _minimize_startup(self, state: LoopState) -> bool:
        if self.ctx.startup_minimization_segments <= 0:
            return False
        minimizer = self._native_minimizer()
        minimize = getattr(minimizer, "minimize_startup", None)
        if minimize is not None:
            return bool(minimize(self.ctx, state, dynamics_backend=self.ctx.dynamics_backend))
        return False

    def _prepare_startup(self, state: LoopState) -> None:
        if self.ctx.dynamics_backend is DynamicsBackend.BLADE:
            self._minimize_startup(state)
            self._initialize_native_dynamics()
            self._preflight_native_dynamics()
            return
        self._minimize_startup(state)
        self._initialize_native_dynamics()
        self._preflight_native_dynamics()

    def _restore_training_state(self, state: LoopState) -> None:
        cache = getattr(self.hooks, "cache", None)
        if cache is None and not getattr(self.hooks, "uses_training_sidecars", False):
            return
        cache_reader = getattr(self.checkpoint, "read_segment_cache", None)
        if cache is not None and cache_reader is not None:
            restored_cache = cache_reader(max_segments=cache.max_segments, state=state)
            if restored_cache is not None:
                self.hooks.cache = restored_cache

        bias_reader = getattr(self.checkpoint, "read_bias_snapshot", None)
        if bias_reader is None:
            return
        snapshot = bias_reader(nsubs=self.ctx.nsubsites[1:], state=state)
        if snapshot is not None:
            self._bias_rebuilder().apply(snapshot)

    def _should_stop_before_next_segment(self, state: LoopState) -> bool:
        if self._stop_requested or state.stop_requested:
            return True
        if getattr(self._walltime_guard, "job_end_time", True) is None:
            return False
        return not self._walltime_guard.can_start_segment(
            will_fire_cycle=self._cycle_would_fire_next_segment(state)
        )

    def _cycle_would_fire_next_segment(self, state: LoopState) -> bool:
        trigger = getattr(self.hooks, "should_trigger_cycle", None)
        if trigger is None:
            return False
        return bool(trigger(state.advance_segment()))

    def _record_duration(self, bucket: list[float], start: float) -> float:
        duration = max(0.0, self._time_fn() - start)
        bucket.append(duration)
        return duration

    def _install_signal_handlers(self):
        try:
            from cphmd.cli._signals import install_clean_shutdown_handler

            return install_clean_shutdown_handler()
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _restore_signal_handlers(token) -> None:
        return None

    def _write_status_summary(self, state: LoopState, *, run_state: str = "running") -> None:
        if self.ctx.rank != 0:
            return
        writer = getattr(self.checkpoint, "write_status_summary", None)
        if writer is not None:
            writer(state, run_state=run_state)

    def _write_timing_summary(self, state: LoopState, *, run_state: str) -> None:
        if not _debug_timings_enabled():
            return
        payload = {
            "schema_version": 1,
            "rank": self.ctx.rank,
            "replica_label": state.replica_label,
            "segment_idx": state.segment_idx,
            "cycle_idx": state.cycle_idx,
            "run_state": run_state,
            "timings": {
                "segment_dynamics": _timing_stats(self._segment_seconds),
                "archive": _timing_stats(self._archive_seconds),
                "rex_exchange": _timing_stats(self._rex_seconds),
                "rex_hook": _timing_stats(self._rex_hook_seconds),
                "analysis_cycle": _timing_stats(self._cycle_seconds),
                "checkpoint": _timing_stats(self._checkpoint_seconds),
            },
        }
        path = self.ctx.rank_dir / "timing_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def _native_dynamics(self):
        if self._native_dynamics_override is not None:
            return self._native_dynamics_override
        if not self._native_modules:
            return _NoopNativeDynamics()
        from cphmd.native import dynamics as native_dynamics

        return native_dynamics

    def _native_minimizer(self):
        if self._native_minimizer_override is not None:
            return self._native_minimizer_override
        from cphmd.native import minimization as native_minimizer

        return native_minimizer

    def _bias_rebuilder(self):
        if self._bias_rebuilder_override is not None:
            return self._bias_rebuilder_override
        from cphmd.training import BiasRebuilder

        return BiasRebuilder()

    @staticmethod
    def _run_native_segment(**kwargs) -> tuple[np.ndarray, np.ndarray]:
        from cphmd.native import dynamics as native_dynamics

        result = native_dynamics.run_segment(**kwargs)
        return result.lambda_matrix, result.bias_matrix


def _partner_rank(exchange_result: Any) -> int | None:
    for attr in ("partner_rank", "partner"):
        value = getattr(exchange_result, attr, None)
        if value is not None:
            return int(value)
    partner_state = getattr(exchange_result, "partner_state", None)
    if partner_state is not None:
        rank = getattr(partner_state, "rank", None)
        if rank is not None:
            return int(rank)
    return None


class _NoopNativeDynamics:
    def use_blade(self, gpu_id: int | None) -> None:
        return None

    def enable_fast_routines(self) -> None:
        return None

    def use_domdec(self, **kwargs) -> None:
        return None

    def preflight_blade_energy(self) -> None:
        return None

    def preflight_domdec_energy(self, *, gpu: bool) -> None:
        return None


def _with_production_rng_state(rng_state: dict[str, Any], state: LoopState) -> dict[str, Any]:
    if state.integrator_seed is None:
        return rng_state
    updated = dict(rng_state)
    updated["production_chunk"] = {
        "chunk_idx": state.chunk_idx,
        "integrator_seed": state.integrator_seed,
    }
    return updated


def _debug_timings_enabled() -> bool:
    value = os.environ.get("CPHMD_DEBUG_TIMINGS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _timing_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
        }
    total = float(sum(values))
    return {
        "count": len(values),
        "total_s": total,
        "mean_s": total / len(values),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }
