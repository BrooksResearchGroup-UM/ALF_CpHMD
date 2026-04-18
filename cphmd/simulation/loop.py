from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

import numpy as np

from cphmd.simulation.archiver import Archiver
from cphmd.simulation.checkpoint import CheckpointManager
from cphmd.simulation.context import LoopHooks, LoopState, RunContext

RunSegmentFn = Callable[..., tuple[np.ndarray, np.ndarray]]


def default_native_modules() -> tuple[ModuleType, ...]:
    from cphmd.native import block, dynamics, mpi, rex

    return block, dynamics, mpi, rex


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
        bias_rebuilder: Any | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        self.ctx = ctx
        self.hooks = hooks
        modules = tuple(native_modules) if native_modules is not None else default_native_modules()
        self.checkpoint = checkpoint or CheckpointManager(
            ctx,
            native_modules=modules,
        )
        self.archiver = archiver or Archiver(ctx)
        self.rex_driver = rex_driver if rex_driver is not None else self._build_rex_driver()
        self._run_segment = run_segment or self._run_native_segment
        self._native_dynamics_override = native_dynamics
        self._bias_rebuilder_override = bias_rebuilder
        self._time_fn = time_fn or time.time
        self._stop_requested = False
        self._segment_seconds: list[float] = []
        self._checkpoint_seconds: list[float] = []
        self._cycle_seconds: list[float] = []

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> LoopState:
        state, rng_state = self.checkpoint.resume_or_fresh()
        resumed = state.segment_idx > 0 or state.run_idx > 0 or state.cycle_idx > 0
        if self.ctx.rex_enabled and state.replica_label is None:
            state = state.with_initial_label(self.ctx.replica_label)
        self._initialize_native_dynamics()
        if resumed:
            self._restore_training_state(state)
        previous_handlers = self._install_signal_handlers()
        force_start = resumed
        try:
            while not self.hooks.is_done(state):
                if self._should_stop_before_next_segment(state):
                    state = state.with_stop_requested()
                    break
                segment_started = self._time_fn()
                lambda_matrix, bias_matrix = self._run_segment(
                    nsteps=self.ctx.nsteps_per_segment,
                    nsavl=self.ctx.nsavl,
                    nsavc=self.ctx.nsavc,
                    timestep=self.ctx.time_step_ps,
                    temperature=self.ctx.temperature,
                    gpu_id=self.ctx.gpu_id,
                    blade=self.ctx.use_blade,
                    start=state.segment_idx == 0 or force_start,
                    lambda_headers=self.ctx.lambda_headers,
                )
                force_start = False
                self._record_duration(self._segment_seconds, segment_started)
                self.archiver.write_segment(lambda_matrix, bias_matrix, state)
                self.hooks.after_segment(state, lambda_matrix, bias_matrix)
                state = state.advance_segment()
                if self.rex_driver is not None and self.rex_driver.should_attempt(state):
                    rex_result = self.rex_driver.attempt(state)
                    state = state.with_rex_result(
                        replica_label=rex_result.replica_label,
                        attempted=rex_result.attempted,
                        accepted=rex_result.accepted,
                    )
                if self._should_trigger_cycle(state):
                    cycle_started = self._time_fn()
                    snapshot = self.hooks.run_cycle(state)
                    self._record_duration(self._cycle_seconds, cycle_started)
                    next_state = state.with_cycle_result()
                    self.checkpoint.write_training_sidecars(
                        cache=getattr(self.hooks, "cache", None),
                        bias_snapshot=snapshot,
                        state=next_state,
                    )
                    state = next_state
                if state.segment_idx % self.ctx.checkpoint_every_segments == 0:
                    checkpoint_started = self._time_fn()
                    self.checkpoint.write(state, rng_state=rng_state)
                    self._record_duration(self._checkpoint_seconds, checkpoint_started)
        finally:
            self._restore_signal_handlers(previous_handlers)
        self.checkpoint.write_final(state, rng_state=rng_state)
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
        if not self.ctx.use_blade:
            return
        native_dynamics = self._native_dynamics()
        native_dynamics.use_blade(self.ctx.gpu_id)
        native_dynamics.enable_fast_routines()

    def _restore_training_state(self, state: LoopState) -> None:
        cache = getattr(self.hooks, "cache", None)
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
        end_epoch = self._walltime_end_epoch()
        if end_epoch is None or not self._segment_seconds:
            return False
        needed = self.ctx.walltime_safety_factor * (
            self._average(self._segment_seconds) + self._average(self._checkpoint_seconds)
        )
        if self._cycle_would_fire_next_segment(state):
            needed += self._average(self._cycle_seconds)
        return end_epoch - self._time_fn() <= needed

    def _cycle_would_fire_next_segment(self, state: LoopState) -> bool:
        trigger = getattr(self.hooks, "should_trigger_cycle", None)
        if trigger is None:
            return False
        return bool(trigger(state.advance_segment()))

    def _walltime_end_epoch(self) -> float | None:
        if self.ctx.walltime_end_epoch is not None:
            return float(self.ctx.walltime_end_epoch)
        raw = os.environ.get("SLURM_JOB_END_TIME")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _record_duration(self, bucket: list[float], start: float) -> None:
        bucket.append(max(0.0, self._time_fn() - start))

    @staticmethod
    def _average(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _install_signal_handlers(self):
        previous = {}

        def handler(signum, frame):
            self.request_stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
            except (AttributeError, ValueError):
                continue
        return previous

    @staticmethod
    def _restore_signal_handlers(previous) -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    def _native_dynamics(self):
        if self._native_dynamics_override is not None:
            return self._native_dynamics_override
        from cphmd.native import dynamics as native_dynamics

        return native_dynamics

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
