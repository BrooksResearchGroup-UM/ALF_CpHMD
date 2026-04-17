from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType

import numpy as np

from cphmd.simulation.archiver import Archiver
from cphmd.simulation.checkpoint import CheckpointManager
from cphmd.simulation.context import LoopHooks, LoopState, RunContext

RunSegmentFn = Callable[..., tuple[np.ndarray, np.ndarray]]


class SimulationLoop:
    def __init__(
        self,
        ctx: RunContext,
        hooks: LoopHooks,
        *,
        checkpoint: CheckpointManager | None = None,
        archiver: Archiver | None = None,
        run_segment: RunSegmentFn | None = None,
        native_modules: Iterable[ModuleType] | None = None,
    ):
        self.ctx = ctx
        self.hooks = hooks
        self.checkpoint = checkpoint or CheckpointManager(
            ctx,
            native_modules=native_modules or [],
        )
        self.archiver = archiver or Archiver(ctx)
        self._run_segment = run_segment or self._run_native_segment

    def run(self) -> LoopState:
        state, rng_state = self.checkpoint.resume_or_fresh()
        while not self.hooks.is_done(state):
            lambda_matrix, bias_matrix = self._run_segment(
                nsteps=self.ctx.nsteps_per_segment,
                nsavl=self.ctx.nsavl,
                nsavc=self.ctx.nsavc,
                timestep=self.ctx.time_step_ps,
                temperature=self.ctx.temperature,
                gpu_id=self.ctx.gpu_id,
                blade=False,
                start=state.segment_idx == 0,
            )
            self.archiver.write_segment(lambda_matrix, bias_matrix, state)
            self.hooks.after_segment(state, lambda_matrix, bias_matrix)
            state = state.advance_segment()
            if state.segment_idx % self.ctx.checkpoint_every_segments == 0:
                self.checkpoint.write(state, rng_state=rng_state)
        self.checkpoint.write_final(state, rng_state=rng_state)
        return state

    @staticmethod
    def _run_native_segment(**kwargs) -> tuple[np.ndarray, np.ndarray]:
        from cphmd.native import dynamics as native_dynamics

        result = native_dynamics.run_segment(**kwargs)
        return result.lambda_matrix, result.bias_matrix
