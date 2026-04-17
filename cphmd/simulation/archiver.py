from __future__ import annotations

from pathlib import Path

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.simulation.shrinker import LambdaPrecision, ShrinkerMetadata, write_segment_parquet


class Archiver:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def write_segment(
        self,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
        state: LoopState,
    ) -> Path:
        start_step = state.segment_idx * self.ctx.nsteps_per_segment
        nframes = lambda_matrix.shape[0]
        frame_steps = start_step + np.arange(nframes, dtype=np.float32) * self.ctx.nsavl
        timestamps = frame_steps * self.ctx.time_step_ps
        metadata = ShrinkerMetadata(
            ph=self.ctx.ph,
            nblocks=len(self.ctx.lambda_headers) + 1,
            nsites=self.ctx.nsites,
            nsubsites=self.ctx.nsubsites,
            lambda_scale=1 if self.ctx.lambda_precision is LambdaPrecision.FULL else 10000,
            simulation=self.ctx.simulation_name,
            name=f"rep{self.ctx.rank:02d}",
            temperature=self.ctx.temperature,
            time_step=self.ctx.time_step_ps,
            time_start=float(timestamps[0]) if len(timestamps) else 0.0,
            time_end=float(timestamps[-1]) if len(timestamps) else 0.0,
            save_frequency=self.ctx.nsavl,
            start_step=start_step,
            total_steps=self.ctx.nsteps_per_segment,
            end_step=start_step + self.ctx.nsteps_per_segment - 1,
        )
        return write_segment_parquet(
            self.ctx.segment_path(state.segment_idx),
            lambda_matrix,
            timestamps,
            self.ctx.lambda_headers,
            metadata,
            precision=self.ctx.lambda_precision,
        )
