from __future__ import annotations

from pathlib import Path

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.simulation.shrinker import LambdaPrecision, ShrinkerMetadata, write_segment_parquet


class Archiver:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def _archive_label(self, state: LoopState) -> int:
        if state.replica_label is not None:
            return state.replica_label
        return self.ctx.replica_label

    def _archive_metadata(self, state: LoopState) -> ShrinkerMetadata:
        if not self.ctx.rex_enabled:
            return ShrinkerMetadata(
                ph=self.ctx.ph,
                nblocks=len(self.ctx.lambda_headers) + 1,
                nsites=self.ctx.nsites,
                nsubsites=self.ctx.nsubsites,
                lambda_scale=1 if self.ctx.lambda_precision is LambdaPrecision.FULL else 10000,
                replica_label=self.ctx.replica_label,
                simulation=self.ctx.simulation_name,
                name=f"rep{self.ctx.rank:02d}",
                temperature=self.ctx.temperature,
                time_step=self.ctx.time_step_ps,
                time_start=0.0,
                time_end=0.0,
                save_frequency=self.ctx.nsavl,
                start_step=0,
                total_steps=self.ctx.nsteps_per_segment,
                end_step=self.ctx.nsteps_per_segment - 1,
            )

        label = self._archive_label(state)
        if not 0 <= label < len(self.ctx.replica_ph_values):
            raise ValueError(
                "replica_label must be within replica_ph_values for REX archive metadata "
                f"(got {label}, ladder size {len(self.ctx.replica_ph_values)})"
            )
        return ShrinkerMetadata(
            ph=self.ctx.replica_ph_values[label],
            nblocks=len(self.ctx.lambda_headers) + 1,
            nsites=self.ctx.nsites,
            nsubsites=self.ctx.nsubsites,
            lambda_scale=1 if self.ctx.lambda_precision is LambdaPrecision.FULL else 10000,
            replica_label=label,
            simulation=self.ctx.simulation_name,
            name=f"rep{self.ctx.rank:02d}-label{label:02d}",
            temperature=self.ctx.temperature,
            time_step=self.ctx.time_step_ps,
            time_start=0.0,
            time_end=0.0,
            save_frequency=self.ctx.nsavl,
            start_step=0,
            total_steps=self.ctx.nsteps_per_segment,
            end_step=self.ctx.nsteps_per_segment - 1,
        )

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
        metadata = self._archive_metadata(state)
        metadata = ShrinkerMetadata(
            ph=metadata.ph,
            nblocks=metadata.nblocks,
            nsites=metadata.nsites,
            nsubsites=metadata.nsubsites,
            lambda_scale=metadata.lambda_scale,
            replica_label=metadata.replica_label,
            simulation=metadata.simulation,
            name=metadata.name,
            temperature=metadata.temperature,
            time_step=metadata.time_step,
            time_start=float(timestamps[0]) if len(timestamps) else 0.0,
            time_end=float(timestamps[-1]) if len(timestamps) else 0.0,
            save_frequency=metadata.save_frequency,
            start_step=start_step,
            total_steps=metadata.total_steps,
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

    def prune_from_segment(self, segment_idx: int) -> None:
        for path in self.ctx.rank_dir.glob("segment_*.parquet"):
            try:
                current = int(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if current >= int(segment_idx):
                path.unlink(missing_ok=True)
