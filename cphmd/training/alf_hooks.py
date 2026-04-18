from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.cycle_every_segments <= 0:
            raise ValueError("cycle_every_segments must be positive")
        if self.end_cycle <= 0:
            raise ValueError("end_cycle must be positive")
        if self.cache_segments <= 0:
            raise ValueError("cache_segments must be positive")


class ALFHooks:
    def __init__(
        self,
        config: ALFTrainingConfig,
        *,
        nsubs: tuple[int, ...],
        cycle_runner: Any,
        cache: SegmentCache | None = None,
    ):
        self.config = config
        self.nsubs = tuple(nsubs)
        self.cycle_runner = cycle_runner
        self.cache = cache or SegmentCache(max_segments=config.cache_segments)
        self.last_snapshot: BiasSnapshot | None = None

    def after_segment(
        self,
        state: LoopState,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> None:
        self.cache = self.cache.append(state.segment_idx, lambda_matrix, bias_matrix)

    def should_trigger_cycle(self, state: LoopState) -> bool:
        return (
            state.segment_idx > 0
            and state.segment_idx % self.config.cycle_every_segments == 0
        )

    def run_cycle(self, state: LoopState) -> BiasSnapshot:
        snapshot = self.cycle_runner.run_cycle(state=state, cache=self.cache)
        self.last_snapshot = snapshot
        return snapshot

    def is_done(self, state: LoopState) -> bool:
        return state.cycle_idx >= self.config.end_cycle
