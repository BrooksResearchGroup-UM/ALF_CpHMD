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
    ):
        self.config = config
        self.nsubs = tuple(nsubs)
        self.cycle_runner = cycle_runner
        self.native_block = native_block
        self.replica_ph_values = tuple(replica_ph_values)
        self.cache = cache or SegmentCache(max_segments=config.cache_segments)
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
        return snapshot

    def after_rex_swap(
        self,
        state: LoopState,
        *,
        partner_rank: int | None,
        accepted: bool,
    ) -> None:
        if not accepted:
            return
        native = self._native_block()
        native.set_ph(_ph_for_state_from_values(state, self.replica_ph_values))
        native.sync_state()

    def is_done(self, state: LoopState) -> bool:
        return state.cycle_idx >= self.config.end_cycle

    def _native_block(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block


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
