from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cphmd.simulation.context import LoopState
from cphmd.training.bias_rebuilder import BiasRebuilder
from cphmd.training.bias_snapshot import BiasSnapshot
from cphmd.training.segment_cache import SegmentCache


@dataclass(frozen=True)
class ALFCycleRunner:
    analyzer: Any
    rebuilder: BiasRebuilder

    def initialize_biases(self) -> BiasSnapshot | None:
        initializer = getattr(self.analyzer, "initialize_initial_analysis", None)
        if initializer is None:
            return None
        snapshot = initializer()
        if snapshot is None:
            return None
        if not isinstance(snapshot, BiasSnapshot):
            raise TypeError("ALF initial bias provider must return a BiasSnapshot")
        self.rebuilder.apply(snapshot)
        return snapshot

    def run_cycle(self, *, state: LoopState, cache: SegmentCache) -> BiasSnapshot:
        snapshot = self.analyzer.analyze(state=state, cache=cache)
        if not isinstance(snapshot, BiasSnapshot):
            raise TypeError("ALF analyzer must return a BiasSnapshot")
        self.rebuilder.apply(snapshot)
        return snapshot
