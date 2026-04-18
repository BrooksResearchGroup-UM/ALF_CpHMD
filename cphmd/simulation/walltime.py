"""Walltime guard for segment-based native simulations."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass
class WalltimeGuard:
    """EMA-based walltime guard.

    When ``job_end_time`` is ``None`` the guard is disabled. This covers
    interactive runs and schedulers that do not expose ``SLURM_JOB_END_TIME``.
    """

    job_end_time: float | None
    safety_factor: float = 2.0
    alpha: float = 0.3

    _avg_segment_s: float = 0.0
    _avg_checkpoint_s: float = 0.0
    _avg_cycle_s: float = 0.0

    def record_segment_duration(self, duration_s: float) -> None:
        self._avg_segment_s = self._update_ema(self._avg_segment_s, duration_s)

    def record_checkpoint_duration(self, duration_s: float) -> None:
        self._avg_checkpoint_s = self._update_ema(self._avg_checkpoint_s, duration_s)

    def record_cycle_duration(self, duration_s: float) -> None:
        self._avg_cycle_s = self._update_ema(self._avg_cycle_s, duration_s)

    def can_start_segment(self, *, will_fire_cycle: bool) -> bool:
        if self.job_end_time is None:
            return True

        remaining = self.job_end_time - time.time()
        needed = self.safety_factor * (self._avg_segment_s + self._avg_checkpoint_s)
        if will_fire_cycle:
            needed += self._avg_cycle_s
        return remaining >= needed

    def _update_ema(self, current: float, sample: float) -> float:
        if current <= 0.0:
            return sample
        return self.alpha * sample + (1.0 - self.alpha) * current


def job_end_time_from_env() -> float | None:
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
