"""Phase switching logic for ALF simulations.

Implements automatic phase progression (1→2→3) based on:
- Lambda overlap (good sampling across states)
- Sample count thresholds
- pKa convergence (optional, for CpHMD)

Also implements Phase 3 → STOP criteria based on:
- Physical state sample counts (unified N-state formula)
- Balance across states (spread tolerance)
- Bias parameter stability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition criteria."""

    # Lambda threshold for counting "occupied" states
    lambda_threshold: float = 0.8

    # Spread tolerance for phase 1→2 transition (relaxed: Phase 1 is rough equilibration)
    spread_1to2: float = 0.5

    # Spread tolerance for phase 2→3 transition
    spread_2to3: float = 0.2

    # Minimum sample count per state for phase 1→2
    min_hits_1to2: int = 100

    # Minimum sample count per state for phase 2→3
    min_hits_2to3: int = 1000

    # pKa tolerance for phase 1→2 (relaxed)
    pka_tolerance_1to2: float = 1.5

    # pKa tolerance for phase 2→3 (strict)
    pka_tolerance_2to3: float = 0.3

    # Minimum transition connectivity for phase 1→2 transition
    # Prevents transition when no actual within-run transitions occur
    # (catches random-init false positives where accumulated spread looks OK
    # but every individual run is trapped in a single state)
    min_connectivity_1to2: float = 0.01

    # Minimum number of individual runs (out of accumulated window) that must
    # show multi-state behavior (2+ states visited) for the accumulated
    # Phase 1 check to be valid. Prevents random initialization from
    # creating fake balanced populations across runs.
    min_multistate_runs_1to2: int = 3

    # Minimum transition connectivity for phase 2→3 transition
    # Prevents premature transition when sampling is poor
    min_connectivity_2to3: float = 0.2

    # Per-state minimum normalized fraction at strict threshold for 2→3
    # Prevents transition when any state has near-zero population
    min_state_fraction_2to3: float = 0.01  # 1% per state
    strict_threshold_2to3: float = 0.985

    # Minimum runs in Phase 2 before allowing 2→3 transition
    # Ensures x/s coupling cutoffs reach near-target values
    min_phase2_runs: int = 15

    # EWBS threshold for 2→3 transition (bias stability gate)
    ewbs_2to3: float = 0.10
    ewbs_2to3_window: int = 5  # Consecutive runs below threshold


@dataclass
class StopCriteriaConfig:
    """Configuration for Phase 3 → STOP criteria.

    Uses step5_bias_search-style scoring based on population fractions
    at the strict lambda threshold (0.985).

    Stop when:
        1. Population fraction difference < max_frac_diff (e.g., 2%)
        2. Minimum sample count per state achieved
        3. Bias parameters are stable
    """

    # Lambda threshold for physical state detection (strict endpoint)
    threshold_strict: float = 0.985

    # Maximum allowed fraction difference between states (e.g., 0.02 = 2%)
    max_frac_diff: float = 0.02

    # Minimum total samples required (not per-state)
    min_total_samples: int = 100_000

    # Timestep in femtoseconds (for sample count scaling with HMR)
    timestep_fs: float = 2.0

    # Bias stability: rolling window and max std tolerance
    bias_window: int = 10
    bias_max_std: float = 0.5

    # Minimum normalized entropy per site (S/log(N), range [0,1])
    # Catches trapped states where frac_diff looks fine but one state is unsampled
    min_entropy: float = 0.85

    # Block averaging: number of blocks and max allowed per-state variance
    n_blocks: int = 5
    max_block_variance: float = 0.01

    # EWBS (Energy-Weighted Bias Stability) threshold
    # Max per-type smoothed RMS bias change for convergence
    max_ewbs: float = 0.03  # ~kT/20 at 300K
    ewbs_window: int = 10   # Consecutive runs below threshold required

    # Scoring parameters (step5-style combined score)
    # score = avg_frac - alpha * frac_diff
    alpha: float = 10.0

    def min_samples(self) -> int:
        """Compute min samples adjusted for timestep.

        Returns:
            Minimum total samples required
        """
        timestep_factor = 2.0 / self.timestep_fs  # HMR uses 4fs → factor 0.5
        return int(self.min_total_samples * timestep_factor)


@dataclass
class ReplicaLambdaData:
    """Lambda data organized by replica for pKa fitting.

    Attributes:
        replica_idx: Replica index (0-based)
        pH: Effective pH for this replica
        lambda_data: Combined lambda array for this replica (samples x states)
        populations: Population fraction per state (after threshold)
    """
    replica_idx: int
    pH: float
    lambda_data: np.ndarray
    populations: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class PKaFitResult:
    """Result of pKa fitting from multi-replica data.

    Attributes:
        fitted_pKa: The pKa value from HH curve fitting
        fit_type: Type of fit ("acidic", "basic", "three_state", "constant")
        r_squared: Coefficient of determination
        n_points: Number of pH points used in fit
        pH_values: Array of pH values used
        populations: Array of populations at each pH
    """
    fitted_pKa: float
    fit_type: str = "unknown"
    r_squared: float = 0.0
    n_points: int = 0
    pH_values: np.ndarray = field(default_factory=lambda: np.array([]))
    populations: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class StopCriteriaResult:
    """Result of Phase 3 → STOP criteria check.

    Uses step5_bias_search-style metrics based on population fractions.

    Attributes:
        should_stop: True if all criteria are met
        n_states: Number of states in the system
        n_samples: Total number of samples
        fractions: Population fractions per state (at strict threshold)
        avg_frac: Average fraction across states
        frac_diff: Max fraction - min fraction
        frac_diff_pct: Fraction difference as percentage
        score: Combined score (avg - alpha * diff)
        bias_stable: True if bias parameters are stable
        bias_rolling_std: Rolling std of bias parameters (if checked)
        entropy: Raw population entropy S = -Σ pᵢ log(pᵢ)
        entropy_normalized: Worst-site normalized entropy S/log(N) in [0,1]
        block_variance: Worst per-state population variance across blocks
        reasons: List of reasons for not stopping (empty if should_stop)
    """
    should_stop: bool
    n_states: int
    n_samples: int = 0
    fractions: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_frac: float = 0.0
    frac_diff: float = 0.0
    frac_diff_pct: float = 0.0
    score: float = 0.0
    bias_stable: bool = True
    bias_rolling_std: float = 0.0
    entropy: float = 0.0
    entropy_normalized: float = 0.0
    block_variance: float = 0.0
    ewbs: float = float("inf")
    ewbs_bottleneck: str = ""
    reasons: list[str] = field(default_factory=list)


def _per_site_ranges(nsubs: list[int]) -> list[tuple[int, int]]:
    """Compute column index ranges for each site.

    Args:
        nsubs: Number of substates per site, e.g. [2, 3]

    Returns:
        List of (start, end) tuples for slicing columns per site
    """
    ranges = []
    offset = 0
    for ns in nsubs:
        ranges.append((offset, offset + ns))
        offset += ns
    return ranges


def good_overlap(fracs: np.ndarray, spread_tol: float, nsubs: list[int] | None = None) -> bool:
    """Check if lambda fractions have good overlap across states.

    When nsubs is provided, checks spread **per site** and requires ALL
    sites to pass. This prevents multi-site systems from producing
    meaningless global spread values.

    Args:
        fracs: Array of fraction of samples above threshold per state
        spread_tol: Maximum allowed spread (max - min)
        nsubs: Number of substates per site (e.g. [2, 3]). If None,
               treats all states as one site (legacy behavior).

    Returns:
        True if spread is within tolerance (for every site when nsubs given)
    """
    if fracs.size == 0:
        return False
    if nsubs is None:
        return float(np.max(fracs) - np.min(fracs)) < spread_tol
    for start, end in _per_site_ranges(nsubs):
        site_fracs = fracs[start:end]
        if float(np.max(site_fracs) - np.min(site_fracs)) >= spread_tol:
            return False
    return True


def enough_samples(mask: np.ndarray, min_hits: int = 200) -> bool:
    """Check if every state has minimum sample count.

    Args:
        mask: Boolean array (samples x states) indicating above-threshold
        min_hits: Minimum required samples per state

    Returns:
        True if all states have sufficient samples
    """
    if mask.size == 0:
        return False
    hits = np.sum(mask, axis=0)
    return bool(np.all(hits >= min_hits))


def load_lambda_data(data_dir: Path) -> tuple[np.ndarray | None, list[str]]:
    """Load and combine lambda data from all replica files.

    Supports .parquet (preferred) with .dat fallback for old runs.

    Args:
        data_dir: Path to data/ directory containing Lambda.*.*.{parquet,dat} files

    Returns:
        Tuple of (combined lambda data array, list of representative files)
    """
    from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values

    if not data_dir.exists():
        return None, []

    lambda_fpaths = find_lambda_files(data_dir)
    if not lambda_fpaths:
        return None, []

    # Organize by replica
    replica_files: dict[int, list[Path]] = {}
    for fpath in lambda_fpaths:
        try:
            parts = fpath.name.split('.')
            if len(parts) >= 4:
                replica_idx = int(parts[2])
                if replica_idx not in replica_files:
                    replica_files[replica_idx] = []
                replica_files[replica_idx].append(fpath)
        except (ValueError, IndexError):
            continue

    if not replica_files:
        return None, []

    # Combine data from all replicas
    lambda_data = None
    l_files: list[str] = []

    for replica_idx in sorted(replica_files.keys()):
        repeat_fpaths = sorted(replica_files[replica_idx])

        replica_combined = None
        for fpath in repeat_fpaths:
            l = read_lambda_values(fpath)
            if replica_combined is None:
                replica_combined = l
            else:
                replica_combined = np.vstack([replica_combined, l])

        if replica_combined is not None:
            if lambda_data is None:
                lambda_data = replica_combined
            else:
                lambda_data = np.vstack([lambda_data, replica_combined])

            l_files.append(repeat_fpaths[0].name)

    return lambda_data, l_files


def load_lambda_data_per_replica(
    data_dir: Path,
    effective_pH: float,
    delta_pKa: float,
    nreps: int,
    ncentral: int | None = None,
) -> list[ReplicaLambdaData]:
    """Load lambda data organized by replica with per-replica pH values.

    Supports .parquet (preferred) with .dat fallback for old runs.

    Args:
        data_dir: Path to data/ directory containing Lambda.*.*.{parquet,dat} files
        effective_pH: Base pH for the central replica
        delta_pKa: pH increment between adjacent replicas
        nreps: Total number of replicas
        ncentral: Index of central replica (defaults to nreps // 2)

    Returns:
        List of ReplicaLambdaData objects, one per replica with valid data
    """
    from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values

    if not data_dir.exists():
        return []

    if ncentral is None:
        ncentral = nreps // 2

    lambda_fpaths = find_lambda_files(data_dir)
    if not lambda_fpaths:
        return []

    # Organize files by replica
    replica_files: dict[int, list[Path]] = {}
    for fpath in lambda_fpaths:
        try:
            parts = fpath.name.split('.')
            if len(parts) >= 4:
                replica_idx = int(parts[2])
                if replica_idx not in replica_files:
                    replica_files[replica_idx] = []
                replica_files[replica_idx].append(fpath)
        except (ValueError, IndexError):
            continue

    if not replica_files:
        return []

    # Build per-replica data structures
    result: list[ReplicaLambdaData] = []

    for replica_idx in sorted(replica_files.keys()):
        if replica_idx >= nreps:
            continue

        replica_pH = effective_pH + delta_pKa * (replica_idx - ncentral)

        replica_lambda = None
        for fpath in sorted(replica_files[replica_idx]):
            try:
                data = read_lambda_values(fpath)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if replica_lambda is None:
                    replica_lambda = data
                else:
                    replica_lambda = np.vstack([replica_lambda, data])
            except Exception:
                continue

        if replica_lambda is not None and replica_lambda.size > 0:
            result.append(ReplicaLambdaData(
                replica_idx=replica_idx,
                pH=replica_pH,
                lambda_data=replica_lambda,
            ))

    return result


def compute_replica_populations(
    replica_data: list[ReplicaLambdaData],
    lambda_threshold: float = 0.8,
    state_indices: list[int] | None = None,
) -> list[ReplicaLambdaData]:
    """Compute state populations for each replica.

    For each replica, calculates the fraction of samples where lambda > threshold
    for the specified states (or all states if not specified).

    Args:
        replica_data: List of ReplicaLambdaData with lambda_data filled
        lambda_threshold: Threshold for counting a sample as "in" a state
        state_indices: Optional list of state indices to sum (for multi-state sites)

    Returns:
        Same list with populations field filled in
    """
    for rd in replica_data:
        if rd.lambda_data is None or rd.lambda_data.size == 0:
            rd.populations = np.array([])
            continue

        mask = rd.lambda_data > lambda_threshold
        n_samples = rd.lambda_data.shape[0]

        if state_indices is not None:
            # Sum populations over specified states
            total_hits = 0
            for idx in state_indices:
                if idx < mask.shape[1]:
                    total_hits += mask[:, idx].sum()
            rd.populations = np.array([total_hits / n_samples])
        else:
            # Per-state populations
            rd.populations = mask.mean(axis=0)

    return replica_data


def calculate_populations(
    lambda_data: np.ndarray,
    thresholds: tuple[float, float] = (0.8, 0.985),
    nsubs: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Calculate state populations at different lambda thresholds.

    For each threshold, computes two types of populations:
    1. Raw: fraction of ALL samples where lambda > threshold per state
    2. Normalized: distribution among physical-state samples only

    When nsubs is provided, normalization is done **per site** so that each
    site's substates sum to 1.0. Without nsubs, global normalization is used
    (all states sum to 1.0).

    Args:
        lambda_data: Combined lambda data array (samples x states)
        thresholds: Tuple of (relaxed, strict) lambda thresholds
        nsubs: Number of substates per site (e.g. [2, 3]). If None,
               normalizes globally (legacy behavior).

    Returns:
        Dictionary with population arrays and statistics
    """
    if lambda_data is None or lambda_data.size == 0:
        return {}

    n_samples, n_states = lambda_data.shape
    relaxed_thresh, strict_thresh = thresholds

    # Calculate masks (samples above threshold per state)
    mask_relaxed = lambda_data > relaxed_thresh
    mask_strict = lambda_data > strict_thresh

    # Count hits (number of samples above threshold per state)
    hits_relaxed = mask_relaxed.sum(axis=0)
    hits_strict = mask_strict.sum(axis=0)

    # Raw populations (fraction of ALL samples per state) — site-independent
    pop_relaxed_raw = mask_relaxed.mean(axis=0)
    pop_strict_raw = mask_strict.mean(axis=0)

    # Normalized populations — per-site when nsubs provided
    total_hits_relaxed = int(hits_relaxed.sum())
    total_hits_strict = int(hits_strict.sum())

    if nsubs is not None:
        # Per-site normalization: each site's substates sum to 1.0
        pop_relaxed_norm = np.zeros(n_states)
        pop_strict_norm = np.zeros(n_states)
        for start, end in _per_site_ranges(nsubs):
            site_total_r = hits_relaxed[start:end].sum()
            site_total_s = hits_strict[start:end].sum()
            if site_total_r > 0:
                pop_relaxed_norm[start:end] = hits_relaxed[start:end] / site_total_r
            if site_total_s > 0:
                pop_strict_norm[start:end] = hits_strict[start:end] / site_total_s
    else:
        # Global normalization (legacy)
        if total_hits_relaxed > 0:
            pop_relaxed_norm = hits_relaxed / total_hits_relaxed
        else:
            pop_relaxed_norm = np.zeros(n_states)

        if total_hits_strict > 0:
            pop_strict_norm = hits_strict / total_hits_strict
        else:
            pop_strict_norm = np.zeros(n_states)

    return {
        "n_samples": n_samples,
        "n_states": n_states,
        "nsubs": nsubs,
        "threshold_relaxed": relaxed_thresh,
        "threshold_strict": strict_thresh,
        # Raw populations (fraction of all samples)
        "pop_relaxed_raw": pop_relaxed_raw,
        "pop_strict_raw": pop_strict_raw,
        # Normalized populations (per-site when nsubs given)
        "pop_relaxed_norm": pop_relaxed_norm,
        "pop_strict_norm": pop_strict_norm,
        # Hit counts
        "hits_relaxed": hits_relaxed,
        "hits_strict": hits_strict,
        "total_hits_relaxed": total_hits_relaxed,
        "total_hits_strict": total_hits_strict,
    }



@dataclass
class EWBSState:
    """Exponentially Weighted Bias Stability tracking.

    Tracks per-type RMS bias changes with exponential moving average (EWMA).
    EWBS = max(ema_b, ema_c, ema_x, ema_s) — the worst-type smoothed
    per-parameter RMS change. All types must converge for EWBS to be low.

    Units are in CHARMM LDBV bias units (effectively kT at simulation temp).
    """

    ema_b: float = 0.0
    ema_c: float = 0.0
    ema_x: float = 0.0
    ema_s: float = 0.0
    ewbs: float = float("inf")
    history: list[float] = field(default_factory=list)
    alpha: float = 0.3  # EWMA smoothing (α=0.3 ≈ 5-run memory)

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        import json
        import math

        def _sanitize(v: float) -> float | None:
            """Convert inf/nan to None for JSON compatibility."""
            if not math.isfinite(v):
                return None
            return v

        data = {
            "ema_b": _sanitize(self.ema_b),
            "ema_c": _sanitize(self.ema_c),
            "ema_x": _sanitize(self.ema_x),
            "ema_s": _sanitize(self.ema_s),
            "ewbs": _sanitize(self.ewbs),
            "history": [_sanitize(h) for h in self.history],
            "alpha": self.alpha,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EWBSState":
        """Load state from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        state = cls()
        def _restore(v, default=0.0):
            """Restore None back to default (inf/nan were saved as None)."""
            return default if v is None else float(v)

        state.ema_b = _restore(data.get("ema_b"), 0.0)
        state.ema_c = _restore(data.get("ema_c"), 0.0)
        state.ema_x = _restore(data.get("ema_x"), 0.0)
        state.ema_s = _restore(data.get("ema_s"), 0.0)
        state.ewbs = _restore(data.get("ewbs"), float("inf"))
        raw_history = data.get("history", [])
        if not isinstance(raw_history, list):
            raw_history = []
        state.history = [_restore(h, float("inf")) for h in raw_history]
        state.alpha = _restore(data.get("alpha"), 0.3)
        return state


def compute_rms_changes(
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute per-type RMS magnitude of bias changes.

    Each type's RMS is computed over its nonzero entries (zero entries
    indicate disabled or reference parameters).

    Args:
        b: Linear bias changes (1 x nblocks or flat).
        c: Quadratic bias changes (nblocks x nblocks).
        x: Cross-term x bias changes (nblocks x nblocks).
        s: Cross-term s bias changes (nblocks x nblocks).

    Returns:
        Tuple of (rms_b, rms_c, rms_x, rms_s).
    """

    def _rms_nonzero(arr: np.ndarray) -> float:
        vals = arr.ravel()
        vals = vals[np.isfinite(vals) & (vals != 0)]
        if vals.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(vals**2)))

    return _rms_nonzero(b), _rms_nonzero(c), _rms_nonzero(x), _rms_nonzero(s)


def update_ewbs_state(
    state: EWBSState,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
) -> float:
    """Update EWBS state with new bias changes and return current EWBS.

    EWBS is the worst-type smoothed per-parameter RMS change. When all
    four types (b, c, x, s) have small smoothed RMS, the biases are stable.

    Args:
        state: EWBSState to update in-place.
        b: Linear bias changes from current analysis.
        c: Quadratic bias changes from current analysis.
        x: Cross-term x bias changes from current analysis.
        s: Cross-term s bias changes from current analysis.

    Returns:
        Current EWBS value.
    """
    rms_b, rms_c, rms_x, rms_s = compute_rms_changes(b, c, x, s)

    a = state.alpha
    if not state.history:
        # First update: initialize directly
        state.ema_b = rms_b
        state.ema_c = rms_c
        state.ema_x = rms_x
        state.ema_s = rms_s
    else:
        state.ema_b = a * rms_b + (1 - a) * state.ema_b
        state.ema_c = a * rms_c + (1 - a) * state.ema_c
        state.ema_x = a * rms_x + (1 - a) * state.ema_x
        state.ema_s = a * rms_s + (1 - a) * state.ema_s

    state.ewbs = max(state.ema_b, state.ema_c, state.ema_x, state.ema_s)
    state.history.append(state.ewbs)
    return state.ewbs


def ewbs_bottleneck_type(state: EWBSState) -> str:
    """Return which parameter type is the EWBS bottleneck."""
    emas = {"b": state.ema_b, "c": state.ema_c, "x": state.ema_x, "s": state.ema_s}
    return max(emas, key=emas.get)


def compute_entropy(
    fractions: np.ndarray,
    nsubs: list[int] | None = None,
) -> tuple[float, float]:
    """Compute population entropy and worst-site normalized entropy.

    For per-site systems, computes entropy per site and returns the worst
    (lowest) normalized entropy. Catches trapped states where frac_diff
    looks fine but the distribution is far from uniform.

    Args:
        fractions: Per-state population fractions (normalized per site).
        nsubs: Number of substates per site.

    Returns:
        (raw_entropy, worst_site_normalized_entropy) where normalized
        is S/log(N) in [0, 1]. A value of 1.0 means perfectly uniform.
    """
    if fractions.size == 0 or fractions.sum() == 0:
        return 0.0, 0.0

    if nsubs is not None:
        worst_norm = 1.0
        for start, end in _per_site_ranges(nsubs):
            site_fracs = fractions[start:end]
            nonzero = site_fracs[site_fracs > 0]
            if len(nonzero) == 0:
                worst_norm = 0.0
                continue
            s = float(-np.sum(nonzero * np.log(nonzero)))
            s_max = np.log(end - start)
            norm = s / s_max if s_max > 0 else 1.0  # 1-substate is trivially uniform
            worst_norm = min(worst_norm, norm)
        # Raw entropy over all states
        nonzero_all = fractions[fractions > 0]
        raw = float(-np.sum(nonzero_all * np.log(nonzero_all)))
        return raw, worst_norm
    else:
        nonzero = fractions[fractions > 0]
        raw = float(-np.sum(nonzero * np.log(nonzero)))
        s_max = np.log(len(fractions))
        norm = raw / s_max if s_max > 0 else 0.0
        return raw, norm


def compute_block_variance(
    lambda_data: np.ndarray,
    n_blocks: int,
    threshold: float,
    nsubs: list[int] | None = None,
) -> float:
    """Compute worst per-state population variance across blocks.

    Splits lambda data into equal blocks, computes per-site normalized
    populations in each block, then returns the maximum per-state
    variance across blocks. High variance means populations are still
    drifting even though their average looks converged.

    Args:
        lambda_data: Combined lambda data array (samples x states).
        n_blocks: Number of blocks to split into.
        threshold: Lambda threshold for physical state detection.
        nsubs: Number of substates per site.

    Returns:
        Maximum per-state population variance across blocks.
        Returns 0.0 if too few samples for meaningful blocking.
    """
    n_samples = lambda_data.shape[0]
    block_size = n_samples // n_blocks
    if block_size < 50:
        return 0.0

    block_pops = []
    for i in range(n_blocks):
        block = lambda_data[i * block_size : (i + 1) * block_size]
        mask = block > threshold
        fracs = mask.mean(axis=0)

        if nsubs is not None:
            norm_fracs = np.zeros_like(fracs)
            for start, end in _per_site_ranges(nsubs):
                site_total = fracs[start:end].sum()
                if site_total > 0:
                    norm_fracs[start:end] = fracs[start:end] / site_total
            fracs = norm_fracs
        else:
            total = fracs.sum()
            if total > 0:
                fracs = fracs / total

        block_pops.append(fracs)

    block_pops_arr = np.array(block_pops)  # (n_blocks, n_states)
    per_state_var = np.var(block_pops_arr, axis=0)
    return float(per_state_var.max())


def compute_spread(counts: np.ndarray) -> float:
    """Compute normalized spread: (max - min) / (max + min).

    Args:
        counts: Array of sample counts per state

    Returns:
        Spread value between 0 (perfectly balanced) and 1 (one state dominates)
    """
    if counts.size == 0:
        return 1.0
    max_c, min_c = counts.max(), counts.min()
    if max_c + min_c == 0:
        return 1.0
    return float((max_c - min_c) / (max_c + min_c))


def check_stop_criteria(
    lambda_data: np.ndarray,
    config: StopCriteriaConfig | None = None,
    bias_history: np.ndarray | None = None,
    nsubs: list[int] | None = None,
    ewbs_state: EWBSState | None = None,
) -> StopCriteriaResult:
    """Check if Phase 3 → STOP criteria are met.

    Uses step5_bias_search-style scoring based on population fractions
    at the strict lambda threshold (0.985).

    When nsubs is provided, frac_diff is computed **per site** and the
    worst (largest) site frac_diff is used. Fractions are also normalized
    per-site for the score calculation. All sites must be balanced for
    the overall check to pass.

    Stop criteria:
        1. Total samples >= min_total_samples (adjusted for HMR)
        2. Fraction difference < max_frac_diff (e.g., 2%) — per site
        3. Bias parameters stable (rolling std < threshold)

    Args:
        lambda_data: Combined lambda data array (samples x states)
        config: Stop criteria configuration (uses defaults if None)
        bias_history: Optional array of bias values over iterations for stability check
                      Shape: (n_iterations, n_bias_params) or 1D for single param
        nsubs: Number of substates per site (e.g. [2, 3]). If None,
               uses global metrics (legacy behavior).
        ewbs_state: Optional EWBSState for bias stability metric.

    Returns:
        StopCriteriaResult with detailed metrics and decision
    """
    if config is None:
        config = StopCriteriaConfig()

    if lambda_data is None or lambda_data.size == 0:
        return StopCriteriaResult(
            should_stop=False,
            n_states=0,
            reasons=["No lambda data available"],
        )

    n_samples, n_states = lambda_data.shape

    # Compute population fractions at strict threshold (step5 style)
    mask_strict = lambda_data > config.threshold_strict
    raw_fractions = mask_strict.sum(axis=0) / n_samples

    if nsubs is not None:
        # Per-site normalization and frac_diff
        fractions = np.zeros(n_states)
        site_diffs = []
        for start, end in _per_site_ranges(nsubs):
            site_raw = raw_fractions[start:end]
            site_total = site_raw.sum()
            if site_total > 0:
                fractions[start:end] = site_raw / site_total
            else:
                fractions[start:end] = 0.0
            site_diffs.append(float(np.max(site_raw) - np.min(site_raw)))
        frac_diff = max(site_diffs)
    else:
        fractions = raw_fractions
        frac_diff = float(np.max(fractions) - np.min(fractions))

    # Step5-style metrics
    avg_frac = float(np.mean(fractions))
    frac_diff_pct = frac_diff * 100
    score = avg_frac - config.alpha * frac_diff

    # Check bias stability if history provided
    bias_stable = True
    bias_rolling_std = 0.0

    if bias_history is not None and bias_history.size > 0:
        if bias_history.ndim == 1:
            bias_history = bias_history.reshape(-1, 1)

        if bias_history.shape[0] >= config.bias_window:
            # Compute rolling std over last window iterations
            window_data = bias_history[-config.bias_window:]
            rolling_std = np.std(window_data, axis=0).max()
            bias_rolling_std = float(rolling_std)
            bias_stable = bias_rolling_std <= config.bias_max_std

    # Compute population entropy (catches trapped states)
    entropy_raw, entropy_norm = compute_entropy(fractions, nsubs)

    # Compute block variance (catches drifting populations)
    block_var = compute_block_variance(
        lambda_data, config.n_blocks, config.threshold_strict, nsubs,
    )

    # Determine if all criteria are met
    reasons: list[str] = []

    min_samples = config.min_samples()
    if n_samples < min_samples:
        reasons.append(f"samples={n_samples:,}<{min_samples:,}")

    if frac_diff > config.max_frac_diff:
        reasons.append(f"frac_diff={frac_diff_pct:.2f}%>{config.max_frac_diff*100:.1f}%")

    if not bias_stable:
        reasons.append(f"bias_std={bias_rolling_std:.3f}>{config.bias_max_std}")

    if entropy_norm < config.min_entropy:
        reasons.append(f"entropy={entropy_norm:.2f}<{config.min_entropy}")

    if block_var > config.max_block_variance:
        reasons.append(f"block_var={block_var:.4f}>{config.max_block_variance}")

    # EWBS check: bias changes must be small for sufficient consecutive runs
    ewbs_val = float("inf")
    ewbs_btn = ""
    if ewbs_state is not None:
        if not ewbs_state.history:
            reasons.append("ewbs: no history available")
        else:
            ewbs_val = ewbs_state.ewbs
            ewbs_btn = ewbs_bottleneck_type(ewbs_state)
            window = min(config.ewbs_window, len(ewbs_state.history))
            recent = ewbs_state.history[-window:]
            if any(v > config.max_ewbs for v in recent):
                reasons.append(
                    f"ewbs={ewbs_val:.4f}>{config.max_ewbs}({ewbs_btn})"
                )

    should_stop = len(reasons) == 0

    return StopCriteriaResult(
        should_stop=should_stop,
        n_states=n_states,
        n_samples=n_samples,
        fractions=fractions,
        avg_frac=avg_frac,
        frac_diff=frac_diff,
        frac_diff_pct=frac_diff_pct,
        score=score,
        bias_stable=bias_stable,
        bias_rolling_std=bias_rolling_std,
        entropy=entropy_raw,
        entropy_normalized=entropy_norm,
        block_variance=block_var,
        ewbs=ewbs_val,
        ewbs_bottleneck=ewbs_btn,
        reasons=reasons,
    )


def write_populations_file(
    filepath: Path,
    pop_data: dict[str, np.ndarray],
) -> None:
    """Write population statistics to a file.

    Args:
        filepath: Path to output file
        pop_data: Dictionary from calculate_populations()
    """
    if not pop_data:
        return

    n_states = pop_data["n_states"]
    n_samples = pop_data["n_samples"]
    thresh_r = pop_data["threshold_relaxed"]
    thresh_s = pop_data["threshold_strict"]
    total_r = pop_data["total_hits_relaxed"]
    total_s = pop_data["total_hits_strict"]

    nsubs = pop_data.get("nsubs")

    with open(filepath, "w") as f:
        f.write("# Lambda Population Analysis\n")
        f.write(f"# Total samples: {n_samples}\n")
        if nsubs is not None:
            f.write(f"# Sites: {len(nsubs)}, nsubs={nsubs}\n")
        f.write(f"# Relaxed threshold: {thresh_r} (total physical: {total_r})\n")
        f.write(f"# Strict threshold: {thresh_s} (total physical: {total_s})\n")
        f.write("#\n")
        f.write("# Raw = fraction of ALL samples in physical state\n")
        if nsubs is not None:
            f.write("# Norm = fraction among physical samples per site (each site sums to 1)\n")
        else:
            f.write("# Norm = fraction among physical samples only (sums to 1)\n")
        f.write("#\n")

        # Header
        f.write(f"# State  Raw(>{thresh_r})  Norm(>{thresh_r})  Hits(>{thresh_r})  ")
        f.write(f"Raw(>{thresh_s})  Norm(>{thresh_s})  Hits(>{thresh_s})\n")

        for i in range(n_states):
            f.write(
                f"  {i:5d}  "
                f"{pop_data['pop_relaxed_raw'][i]:11.4f}  "
                f"{pop_data['pop_relaxed_norm'][i]:12.4f}  "
                f"{int(pop_data['hits_relaxed'][i]):12d}  "
                f"{pop_data['pop_strict_raw'][i]:11.4f}  "
                f"{pop_data['pop_strict_norm'][i]:12.4f}  "
                f"{int(pop_data['hits_strict'][i]):12d}\n"
            )

        # Summary statistics
        f.write("#\n")
        f.write(f"# Raw summary (>{thresh_r}): "
                f"min={pop_data['pop_relaxed_raw'].min():.4f}, "
                f"max={pop_data['pop_relaxed_raw'].max():.4f}, "
                f"spread={pop_data['pop_relaxed_raw'].max() - pop_data['pop_relaxed_raw'].min():.4f}\n")
        f.write(f"# Raw summary (>{thresh_s}): "
                f"min={pop_data['pop_strict_raw'].min():.4f}, "
                f"max={pop_data['pop_strict_raw'].max():.4f}, "
                f"spread={pop_data['pop_strict_raw'].max() - pop_data['pop_strict_raw'].min():.4f}\n")
        f.write("#\n")
        if nsubs is not None:
            # Per-site norm summaries
            for site_idx, (start, end) in enumerate(_per_site_ranges(nsubs)):
                site_r = pop_data['pop_relaxed_norm'][start:end]
                site_s = pop_data['pop_strict_norm'][start:end]
                f.write(f"# Norm site {site_idx} (>{thresh_r}): "
                        f"min={site_r.min():.4f}, "
                        f"max={site_r.max():.4f}, "
                        f"sum={site_r.sum():.4f}\n")
                f.write(f"# Norm site {site_idx} (>{thresh_s}): "
                        f"min={site_s.min():.4f}, "
                        f"max={site_s.max():.4f}, "
                        f"sum={site_s.sum():.4f}\n")
        else:
            f.write(f"# Norm summary (>{thresh_r}): "
                    f"min={pop_data['pop_relaxed_norm'].min():.4f}, "
                    f"max={pop_data['pop_relaxed_norm'].max():.4f}, "
                    f"sum={pop_data['pop_relaxed_norm'].sum():.4f}\n")
            f.write(f"# Norm summary (>{thresh_s}): "
                    f"min={pop_data['pop_strict_norm'].min():.4f}, "
                    f"max={pop_data['pop_strict_norm'].max():.4f}, "
                    f"sum={pop_data['pop_strict_norm'].sum():.4f}\n")


def _compute_worst_pka_deviation(
    fit_results: dict,
    patch_info: "pd.DataFrame",
) -> float:
    """Compute the worst pKa deviation from theoretical values.

    Args:
        fit_results: Dictionary of {site_id: FitResult} from check_pka_convergence.
        patch_info: DataFrame with patch info including TAG column with theo pKa.

    Returns:
        Maximum absolute deviation across all sites.
    """
    worst_dev = 0.0
    for site_id, result in fit_results.items():
        site_patches = patch_info[patch_info["site"] == site_id]
        for _, row in site_patches.iterrows():
            tag = str(row.get("TAG", "NONE")).strip().upper()
            parts = tag.split()
            if len(parts) >= 2:
                try:
                    theo_pKa = float(parts[1])
                    dev = abs(result.fitted_pKa - theo_pKa)
                    worst_dev = max(worst_dev, dev)
                except ValueError:
                    pass
    return worst_dev


def check_phase_transition(
    current_phase: int,
    lambda_data: np.ndarray,
    config: PhaseTransitionConfig | None = None,
    *,
    # Optional CpHMD parameters for pKa convergence check
    data_dir: Path | None = None,
    patch_info: "pd.DataFrame | None" = None,
    effective_pH: float | None = None,
    delta_pKa: float | None = None,
    nreps: int | None = None,
    nsubs: list[int] | None = None,
    connectivity: float | None = None,
    phase2_run_count: int | None = None,
    ewbs_state: EWBSState | None = None,
) -> tuple[int, str]:
    """Check if phase transition criteria are met.

    For phase transitions, these criteria must be satisfied:
    1. Good overlap (spread within tolerance) — checked per site
    2. Enough samples per state
    3. pKa convergence (if CpHMD parameters provided)
    4. Minimum transition connectivity (for 1→2 and 2→3, if provided)
    5. Per-state minimum fraction at strict threshold (for 2→3)
    6. Minimum Phase 2 duration (for 2→3)

    Args:
        current_phase: Current simulation phase (1, 2, or 3)
        lambda_data: Combined lambda data array (samples x states)
        config: Phase transition configuration (uses defaults if None)
        data_dir: Path to data/ directory (for pKa check)
        patch_info: DataFrame with patch info (for pKa check)
        effective_pH: Base pH for central replica (for pKa check)
        delta_pKa: pH increment between replicas (for pKa check)
        nreps: Number of replicas (for pKa check)
        nsubs: Number of substates per site (e.g. [2, 3]). If None,
               uses global spread check (legacy behavior).
        connectivity: Transition matrix connectivity metric.
        phase2_run_count: Number of runs completed in Phase 2 (for min duration).
        ewbs_state: Optional EWBSState for bias stability gate (for 2→3).

    Returns:
        Tuple of (new_phase, reason_string)
    """
    if config is None:
        config = PhaseTransitionConfig()

    if lambda_data is None or lambda_data.size == 0:
        return current_phase, "No lambda data available"

    # Compute occupancy mask and fractions
    mask = lambda_data > config.lambda_threshold
    col_fracs = mask.mean(axis=0)
    spread = float(np.max(col_fracs) - np.min(col_fracs))
    min_sample_count = int(mask.sum(axis=0).min())

    # Check if CpHMD parameters provided for pKa convergence
    cphmd_params_available = all([
        data_dir is not None,
        patch_info is not None,
        effective_pH is not None,
        delta_pKa is not None,
        nreps is not None and nreps > 3,
    ])

    # Phase 1 → 2 transition
    if current_phase == 1:
        overlap_ok = good_overlap(col_fracs, config.spread_1to2, nsubs=nsubs)
        samples_ok = enough_samples(mask, config.min_hits_1to2)

        # Check pKa convergence if CpHMD
        pka_ok = True
        pka_reason = ""
        if cphmd_params_available:
            pka_ok, fit_results = check_pka_convergence(
                data_dir,
                patch_info,
                effective_pH,
                delta_pKa,
                nreps,
                tolerance=config.pka_tolerance_1to2,
            )
            if not pka_ok:
                worst_dev = _compute_worst_pka_deviation(fit_results, patch_info)
                pka_reason = f"pKa_dev={worst_dev:.2f}>{config.pka_tolerance_1to2}"

        # Check transition connectivity (requires actual within-run transitions)
        conn_ok = True
        conn_reason = ""
        if connectivity is not None:
            conn_ok = connectivity >= config.min_connectivity_1to2
            if not conn_ok:
                conn_reason = (f"connectivity={connectivity:.2f}"
                               f"<{config.min_connectivity_1to2}")

        if overlap_ok and samples_ok and pka_ok and conn_ok:
            return 2, f"Phase 1→2: spread={spread:.3f}, min_hits={min_sample_count}"
        else:
            reasons = []
            if not overlap_ok:
                reasons.append(f"spread={spread:.3f}>{config.spread_1to2}")
            if not samples_ok:
                reasons.append(f"min_hits={min_sample_count}<{config.min_hits_1to2}")
            if not pka_ok:
                reasons.append(pka_reason)
            if not conn_ok:
                reasons.append(conn_reason)
            return 1, f"Staying in phase 1: {', '.join(reasons)}"

    # Phase 2 → 3 transition
    elif current_phase == 2:
        overlap_ok = good_overlap(col_fracs, config.spread_2to3, nsubs=nsubs)
        samples_ok = enough_samples(mask, config.min_hits_2to3)

        # Check pKa convergence if CpHMD (stricter tolerance)
        pka_ok = True
        pka_reason = ""
        if cphmd_params_available:
            pka_ok, fit_results = check_pka_convergence(
                data_dir,
                patch_info,
                effective_pH,
                delta_pKa,
                nreps,
                tolerance=config.pka_tolerance_2to3,
            )
            if not pka_ok:
                worst_dev = _compute_worst_pka_deviation(fit_results, patch_info)
                pka_reason = f"pKa_dev={worst_dev:.2f}>{config.pka_tolerance_2to3}"

        # Check transition connectivity (prevents premature transition)
        conn_ok = True
        conn_reason = ""
        if connectivity is not None:
            conn_ok = connectivity >= config.min_connectivity_2to3
            if not conn_ok:
                conn_reason = (f"connectivity={connectivity:.2f}"
                               f"<{config.min_connectivity_2to3}")

        # Check per-state minimum fraction at strict threshold
        # Prevents 2→3 when any state has near-zero physical-state population
        state_frac_ok = True
        state_frac_reason = ""
        if nsubs is not None:
            strict_mask = lambda_data > config.strict_threshold_2to3
            strict_fracs = strict_mask.mean(axis=0)
            for si, (start, end) in enumerate(_per_site_ranges(nsubs)):
                site_raw = strict_fracs[start:end]
                site_total = site_raw.sum()
                if site_total > 0:
                    site_norm = site_raw / site_total
                else:
                    site_norm = np.zeros_like(site_raw)
                min_frac = float(site_norm.min())
                if min_frac < config.min_state_fraction_2to3:
                    state_frac_ok = False
                    worst_sub = int(np.argmin(site_norm))
                    state_frac_reason = (
                        f"site{si} sub{worst_sub} frac={min_frac:.1%}"
                        f"<{config.min_state_fraction_2to3:.0%}")
                    break

        # Check minimum Phase 2 duration
        phase2_dur_ok = True
        phase2_dur_reason = ""
        if phase2_run_count is not None:
            phase2_dur_ok = phase2_run_count >= config.min_phase2_runs
            if not phase2_dur_ok:
                phase2_dur_reason = (
                    f"phase2_runs={phase2_run_count}"
                    f"<{config.min_phase2_runs}")

        # Check EWBS bias stability
        ewbs_ok = True
        ewbs_reason = ""
        if ewbs_state is not None:
            if not ewbs_state.history:
                ewbs_ok = False
                ewbs_reason = "ewbs: no history available"
            else:
                window = min(config.ewbs_2to3_window, len(ewbs_state.history))
                recent = ewbs_state.history[-window:]
                if window < config.ewbs_2to3_window:
                    ewbs_ok = False
                    ewbs_reason = (
                        f"ewbs_history={window}<{config.ewbs_2to3_window}")
                elif any(v > config.ewbs_2to3 for v in recent):
                    ewbs_ok = False
                    btn = ewbs_bottleneck_type(ewbs_state)
                    ewbs_reason = (
                        f"ewbs={ewbs_state.ewbs:.3f}>{config.ewbs_2to3}({btn})")

        all_ok = (overlap_ok and samples_ok and pka_ok and conn_ok
                  and state_frac_ok and phase2_dur_ok and ewbs_ok)
        if all_ok:
            return 3, f"Phase 2→3: spread={spread:.3f}, min_hits={min_sample_count}"
        else:
            reasons = []
            if not overlap_ok:
                reasons.append(f"spread={spread:.3f}>{config.spread_2to3}")
            if not samples_ok:
                reasons.append(f"min_hits={min_sample_count}<{config.min_hits_2to3}")
            if not pka_ok:
                reasons.append(pka_reason)
            if not conn_ok:
                reasons.append(conn_reason)
            if not state_frac_ok:
                reasons.append(state_frac_reason)
            if not phase2_dur_ok:
                reasons.append(phase2_dur_reason)
            if not ewbs_ok:
                reasons.append(ewbs_reason)
            return 2, f"Staying in phase 2: {', '.join(reasons)}"

    # Phase 3 - no further transitions
    return 3, "Already in phase 3 (production)"


def check_phase3_stop(
    lambda_data: np.ndarray,
    stop_config: StopCriteriaConfig | None = None,
    bias_history: np.ndarray | None = None,
    nsubs: list[int] | None = None,
    ewbs_state: EWBSState | None = None,
) -> tuple[bool, str, StopCriteriaResult]:
    """Check if Phase 3 simulation should stop.

    This is the main entry point for checking stop criteria in Phase 3.
    Uses step5_bias_search-style scoring based on population balance.

    Args:
        lambda_data: Combined lambda data array (samples x states)
        stop_config: Stop criteria configuration (uses defaults if None)
        bias_history: Optional array of bias values for stability check
                      Shape: (n_iterations, n_bias_params)
        nsubs: Number of substates per site (e.g. [2, 3]). If None,
               uses global metrics (legacy behavior).
        ewbs_state: Optional EWBSState for bias stability metric.

    Returns:
        Tuple of (should_stop, reason_string, full_result)

    Example:
        >>> config = StopCriteriaConfig(timestep_fs=4.0, max_frac_diff=0.02)
        >>> should_stop, reason, result = check_phase3_stop(lambda_data, config)
        >>> if should_stop:
        ...     print(f"Converged: diff={result.frac_diff_pct:.2f}%")
    """
    result = check_stop_criteria(
        lambda_data, stop_config, bias_history, nsubs=nsubs,
        ewbs_state=ewbs_state,
    )

    if result.should_stop:
        ewbs_info = ""
        if result.ewbs < float("inf"):
            ewbs_info = f", ewbs={result.ewbs:.4f}({result.ewbs_bottleneck})"
        reason = (
            f"STOP: converged with {result.n_states} states, "
            f"samples={result.n_samples:,}, "
            f"frac_diff={result.frac_diff_pct:.2f}%, "
            f"entropy={result.entropy_normalized:.2f}, "
            f"block_var={result.block_variance:.4f}{ewbs_info}, "
            f"score={result.score:.4f}"
        )
    else:
        reason = f"Continue Phase 3: {', '.join(result.reasons)}"

    return result.should_stop, reason, result


def _hh_acidic(pH: np.ndarray, pKa: float, n: float = 1.0) -> np.ndarray:
    """Henderson-Hasselbalch for acidic residues (population of deprotonated state)."""
    return 1.0 / (1.0 + 10**(n * (pKa - pH)))


def _hh_basic(pH: np.ndarray, pKa: float, n: float = 1.0) -> np.ndarray:
    """Henderson-Hasselbalch for basic residues (population of deprotonated state)."""
    return 1.0 / (1.0 + 10**(n * (pH - pKa)))


def fit_pka_from_replicas(
    replica_data: list[ReplicaLambdaData],
    fit_type: str = "auto",
    state_indices: list[int] | None = None,
    lambda_threshold: float = 0.8,
) -> PKaFitResult:
    """Fit pKa from multi-replica population data.

    This is the core fitting function that takes per-replica lambda data,
    computes populations, and fits a Henderson-Hasselbalch curve.

    Args:
        replica_data: List of ReplicaLambdaData with per-replica lambda values
        fit_type: "acidic", "basic", "auto" to detect from trend
        state_indices: Which states to sum for population (e.g., charged states)
        lambda_threshold: Threshold for counting samples

    Returns:
        PKaFitResult with fitted parameters
    """
    if len(replica_data) < 3:
        return PKaFitResult(
            fitted_pKa=7.0,
            fit_type="insufficient_data",
            n_points=len(replica_data),
        )

    # Compute populations for specified states
    replica_data = compute_replica_populations(
        replica_data, lambda_threshold, state_indices
    )

    # Extract pH and population arrays
    pH_values = np.array([rd.pH for rd in replica_data])
    populations = np.array([
        rd.populations[0] if len(rd.populations) > 0 else 0.0
        for rd in replica_data
    ])

    # Sort by pH
    sort_idx = np.argsort(pH_values)
    pH_values = pH_values[sort_idx]
    populations = populations[sort_idx]

    # Auto-detect fit type from population trend
    if fit_type == "auto":
        trend = populations[-1] - populations[0]
        if trend > 0.2:
            fit_type = "acidic"  # Population rises with pH → deprotonation
        elif trend < -0.2:
            fit_type = "basic"  # Population falls with pH → protonation loss
        else:
            # Nearly constant - can't determine pKa reliably
            return PKaFitResult(
                fitted_pKa=np.mean(pH_values),
                fit_type="constant",
                n_points=len(pH_values),
                pH_values=pH_values,
                populations=populations,
            )

    # Fit HH curve
    try:
        if fit_type == "acidic":
            fit_func = _hh_acidic
        else:
            fit_func = _hh_basic

        # Initial pKa guess from midpoint
        mid_pop = 0.5
        closest_idx = np.argmin(np.abs(populations - mid_pop))
        initial_pKa = pH_values[closest_idx]

        popt, _ = curve_fit(
            fit_func,
            pH_values,
            populations,
            p0=[initial_pKa, 1.0],
            bounds=([0, 0.1], [14, 5]),
            maxfev=5000,
        )

        # Calculate R²
        y_pred = fit_func(pH_values, *popt)
        ss_res = np.sum((populations - y_pred)**2)
        ss_tot = np.sum((populations - np.mean(populations))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return PKaFitResult(
            fitted_pKa=popt[0],
            fit_type=fit_type,
            r_squared=r_squared,
            n_points=len(pH_values),
            pH_values=pH_values,
            populations=populations,
        )

    except Exception:
        # Fitting failed - return estimate from midpoint
        return PKaFitResult(
            fitted_pKa=np.mean(pH_values),
            fit_type="fit_failed",
            n_points=len(pH_values),
            pH_values=pH_values,
            populations=populations,
        )


def check_pka_convergence(
    data_dir: Path,
    patch_info: "pd.DataFrame",
    effective_pH: float,
    delta_pKa: float,
    nreps: int,
    tolerance: float = 1.5,
    ncentral: int | None = None,
    lambda_threshold: float = 0.8,
) -> tuple[bool, dict[str, PKaFitResult]]:
    """Check if fitted pKa values are within tolerance of theoretical values.

    This is the full pKa convergence check for phase transitions. It:
    1. Loads per-replica lambda data
    2. Groups substates by titratable site
    3. Fits HH curves to get per-site pKa values
    4. Compares fitted vs theoretical pKa (from TAG field)

    Args:
        data_dir: Path to data/ directory with Lambda files
        patch_info: DataFrame with patch info including TAG column
        effective_pH: Base pH for central replica
        delta_pKa: pH increment between replicas
        nreps: Number of replicas
        tolerance: Maximum allowed pKa deviation
        ncentral: Central replica index (defaults to nreps // 2)
        lambda_threshold: Threshold for state occupancy

    Returns:
        Tuple of (converged: bool, fit_results: dict mapping site_id to PKaFitResult)
    """
    if ncentral is None:
        ncentral = nreps // 2

    # Load per-replica data
    replica_data = load_lambda_data_per_replica(
        data_dir, effective_pH, delta_pKa, nreps, ncentral
    )

    if len(replica_data) < 3:
        # Not enough replicas for reliable fitting
        return True, {}

    fit_results: dict[str, PKaFitResult] = {}
    all_converged = True

    # Group patches by site
    if "site" not in patch_info.columns:
        return True, {}

    for site_id in patch_info["site"].unique():
        site_patches = patch_info[patch_info["site"] == site_id]

        # Determine theoretical pKa from TAG values
        theoretical_pKa = None
        site_type = "acidic"  # Default
        pKa_values = []

        for _, row in site_patches.iterrows():
            tag = str(row.get("TAG", "NONE")).strip().upper()
            parts = tag.split()
            if len(parts) >= 2:
                try:
                    pKa = float(parts[1])
                    pKa_values.append(pKa)
                    if parts[0] == "UPOS":
                        site_type = "basic"
                    elif parts[0] == "UNEG":
                        site_type = "acidic"
                except ValueError:
                    pass

        if pKa_values:
            theoretical_pKa = np.mean(pKa_values)

        # Get state indices for this site (charged states for fitting)
        state_indices = []
        for idx, (_, row) in enumerate(site_patches.iterrows()):
            tag = str(row.get("TAG", "NONE")).strip().upper()
            if tag.startswith("UPOS") or tag.startswith("UNEG"):
                state_indices.append(idx)

        if not state_indices:
            continue

        # Fit pKa from replica data
        fit_result = fit_pka_from_replicas(
            replica_data,
            fit_type=site_type,
            state_indices=state_indices,
            lambda_threshold=lambda_threshold,
        )
        fit_results[str(site_id)] = fit_result

        # Check convergence
        if theoretical_pKa is not None and fit_result.fit_type not in ("constant", "insufficient_data", "fit_failed"):
            deviation = abs(fit_result.fitted_pKa - theoretical_pKa)
            if deviation > tolerance:
                all_converged = False

    return all_converged, fit_results


def check_pka_convergence_simple(
    fitted_pKa: float,
    theoretical_pKas: list[float],
    tolerance: float = 1.5,
) -> bool:
    """Simple check if fitted pKa is within tolerance of theoretical values.

    This is the legacy interface - use check_pka_convergence() for full analysis.

    Args:
        fitted_pKa: The pKa value obtained from HH curve fitting
        theoretical_pKas: List of theoretical/reference pKa values
        tolerance: Maximum allowed deviation from any theoretical value

    Returns:
        True if fitted pKa is within tolerance of all theoretical values
    """
    if not theoretical_pKas:
        return True  # No reference values, assume converged

    return all(abs(fitted_pKa - pKa) <= tolerance for pKa in theoretical_pKas)
