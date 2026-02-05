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
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition criteria."""

    # Lambda threshold for counting "occupied" states
    lambda_threshold: float = 0.8

    # Spread tolerance for phase 1→2 transition
    spread_1to2: float = 0.3

    # Spread tolerance for phase 2→3 transition
    spread_2to3: float = 0.2

    # Minimum sample count per state
    min_hits: int = 1000

    # pKa tolerance for phase 1→2 (relaxed)
    pka_tolerance_1to2: float = 1.5

    # pKa tolerance for phase 2→3 (strict)
    pka_tolerance_2to3: float = 0.3


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

    Args:
        data_dir: Path to data/ directory containing Lambda.*.*.dat files

    Returns:
        Tuple of (combined lambda data array, list of representative files)
    """
    if not data_dir.exists():
        return None, []

    # Find all lambda files and organize by replica
    replica_files: dict[int, list[str]] = {}

    for fpath in sorted(data_dir.glob("Lambda.*.*.dat")):
        fname = fpath.name
        try:
            parts = fname.split('.')
            if len(parts) >= 4:
                repeat_idx = int(parts[1])
                replica_idx = int(parts[2])

                if replica_idx not in replica_files:
                    replica_files[replica_idx] = []
                replica_files[replica_idx].append(fname)
        except (ValueError, IndexError):
            continue

    if not replica_files:
        return None, []

    # Combine data from all replicas
    lambda_data = None
    l_files: list[str] = []

    for replica_idx in sorted(replica_files.keys()):
        repeat_files = sorted(replica_files[replica_idx])

        replica_combined = None
        for fname in repeat_files:
            l = np.loadtxt(data_dir / fname)
            # Strip time column (column 0) — keep only lambda values
            if l.ndim == 2 and l.shape[1] > 1:
                l = l[:, 1:]
            if replica_combined is None:
                replica_combined = l
            else:
                replica_combined = np.vstack([replica_combined, l])

        if replica_combined is not None:
            if lambda_data is None:
                lambda_data = replica_combined
            else:
                lambda_data = np.vstack([lambda_data, replica_combined])

            # Keep first repeat file as representative
            l_files.append(repeat_files[0])

    return lambda_data, l_files


def load_lambda_data_per_replica(
    data_dir: Path,
    effective_pH: float,
    delta_pKa: float,
    nreps: int,
    ncentral: int | None = None,
) -> list[ReplicaLambdaData]:
    """Load lambda data organized by replica with per-replica pH values.

    This is the key function for pKa convergence checking. Each replica samples
    a different effective pH based on the replica exchange scheme:
        replica_pH = effective_pH + delta_pKa * (replica_idx - ncentral)

    Args:
        data_dir: Path to data/ directory containing Lambda.*.*.dat files
        effective_pH: Base pH for the central replica
        delta_pKa: pH increment between adjacent replicas
        nreps: Total number of replicas
        ncentral: Index of central replica (defaults to nreps // 2)

    Returns:
        List of ReplicaLambdaData objects, one per replica with valid data
    """
    if not data_dir.exists():
        return []

    if ncentral is None:
        ncentral = nreps // 2

    # Organize files by replica
    replica_files: dict[int, list[Path]] = {}

    for fpath in sorted(data_dir.glob("Lambda.*.*.dat")):
        fname = fpath.name
        try:
            parts = fname.split('.')
            if len(parts) >= 4:
                repeat_idx = int(parts[1])
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

        # Compute pH for this replica
        replica_pH = effective_pH + delta_pKa * (replica_idx - ncentral)

        # Combine all repeat files for this replica
        replica_lambda = None
        for fpath in sorted(replica_files[replica_idx]):
            try:
                data = np.loadtxt(fpath)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                # Strip time column (column 0) — keep only lambda values
                if data.shape[1] > 1:
                    data = data[:, 1:]
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

    # Determine if all criteria are met
    reasons: list[str] = []

    min_samples = config.min_samples()
    if n_samples < min_samples:
        reasons.append(f"samples={n_samples:,}<{min_samples:,}")

    if frac_diff > config.max_frac_diff:
        reasons.append(f"frac_diff={frac_diff_pct:.2f}%>{config.max_frac_diff*100:.1f}%")

    if not bias_stable:
        reasons.append(f"bias_std={bias_rolling_std:.3f}>{config.bias_max_std}")

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
        f.write(f"# Lambda Population Analysis\n")
        f.write(f"# Total samples: {n_samples}\n")
        if nsubs is not None:
            f.write(f"# Sites: {len(nsubs)}, nsubs={nsubs}\n")
        f.write(f"# Relaxed threshold: {thresh_r} (total physical: {total_r})\n")
        f.write(f"# Strict threshold: {thresh_s} (total physical: {total_s})\n")
        f.write(f"#\n")
        f.write(f"# Raw = fraction of ALL samples in physical state\n")
        if nsubs is not None:
            f.write(f"# Norm = fraction among physical samples per site (each site sums to 1)\n")
        else:
            f.write(f"# Norm = fraction among physical samples only (sums to 1)\n")
        f.write(f"#\n")

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
        f.write(f"#\n")
        f.write(f"# Raw summary (>{thresh_r}): "
                f"min={pop_data['pop_relaxed_raw'].min():.4f}, "
                f"max={pop_data['pop_relaxed_raw'].max():.4f}, "
                f"spread={pop_data['pop_relaxed_raw'].max() - pop_data['pop_relaxed_raw'].min():.4f}\n")
        f.write(f"# Raw summary (>{thresh_s}): "
                f"min={pop_data['pop_strict_raw'].min():.4f}, "
                f"max={pop_data['pop_strict_raw'].max():.4f}, "
                f"spread={pop_data['pop_strict_raw'].max() - pop_data['pop_strict_raw'].min():.4f}\n")
        f.write(f"#\n")
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
) -> tuple[int, str]:
    """Check if phase transition criteria are met.

    For phase transitions, three criteria must be satisfied:
    1. Good overlap (spread within tolerance) — checked per site
    2. Enough samples per state
    3. pKa convergence (if CpHMD parameters provided)

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
        samples_ok = enough_samples(mask, config.min_hits)

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
                # Find the worst deviation
                worst_dev = 0.0
                for site_id, result in fit_results.items():
                    # Get theoretical from patch_info
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
                pka_reason = f"pKa_dev={worst_dev:.2f}>{config.pka_tolerance_1to2}"

        if overlap_ok and samples_ok and pka_ok:
            return 2, f"Phase 1→2: spread={spread:.3f}, min_hits={min_sample_count}"
        else:
            reasons = []
            if not overlap_ok:
                reasons.append(f"spread={spread:.3f}>{config.spread_1to2}")
            if not samples_ok:
                reasons.append(f"min_hits={min_sample_count}<{config.min_hits}")
            if not pka_ok:
                reasons.append(pka_reason)
            return 1, f"Staying in phase 1: {', '.join(reasons)}"

    # Phase 2 → 3 transition
    elif current_phase == 2:
        overlap_ok = good_overlap(col_fracs, config.spread_2to3, nsubs=nsubs)
        samples_ok = enough_samples(mask, config.min_hits)

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
                pka_reason = f"pKa_dev={worst_dev:.2f}>{config.pka_tolerance_2to3}"

        if overlap_ok and samples_ok and pka_ok:
            return 3, f"Phase 2→3: spread={spread:.3f}, min_hits={min_sample_count}"
        else:
            reasons = []
            if not overlap_ok:
                reasons.append(f"spread={spread:.3f}>{config.spread_2to3}")
            if not samples_ok:
                reasons.append(f"min_hits={min_sample_count}<{config.min_hits}")
            if not pka_ok:
                reasons.append(pka_reason)
            return 2, f"Staying in phase 2: {', '.join(reasons)}"

    # Phase 3 - no further transitions
    return 3, "Already in phase 3 (production)"


def check_phase3_stop(
    lambda_data: np.ndarray,
    stop_config: StopCriteriaConfig | None = None,
    bias_history: np.ndarray | None = None,
    nsubs: list[int] | None = None,
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

    Returns:
        Tuple of (should_stop, reason_string, full_result)

    Example:
        >>> config = StopCriteriaConfig(timestep_fs=4.0, max_frac_diff=0.02)
        >>> should_stop, reason, result = check_phase3_stop(lambda_data, config)
        >>> if should_stop:
        ...     print(f"Converged: diff={result.frac_diff_pct:.2f}%")
    """
    result = check_stop_criteria(lambda_data, stop_config, bias_history, nsubs=nsubs)

    if result.should_stop:
        reason = (
            f"STOP: converged with {result.n_states} states, "
            f"samples={result.n_samples:,}, "
            f"frac_diff={result.frac_diff_pct:.2f}%, "
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
