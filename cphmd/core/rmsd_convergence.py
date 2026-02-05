"""RMSD-based convergence for phase transitions and auto-stop.

Uses precomputed G-file free energy profiles from WHAM to detect
when the bias landscape stabilizes. Alternative to population-based
criteria for large multi-site systems (5+ substates per site).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import json
import numpy as np

PhaseType = Literal[1, 2, 3]


@dataclass
class RMSDConvergenceConfig:
    """Configuration for RMSD-based convergence detection."""
    lag: int = 5  # Compare run N with run N-lag
    window: int = 3  # Rolling mean window for plateau detection
    plateau_1to2: float = 0.30  # Phase 1→2: RMSD drops below 30% of peak
    plateau_2to3: float = 0.10  # Phase 2→3: RMSD drops below 10% of Phase 2 peak
    stop_threshold: float = 0.10  # Auto-stop: below 10% of Phase 2 entry RMSD
    min_coverage_1to2: float = 0.20  # Minimum coverage fraction for Phase 1→2
    min_coverage_2to3: float = 0.50  # Minimum coverage fraction for Phase 2→3


@dataclass
class RMSDState:
    """Persistent state for RMSD convergence tracking."""
    rmsd_history: list[list[float]] = field(default_factory=list)  # [run][site]
    coverage_history: list[list[float]] = field(default_factory=list)
    run_indices: list[int] = field(default_factory=list)  # Which run each entry corresponds to
    phase2_peak_rmsd: list[float] | None = None  # Per-site peak RMSD in Phase 2
    phase2_entry_rmsd: list[float] | None = None  # Per-site RMSD at Phase 2 start

    def save(self, path: Path) -> None:
        data = {
            "rmsd_history": self.rmsd_history,
            "coverage_history": self.coverage_history,
            "run_indices": self.run_indices,
            "phase2_peak_rmsd": self.phase2_peak_rmsd,
            "phase2_entry_rmsd": self.phase2_entry_rmsd,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RMSDState":
        data = json.loads(path.read_text())
        return cls(
            rmsd_history=data["rmsd_history"],
            coverage_history=data["coverage_history"],
            run_indices=data["run_indices"],
            phase2_peak_rmsd=data.get("phase2_peak_rmsd"),
            phase2_entry_rmsd=data.get("phase2_entry_rmsd"),
        )


def _simplex_valid_count(data_size: int) -> int:
    """Count grid points where λᵢ + λⱼ ≤ 1 on a square grid.

    For a grid_size × grid_size grid with λ = k/(grid_size-1),
    valid points satisfy i + j ≤ grid_size - 1.

    Args:
        data_size: Total number of grid points (grid_size²).

    Returns:
        Number of physically accessible points on the simplex.
    """
    gs = int(round(data_size ** 0.5))
    return gs * (gs + 1) // 2


def _rmsd_finite(
    a: np.ndarray, b: np.ndarray, n_valid: int | None = None
) -> tuple[float, float]:
    """RMSD over points finite in both arrays.

    Args:
        a: First array.
        b: Second array.
        n_valid: Denominator for coverage fraction. If None, uses len(a).
            Use _simplex_valid_count() for pairwise profiles to exclude
            physically inaccessible grid points (λᵢ + λⱼ > 1).

    Returns:
        (rmsd, coverage_fraction)
    """
    mask = np.isfinite(a) & np.isfinite(b)
    denom = n_valid if n_valid is not None else len(a)
    coverage = mask.sum() / denom if denom > 0 else 0.0
    if mask.sum() == 0:
        return float("inf"), coverage
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff * diff))), coverage


def compute_site_rmsd(
    analysis_dir: Path,
    ref_dir: Path,
    nsubs: list[int],
) -> tuple[list[float], list[float]]:
    """Compute per-site RMSD from G-file profiles between two analysis dirs.

    Reads multisite/G{n}.dat files and computes RMSD over individual
    substituent and pairwise profiles for each site.

    Args:
        analysis_dir: Current analysis directory
        ref_dir: Reference analysis directory (lag iterations ago)
        nsubs: Number of substates per site

    Returns:
        (site_rmsds, site_coverages) — per-site RMSD and coverage fraction
    """
    ms_curr = analysis_dir / "multisite"
    ms_ref = ref_dir / "multisite"

    if not ms_curr.is_dir() or not ms_ref.is_dir():
        return [float("inf")] * len(nsubs), [0.0] * len(nsubs)

    site_rmsds = []
    site_coverages = []
    iG = 1

    for ns in nsubs:
        rmsds = []
        coverages = []

        # Individual substituent profiles
        for _ in range(ns):
            g_curr_path = ms_curr / f"G{iG}.dat"
            g_ref_path = ms_ref / f"G{iG}.dat"
            if g_curr_path.exists() and g_ref_path.exists():
                g_curr = np.loadtxt(g_curr_path)
                g_ref = np.loadtxt(g_ref_path)
                rmsd, cov = _rmsd_finite(g_curr, g_ref)
                rmsds.append(rmsd)
                coverages.append(cov)
            else:
                rmsds.append(float("inf"))
                coverages.append(0.0)
            iG += 1

        # Pairwise profiles — use simplex-aware coverage denominator
        n_pairs = ns * (ns - 1) // 2
        for _ in range(n_pairs):
            g_curr_path = ms_curr / f"G{iG}.dat"
            g_ref_path = ms_ref / f"G{iG}.dat"
            if g_curr_path.exists() and g_ref_path.exists():
                g_curr = np.loadtxt(g_curr_path)
                g_ref = np.loadtxt(g_ref_path)
                n_valid = _simplex_valid_count(len(g_curr))
                rmsd, cov = _rmsd_finite(g_curr, g_ref, n_valid)
                rmsds.append(rmsd)
                coverages.append(cov)
            else:
                rmsds.append(float("inf"))
                coverages.append(0.0)
            iG += 1

        # Skip higher-order terms (advance counter only)
        if ns > 2:
            iG += ns * (ns - 1) // 2

        # Aggregate: RMS of individual RMSDs, mean coverage
        finite_rmsds = [r for r in rmsds if np.isfinite(r)]
        if finite_rmsds:
            site_rmsds.append(float(np.sqrt(np.mean(np.array(finite_rmsds) ** 2))))
        else:
            site_rmsds.append(float("inf"))
        site_coverages.append(float(np.mean(coverages)))

    return site_rmsds, site_coverages


@dataclass
class PairwiseRMSD:
    """Per-profile RMSD decomposition for one site."""
    ns: int  # Number of substates
    individual: list[tuple[int, float, float]]  # [(sub_idx, rmsd, coverage), ...]
    pairwise: list[tuple[int, int, float, float]]  # [(i, j, rmsd, coverage), ...]


def compute_pairwise_rmsd(
    analysis_dir: Path,
    ref_dir: Path,
    nsubs: list[int],
) -> list[PairwiseRMSD]:
    """Compute per-profile RMSD decomposed by substate and pair.

    Args:
        analysis_dir: Current analysis directory
        ref_dir: Reference analysis directory (lag iterations ago)
        nsubs: Number of substates per site

    Returns:
        List of PairwiseRMSD, one per site
    """
    ms_curr = analysis_dir / "multisite"
    ms_ref = ref_dir / "multisite"

    if not ms_curr.is_dir() or not ms_ref.is_dir():
        return [PairwiseRMSD(ns=ns, individual=[], pairwise=[]) for ns in nsubs]

    results = []
    iG = 1

    for ns in nsubs:
        individual = []
        pairwise = []

        # Individual substituent profiles
        for s in range(ns):
            g_curr_path = ms_curr / f"G{iG}.dat"
            g_ref_path = ms_ref / f"G{iG}.dat"
            if g_curr_path.exists() and g_ref_path.exists():
                rmsd, cov = _rmsd_finite(np.loadtxt(g_curr_path), np.loadtxt(g_ref_path))
            else:
                rmsd, cov = float("inf"), 0.0
            individual.append((s, rmsd, cov))
            iG += 1

        # Pairwise profiles (upper triangle: i < j)
        # Use simplex-aware coverage: only count grid points where λᵢ + λⱼ ≤ 1
        for i in range(ns):
            for j in range(i + 1, ns):
                g_curr_path = ms_curr / f"G{iG}.dat"
                g_ref_path = ms_ref / f"G{iG}.dat"
                if g_curr_path.exists() and g_ref_path.exists():
                    g_curr = np.loadtxt(g_curr_path)
                    g_ref = np.loadtxt(g_ref_path)
                    n_valid = _simplex_valid_count(len(g_curr))
                    rmsd, cov = _rmsd_finite(g_curr, g_ref, n_valid)
                else:
                    rmsd, cov = float("inf"), 0.0
                pairwise.append((i, j, rmsd, cov))
                iG += 1

        # Skip higher-order terms
        if ns > 2:
            iG += ns * (ns - 1) // 2

        results.append(PairwiseRMSD(ns=ns, individual=individual, pairwise=pairwise))

    return results


def check_rmsd_phase_transition(
    phase: PhaseType,
    rmsd_state: RMSDState,
    config: RMSDConvergenceConfig | None = None,
) -> tuple[PhaseType, str]:
    """Check if RMSD plateau warrants a phase transition.

    Args:
        phase: Current phase
        rmsd_state: Accumulated RMSD history
        config: Convergence configuration

    Returns:
        (new_phase, reason) — new_phase == phase means no transition
    """
    if config is None:
        config = RMSDConvergenceConfig()

    n = len(rmsd_state.rmsd_history)
    if n < config.window + 1:
        return phase, f"RMSD: need {config.window + 1} data points, have {n}"

    n_sites = len(rmsd_state.rmsd_history[0])

    # Compute rolling mean over last `window` entries
    recent = np.array(rmsd_state.rmsd_history[-config.window:])  # (window, n_sites)
    recent_cov = np.array(rmsd_state.coverage_history[-config.window:])
    rolling_rmsd = np.mean(recent, axis=0)  # per-site mean
    rolling_cov = np.mean(recent_cov, axis=0)

    if phase == 1:
        # Check for Phase 1→2: rolling mean < 30% of peak
        all_rmsds = np.array(rmsd_state.rmsd_history)  # (n, n_sites)
        # Rolling means for peak detection
        if n >= config.window:
            all_rolling = np.array([
                np.mean(all_rmsds[max(0, i - config.window + 1):i + 1], axis=0)
                for i in range(n)
            ])
            peak_rmsd = np.max(all_rolling, axis=0)  # per-site peak
        else:
            peak_rmsd = np.max(all_rmsds, axis=0)

        # Per-site checks
        reasons = []
        all_ok = True
        for s in range(n_sites):
            if not np.isfinite(rolling_rmsd[s]) or peak_rmsd[s] == 0:
                reasons.append(f"s{s + 1}: no finite RMSD")
                all_ok = False
                continue
            ratio = rolling_rmsd[s] / peak_rmsd[s]
            cov_ok = rolling_cov[s] >= config.min_coverage_1to2
            plateau_ok = ratio < config.plateau_1to2
            if not cov_ok:
                reasons.append(f"s{s + 1}: cov={rolling_cov[s]:.0%}<{config.min_coverage_1to2:.0%}")
                all_ok = False
            elif not plateau_ok:
                reasons.append(f"s{s + 1}: ratio={ratio:.2f}>={config.plateau_1to2}")
                all_ok = False
            else:
                reasons.append(f"s{s + 1}: OK ratio={ratio:.2f} cov={rolling_cov[s]:.0%}")

        reason = "RMSD 1→2: " + ", ".join(reasons)
        return (2, reason) if all_ok else (1, reason)

    elif phase == 2:
        # Check for Phase 2→3: rolling mean < 10% of Phase 2 peak
        if rmsd_state.phase2_peak_rmsd is None:
            # First check in Phase 2 — initialize peak
            rmsd_state.phase2_peak_rmsd = rolling_rmsd.tolist()
            rmsd_state.phase2_entry_rmsd = rolling_rmsd.tolist()
            return 2, f"RMSD 2→3: Phase 2 baseline set ({rolling_rmsd})"

        # Update peak
        for s in range(n_sites):
            if np.isfinite(rolling_rmsd[s]):
                rmsd_state.phase2_peak_rmsd[s] = max(
                    rmsd_state.phase2_peak_rmsd[s], rolling_rmsd[s]
                )

        reasons = []
        all_ok = True
        for s in range(n_sites):
            peak = rmsd_state.phase2_peak_rmsd[s]
            if not np.isfinite(rolling_rmsd[s]) or peak == 0:
                reasons.append(f"s{s + 1}: no finite RMSD")
                all_ok = False
                continue
            ratio = rolling_rmsd[s] / peak
            cov_ok = rolling_cov[s] >= config.min_coverage_2to3
            plateau_ok = ratio < config.plateau_2to3
            if not cov_ok:
                reasons.append(f"s{s + 1}: cov={rolling_cov[s]:.0%}<{config.min_coverage_2to3:.0%}")
                all_ok = False
            elif not plateau_ok:
                reasons.append(f"s{s + 1}: ratio={ratio:.2f}>={config.plateau_2to3}")
                all_ok = False
            else:
                reasons.append(f"s{s + 1}: OK ratio={ratio:.2f} cov={rolling_cov[s]:.0%}")

        reason = "RMSD 2→3: " + ", ".join(reasons)
        return (3, reason) if all_ok else (2, reason)

    return phase, f"RMSD: Phase {phase}, no transition logic"


def check_rmsd_stop(
    rmsd_state: RMSDState,
    config: RMSDConvergenceConfig | None = None,
) -> tuple[bool, str]:
    """Check if Phase 3 RMSD warrants auto-stop.

    Args:
        rmsd_state: Accumulated RMSD history
        config: Convergence configuration

    Returns:
        (should_stop, reason)
    """
    if config is None:
        config = RMSDConvergenceConfig()

    if rmsd_state.phase2_entry_rmsd is None:
        return False, "RMSD stop: no Phase 2 baseline"

    n = len(rmsd_state.rmsd_history)
    if n < config.window:
        return False, f"RMSD stop: need {config.window} data points, have {n}"

    recent = np.array(rmsd_state.rmsd_history[-config.window:])
    rolling_rmsd = np.mean(recent, axis=0)

    reasons = []
    all_ok = True
    for s in range(len(rolling_rmsd)):
        baseline = rmsd_state.phase2_entry_rmsd[s]
        if baseline == 0 or not np.isfinite(baseline):
            reasons.append(f"s{s + 1}: no baseline")
            all_ok = False
            continue
        ratio = rolling_rmsd[s] / baseline
        if ratio < config.stop_threshold:
            reasons.append(f"s{s + 1}: OK ratio={ratio:.3f}")
        else:
            reasons.append(f"s{s + 1}: ratio={ratio:.3f}>={config.stop_threshold}")
            all_ok = False

    reason = "RMSD stop: " + ", ".join(reasons)
    return all_ok, reason
