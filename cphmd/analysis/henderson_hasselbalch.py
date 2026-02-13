"""Henderson-Hasselbalch curve fitting and visualization for CpHMD.

Implements:
- Logistic function fitting with s=+1/-1/0 direction handling
- UPOS/UNEG/NONE microstate weight assignment
- 6 fitting cases for different residue types (including UPOS-only, UNEG-only)
- Multi-replica pH-based titration curve analysis
- Thermodynamic population model using Boltzmann weights
- Site-grouped substate plotting with CSV data export
- Matplotlib plotting with publication-quality output
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

logger = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import pandas as pd

# Microstate types
MicrostateType = Literal["NONE", "UPOS", "UNEG"]


@dataclass
class HHFitResult:
    """Result of Henderson-Hasselbalch curve fitting."""
    pKa_eff: float
    pKa_pos: float | None = None  # For 3-state systems
    pKa_neg: float | None = None  # For 3-state systems
    hill_coeff: float = 1.0
    fit_type: str = "unknown"
    r_squared: float = 0.0


@dataclass
class SubstatePopulation:
    """Population data for a single microstate/substate.

    Attributes:
        state_idx: Index in the lambda array
        select_name: SELECT field value (e.g., "s1s1")
        tag_type: NONE, UPOS, or UNEG
        tag_pKa: pKa value from TAG field (None for NONE states)
        pH_values: Array of pH values
        populations: Array of population fractions at each pH
        theoretical: Array of theoretical populations (from HH model)
    """
    state_idx: int
    select_name: str
    tag_type: str
    tag_pKa: float | None
    pH_values: np.ndarray = field(default_factory=lambda: np.array([]))
    populations: np.ndarray = field(default_factory=lambda: np.array([]))
    theoretical: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SiteHHResult:
    """Complete HH analysis result for a titratable site.

    Attributes:
        site_id: Site identifier
        segid: Segment ID
        resid: Residue ID
        resname: Residue name
        fit_result: HHFitResult from curve fitting
        substates: List of SubstatePopulation for each microstate
        pH_values: Combined pH array used for analysis
        total_populations: Site-level total populations (charged states summed)
    """
    site_id: str
    segid: str
    resid: str
    resname: str
    fit_result: HHFitResult
    substates: list[SubstatePopulation] = field(default_factory=list)
    pH_values: np.ndarray = field(default_factory=lambda: np.array([]))
    total_populations: np.ndarray = field(default_factory=lambda: np.array([]))


def logistic(pH: np.ndarray, pKa: float, s: int, P: float = 1.0) -> np.ndarray:
    """Henderson-Hasselbalch logistic function.

    Args:
        pH: Array of pH values
        pKa: pKa value
        s: Direction (+1=falling, -1=rising, 0=constant)
        P: Amplitude (max population)
    """
    if s == 0:
        return P * np.ones_like(pH)
    return P / (1.0 + 10.0**(s * (pH - pKa)))


def three_state_hh(pH: np.ndarray, pKa_pos: float, pKa_neg: float) -> np.ndarray:
    """Three-state HH for histidine-like residues (UPOS + NONE + UNEG).

    Returns the population of the neutral (NONE) state as a function of pH.
    This creates a bell-shaped curve peaked between pKa_pos and pKa_neg.
    """
    return 1.0 / (1.0 + 10**(pKa_neg - pH) + 10**(pH - pKa_pos))


def two_state_basic_hh(pH: np.ndarray, pKa: float, n: float = 1.0) -> np.ndarray:
    """Two-state HH for basic residues (UPOS + NONE, e.g., Lysine).

    Population of NONE (neutral) state rises with pH.
    """
    return 1.0 / (1.0 + 10**(n * (pKa - pH)))


def two_state_acidic_hh(pH: np.ndarray, pKa: float, n: float = 1.0) -> np.ndarray:
    """Two-state HH for acidic residues (UNEG + NONE, e.g., Aspartate).

    Population of NONE (neutral) state falls with pH.
    """
    return 1.0 / (1.0 + 10**(n * (pH - pKa)))


def compute_block_weights(
    microstates: list[tuple[str, MicrostateType, float]],
) -> dict[str, tuple[float, int]]:
    """Compute Boltzmann weights and curve directions for microstates.

    Args:
        microstates: List of (name, type, pKa) tuples

    Returns:
        Dict mapping name to (weight, sign) where sign is +1/-1/0
    """
    has_none = any(t == "NONE" for _, t, _ in microstates)
    has_upos = any(t == "UPOS" for _, t, _ in microstates)
    has_uneg = any(t == "UNEG" for _, t, _ in microstates)

    weights = {}
    for name, mtype, pKa in microstates:
        if mtype == "NONE":
            w_raw = 1.0
            if has_upos and has_uneg:
                sign = 0  # Bell curve
            elif has_upos:
                sign = -1  # Falls with pH
            else:
                sign = +1  # Rises with pH
        elif mtype == "UPOS":
            w_raw = 10**(-pKa)
            sign = +1  # Always falls with pH
        else:  # UNEG
            w_raw = 10**(-pKa)
            sign = -1  # Always rises with pH
        weights[name] = (w_raw, sign)

    # Normalize weights
    total = sum(w for w, _ in weights.values())
    if total > 0:
        return {k: (w/total, s) for k, (w, s) in weights.items()}
    return weights


def fit_hh_curve(
    pH_values: np.ndarray,
    populations: np.ndarray,
    fit_type: str = "auto",
    initial_pKa: float = 7.0,
) -> HHFitResult:
    """Fit Henderson-Hasselbalch curve to population data.

    Handles 6 cases:
    - three_state: UPOS + UNEG + NONE (e.g., Histidine) - bell curve
    - basic: UPOS + NONE (e.g., Lysine) - deprotonation with rising pH
    - acidic: UNEG + NONE (e.g., Asp/Glu) - deprotonation with rising pH
    - upos_only: Only UPOS state - constant protonated fraction
    - uneg_only: Only UNEG state - constant deprotonated fraction
    - auto: Auto-detect from population trend

    Args:
        pH_values: Array of pH values
        populations: Array of population fractions (0-1)
        fit_type: Fitting case or "auto" to detect
        initial_pKa: Initial guess for pKa

    Returns:
        HHFitResult with fitted parameters
    """
    if len(pH_values) < 3:
        return HHFitResult(pKa_eff=initial_pKa, fit_type="insufficient_data")

    # Sort by pH for consistent analysis
    sort_idx = np.argsort(pH_values)
    pH_sorted = pH_values[sort_idx]
    pop_sorted = populations[sort_idx]

    # Handle constant population cases (UPOS-only or UNEG-only)
    pop_range = np.max(pop_sorted) - np.min(pop_sorted)
    if fit_type == "upos_only" or (fit_type == "auto" and pop_range < 0.1 and np.mean(pop_sorted) > 0.5):
        # Constant high population - likely UPOS-only (fully protonated)
        return HHFitResult(
            pKa_eff=np.mean(pH_sorted),  # pKa undefined, report mean pH
            fit_type="upos_only",
            r_squared=1.0 if pop_range < 0.05 else 0.9,  # High R² for constant
        )

    if fit_type == "uneg_only" or (fit_type == "auto" and pop_range < 0.1 and np.mean(pop_sorted) < 0.5):
        # Constant low population - likely UNEG-only (fully deprotonated)
        return HHFitResult(
            pKa_eff=np.mean(pH_sorted),
            fit_type="uneg_only",
            r_squared=1.0 if pop_range < 0.05 else 0.9,
        )

    # Auto-detect fit type from population trend
    if fit_type == "auto":
        trend = pop_sorted[-1] - pop_sorted[0]
        if trend > 0.2:
            fit_type = "basic"  # Rising with pH → UPOS (deprotonated) grows
        elif trend < -0.2:
            fit_type = "acidic"  # Falling with pH → UNEG (protonated) shrinks
        else:
            # Check for bell shape
            mid_idx = len(pop_sorted) // 2
            mid_pop = pop_sorted[mid_idx]
            edge_pop = max(pop_sorted[0], pop_sorted[-1])
            if mid_pop > edge_pop + 0.1:
                fit_type = "three_state"
            else:
                fit_type = "acidic"  # Default for weak trend

    try:
        if fit_type == "three_state":
            # Fit bell-shaped curve with two pKa values
            def fit_func(pH, pKa_pos, pKa_neg):
                return three_state_hh(pH, pKa_pos, pKa_neg)

            popt, _ = curve_fit(
                fit_func,
                pH_sorted,
                pop_sorted,
                p0=[initial_pKa + 1, initial_pKa - 1],
                bounds=([0, 0], [14, 14]),
                maxfev=5000,
            )
            pKa_eff = (popt[0] + popt[1]) / 2
            y_pred = fit_func(pH_sorted, *popt)
            ss_res = np.sum((pop_sorted - y_pred)**2)
            ss_tot = np.sum((pop_sorted - np.mean(pop_sorted))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return HHFitResult(
                pKa_eff=pKa_eff,
                pKa_pos=popt[0],
                pKa_neg=popt[1],
                fit_type="three_state",
                r_squared=r_squared,
            )

        elif fit_type == "basic":
            def fit_func(pH, pKa, n):
                return two_state_basic_hh(pH, pKa, n)

            popt, _ = curve_fit(
                fit_func,
                pH_sorted,
                pop_sorted,
                p0=[initial_pKa, 1.0],
                bounds=([0, 0.1], [14, 5]),
                maxfev=5000,
            )
            y_pred = fit_func(pH_sorted, *popt)
            ss_res = np.sum((pop_sorted - y_pred)**2)
            ss_tot = np.sum((pop_sorted - np.mean(pop_sorted))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return HHFitResult(
                pKa_eff=popt[0],
                hill_coeff=popt[1],
                fit_type="basic",
                r_squared=r_squared,
            )

        else:  # acidic
            def fit_func(pH, pKa, n):
                return two_state_acidic_hh(pH, pKa, n)

            popt, _ = curve_fit(
                fit_func,
                pH_sorted,
                pop_sorted,
                p0=[initial_pKa, 1.0],
                bounds=([0, 0.1], [14, 5]),
                maxfev=5000,
            )
            y_pred = fit_func(pH_sorted, *popt)
            ss_res = np.sum((pop_sorted - y_pred)**2)
            ss_tot = np.sum((pop_sorted - np.mean(pop_sorted))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return HHFitResult(
                pKa_eff=popt[0],
                hill_coeff=popt[1],
                fit_type="acidic",
                r_squared=r_squared,
            )

    except (RuntimeError, ValueError) as e:
        logger.warning("HH fitting failed: %s", e)
        return HHFitResult(pKa_eff=initial_pKa, fit_type="fit_failed")


def plot_hh_curves(
    results: dict[str, HHFitResult],
    pH_range: tuple[float, float],
    populations_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    run_idx: int,
) -> None:
    """Generate publication-quality HH curve plots.

    Args:
        results: Dict mapping residue name to HHFitResult
        pH_range: (min_pH, max_pH) for plotting
        populations_data: Dict mapping residue name to (pH_values, populations)
        output_path: Directory for output files
        run_idx: Current run number for labeling
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping HH plots")
        return

    from cphmd.analysis.plot_style import apply_pub_style, clean_axes, savefig

    apply_pub_style()
    output_path.mkdir(parents=True, exist_ok=True)

    # Create pH array for fitted curves
    pH_fit = np.linspace(pH_range[0], pH_range[1], 100)

    for resname, result in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot raw data if available
        if resname in populations_data:
            pH_vals, pops = populations_data[resname]
            ax.scatter(pH_vals, pops, c='blue', s=50, alpha=0.7,
                      label='Data', zorder=3)

        # Plot fitted curve
        if result.fit_type == "three_state":
            y_fit = three_state_hh(pH_fit, result.pKa_pos, result.pKa_neg)
            label = f"HH fit: pKa+ = {result.pKa_pos:.2f}, pKa- = {result.pKa_neg:.2f}"
        elif result.fit_type == "basic":
            y_fit = two_state_basic_hh(pH_fit, result.pKa_eff, result.hill_coeff)
            label = f"HH fit: pKa = {result.pKa_eff:.2f}, n = {result.hill_coeff:.2f}"
        elif result.fit_type == "acidic":
            y_fit = two_state_acidic_hh(pH_fit, result.pKa_eff, result.hill_coeff)
            label = f"HH fit: pKa = {result.pKa_eff:.2f}, n = {result.hill_coeff:.2f}"
        else:
            continue  # Skip invalid fits

        ax.plot(pH_fit, y_fit, 'r-', linewidth=2, label=label)

        ax.set_xlabel('pH')
        ax.set_ylabel('Population')
        ax.set_title(f'{resname} \u2014 Run {run_idx}', fontweight='bold')
        ax.set_xlim(pH_range)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        clean_axes(ax)

        ax.annotate(
            f"R\u00b2 = {result.r_squared:.3f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            verticalalignment='top',
        )

        savefig(fig, output_path / f"combined_{resname}_run{run_idx}.png")

    print(f"HH plots saved to {output_path}")


def compute_theoretical_populations(
    pH_grid: np.ndarray,
    microstates: list[tuple[str, MicrostateType, float | None]],
) -> dict[str, np.ndarray]:
    """Compute theoretical HH populations using Boltzmann partition function.

    In CpHMD, the reference state (NONE) is the original residue topology:
    - For acids (ASP, GLU): NONE = deprotonated (COO-)
    - For bases (HIS, LYS): NONE = protonated (HSP, LYS-NH3+)

    Patches create alternative protonation states:
    - UPOS (fewer atoms, H removed): deprotonated form, weight = 10^(pH - pKa)
      The micro-pKa is for the reaction: original -> UPOS + H+
    - UNEG (more atoms, H added): protonated form, weight = 10^(pKa - pH)
      The micro-pKa is for the reaction: original + H+ -> UNEG

    Multiple substates of the same type compete via their micro-pKa values:
    - HIS: UPOS HSD (pKa=6.6) vs HSE (pKa=7.0) -> HSD:HSE ~ 70:30
    - ASP: UNEG ASH1 (pKa=3.67) vs ASH2 (pKa=3.67) -> 50:50

    Args:
        pH_grid: Array of pH values to compute populations for
        microstates: List of (name, type, pKa) tuples from TAG field

    Returns:
        Dict mapping microstate name to population array (same length as pH_grid)
    """
    # Collect pKa values per type
    pKa_upos = [pKa for _, t, pKa in microstates if t == "UPOS" and pKa is not None]
    pKa_uneg = [pKa for _, t, pKa in microstates if t == "UNEG" and pKa is not None]
    n_none = sum(1 for _, t, _ in microstates if t == "NONE")

    populations = {}

    for name, mtype, pKa in microstates:
        pop_array = np.zeros_like(pH_grid, dtype=float)

        for i, pH in enumerate(pH_grid):
            # Partition function: Z = Σ weights
            Z = 0.0

            # NONE: reference state(s), weight = 1.0 each
            Z += n_none * 1.0

            # UPOS: deprotonated forms (H removed), dominant at HIGH pH
            for pk in pKa_upos:
                Z += 10 ** (pH - pk)

            # UNEG: protonated forms (H added), dominant at LOW pH
            for pk in pKa_uneg:
                Z += 10 ** (pk - pH)

            # This state's Boltzmann weight
            if mtype == "NONE":
                w = 1.0
            elif mtype == "UPOS" and pKa is not None:
                w = 10 ** (pH - pKa)
            elif mtype == "UNEG" and pKa is not None:
                w = 10 ** (pKa - pH)
            else:
                w = 0.0

            pop_array[i] = w / Z if Z > 0 else 0.0

        populations[name] = pop_array

    return populations


def _fit_substate_curve(
    ax,
    pH_values: np.ndarray,
    populations: np.ndarray,
    tag_type: str,
    pH_fit: np.ndarray,
    color,
) -> float | None:
    """Fit a sigmoid to one substate's simulation data and plot it.

    Uses HH logistic: pop = U / (1 + 10^(s*(pKa - pH)))
    where s = +1 for rising (UPOS/basic), -1 for falling (UNEG/acidic).
    Lower asymptote is fixed at 0 (physical requirement: every substate
    population must vanish at one pH extreme).

    Returns:
        Fitted pKa value, or None if fit failed or data is flat.
    """
    from scipy.optimize import curve_fit

    pop_range = np.max(populations) - np.min(populations)
    if pop_range < 0.05:
        return None  # Nearly flat — no meaningful curve to fit

    # Determine sigmoid direction from tag type or data trend
    if tag_type == "UNEG":
        s = -1  # Falls with pH (protonated state loses population)
    elif tag_type == "UPOS":
        s = +1  # Rises with pH (deprotonated state gains population)
    else:
        # NONE: detect from trend
        trend = populations[np.argmax(pH_values)] - populations[np.argmin(pH_values)]
        s = -1 if trend < 0 else +1

    def sigmoid(pH, pKa, U):
        return U / (1.0 + 10.0 ** (s * (pKa - pH)))

    try:
        p0 = [np.mean(pH_values), np.max(populations)]
        bounds = ([0, 0.01], [14, 1.1])
        popt, _ = curve_fit(sigmoid, pH_values, populations, p0=p0, bounds=bounds, maxfev=5000)
        ax.plot(pH_fit, sigmoid(pH_fit, *popt), color=color, linewidth=1.5, alpha=0.8)
        return float(popt[0])  # fitted pKa
    except (RuntimeError, ValueError):
        return None


def plot_site_substates(
    site_result: SiteHHResult,
    pH_range: tuple[float, float],
    output_path: Path,
    run_idx: int,
) -> None:
    """Generate plot showing all substates for a single site.

    Creates one plot per SEGID+RESID showing all substates with different
    colors/markers. Includes both simulation data and theoretical curves.

    Args:
        site_result: SiteHHResult with substate populations
        pH_range: (min_pH, max_pH) for plotting
        output_path: Directory for output files
        run_idx: Current run number for labeling
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping substate plots")
        return

    from cphmd.analysis.plot_style import apply_pub_style, clean_axes, savefig

    apply_pub_style()
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Match population_convergence.py colors (Set1: red, blue, green, purple, orange, ...)
    colors = plt.cm.Set1.colors
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*"]

    pH_fit = np.linspace(pH_range[0], pH_range[1], 200)

    # Build microstates list from substates for theoretical curves on dense grid
    microstates_for_theo: list[tuple[str, MicrostateType, float | None]] = []
    for sub in site_result.substates:
        mtype: MicrostateType = "NONE"
        if sub.tag_type == "UPOS":
            mtype = "UPOS"
        elif sub.tag_type == "UNEG":
            mtype = "UNEG"
        microstates_for_theo.append((sub.select_name, mtype, sub.tag_pKa))
    theo_dense = compute_theoretical_populations(pH_fit, microstates_for_theo)

    for idx, substate in enumerate(site_result.substates):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Theoretical curve (dashed)
        if substate.select_name in theo_dense:
            ax.plot(
                pH_fit,
                theo_dense[substate.select_name],
                color=color,
                linestyle='--',
                linewidth=1.5,
                alpha=0.5,
            )

        # Build label with pKa annotations
        theo_pKa = substate.tag_pKa
        fitted_pKa = None

        # Simulation data (scatter points only)
        if len(substate.populations) > 0:
            # Fit sigmoid through this substate's simulation data (solid line)
            if len(substate.populations) >= 3:
                fitted_pKa = _fit_substate_curve(
                    ax, substate.pH_values, substate.populations,
                    substate.tag_type, pH_fit, color,
                )

            # Build legend label: "State 1 (theo=6.60, fit=6.82)" or just "State 1"
            label = f"State {idx + 1}"
            pKa_parts = []
            if theo_pKa is not None:
                pKa_parts.append(f"theo={theo_pKa:.2f}")
            if fitted_pKa is not None:
                pKa_parts.append(f"fit={fitted_pKa:.2f}")
            if pKa_parts:
                label += f" ({', '.join(pKa_parts)})"

            ax.scatter(
                substate.pH_values,
                substate.populations,
                color=color,
                marker=marker,
                s=25,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.9,
                label=label,
                zorder=3,
            )

    site_label = f"{site_result.segid}:{site_result.resname}{site_result.resid}"
    ax.set_xlabel('pH')
    ax.set_ylabel('Population')
    ax.set_title(f'{site_label} \u2014 Run {run_idx}', fontweight='bold')
    ax.set_xlim(pH_range)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', ncol=1, fontsize=8)
    clean_axes(ax)

    savefig(fig, output_path / f"site{site_result.site_id}_run{run_idx}.png")


def write_hh_csv(
    site_results: dict[str, SiteHHResult],
    output_path: Path,
    run_idx: int,
) -> Path:
    """Write HH analysis data to CSV file.

    Format: pH, site_id, state_id, population, fitted_population, tag_type

    Args:
        site_results: Dict mapping site IDs to SiteHHResult
        output_path: Directory for output file
        run_idx: Current run number

    Returns:
        Path to the written CSV file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"data_run{run_idx}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pH', 'site_id', 'segid', 'resid', 'resname',
            'state_id', 'tag_type', 'tag_pKa',
            'population', 'theoretical_population'
        ])

        for site_id, result in site_results.items():
            for substate in result.substates:
                for i, pH in enumerate(substate.pH_values):
                    pop = substate.populations[i] if i < len(substate.populations) else 0.0
                    theo = substate.theoretical[i] if i < len(substate.theoretical) else 0.0
                    writer.writerow([
                        f"{pH:.3f}",
                        site_id,
                        result.segid,
                        result.resid,
                        result.resname,
                        substate.select_name,
                        substate.tag_type,
                        f"{substate.tag_pKa:.2f}" if substate.tag_pKa else "",
                        f"{pop:.6f}",
                        f"{theo:.6f}",
                    ])

    print(f"HH CSV data saved to {csv_path}")
    return csv_path


def _load_run_populations(
    data_dir: Path,
    pH: float,
    delta_pKa: float,
    nreps: int,
    ncentral: int,
    lambda_threshold: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], int] | None:
    """Load one run's lambda data and compute per-state populations.

    Each repeat file within a replica becomes a separate data point at
    that replica's pH, so the scatter plot shows run-to-run spread.

    Returns:
        (valid_pH_array, state_populations, n_states) or None if insufficient data.
        state_populations maps state_idx to array of populations (one per repeat).
    """
    from cphmd.core.phase_switcher import _organize_by_replica
    from cphmd.utils.lambda_io import read_lambda_values

    pH_values = np.array([pH + delta_pKa * (j - ncentral) for j in range(nreps)])

    replica_files = _organize_by_replica(data_dir)
    if replica_files is None:
        return None

    # Build per-repeat data: each repeat file is a separate data point
    all_pH: list[float] = []
    all_repeat_data: list[np.ndarray] = []  # each entry is (frames, n_states)
    n_states = None

    for replica_idx in sorted(replica_files.keys()):
        if replica_idx >= nreps:
            continue
        replica_pH = pH_values[replica_idx]

        for fpath in sorted(replica_files[replica_idx]):
            try:
                data = read_lambda_values(fpath)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.size == 0:
                    continue
                if n_states is None:
                    n_states = data.shape[1]
                all_pH.append(replica_pH)
                all_repeat_data.append(data)
            except Exception:
                continue

    if len(all_pH) < 3 or n_states is None:
        return None

    valid_pH_array = np.array(all_pH)

    state_populations: dict[int, np.ndarray] = {}
    for state_idx in range(n_states):
        pops = []
        for data in all_repeat_data:
            mask = data[:, state_idx] > lambda_threshold
            pops.append(mask.mean())
        state_populations[state_idx] = np.array(pops)

    return valid_pH_array, state_populations, n_states


def generate_hh_analysis(
    run_idx: int,
    data_dir: Path,
    patch_info: "pd.DataFrame",
    pH: float,
    delta_pKa: float,
    nreps: int,
    output_dir: Path,
    ncentral: int | None = None,
    lambda_threshold: float = 0.985,
) -> dict[str, SiteHHResult]:
    """Main entry point for HH curve analysis with multi-replica pH.

    This function implements the full multi-replica titration curve analysis:
    1. Computes per-replica pH values from the replica exchange scheme
    2. Loads lambda data per replica for the current run
    3. Calculates populations at each replica's pH
    4. Fits multi-point HH curves for each titratable site
    5. Generates substate plots and CSV data export

    Each run is analyzed independently since biases differ between runs.

    Args:
        run_idx: Current run number
        data_dir: Path to data/ directory with Lambda files
        patch_info: DataFrame with patch/residue information
        pH: Effective pH for central replica
        delta_pKa: pH increment between replicas
        nreps: Number of replicas
        output_dir: Directory for output plots
        ncentral: Central replica index (defaults to nreps // 2)
        lambda_threshold: Threshold for state occupancy (default 0.985)

    Returns:
        Dict mapping site IDs to SiteHHResult with full analysis
    """
    if ncentral is None:
        ncentral = nreps // 2

    results: dict[str, SiteHHResult] = {}
    legacy_results: dict[str, HHFitResult] = {}
    populations_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Load current run's data
    current_data = _load_run_populations(
        data_dir, pH, delta_pKa, nreps, ncentral, lambda_threshold
    )

    if current_data is None:
        print("Insufficient replicas for HH analysis in current run")
        # Fall back to combined analysis
        from cphmd.core.phase_switcher import calculate_populations, load_lambda_data
        lambda_data, _ = load_lambda_data(data_dir)
        if lambda_data is None:
            return {}
        pop_data = calculate_populations(lambda_data, thresholds=(lambda_threshold, 0.985))
        if not pop_data:
            return {}

        # Create minimal results with single-point estimate
        if patch_info is not None and "site" in patch_info.columns:
            for site_id in patch_info["site"].unique():
                site_patches = patch_info[patch_info["site"] == site_id]
                if len(site_patches) == 0:
                    continue
                first_row = site_patches.iloc[0]
                segid = str(first_row.get("SEGID", ""))
                resid = str(first_row.get("RESID", ""))
                resname = str(first_row.get("RESNAME", first_row.get("resname", "")))

                fit_result = HHFitResult(pKa_eff=pH, fit_type="single_point", r_squared=0.0)
                results[str(site_id)] = SiteHHResult(
                    site_id=str(site_id),
                    segid=segid,
                    resid=resid,
                    resname=resname,
                    fit_result=fit_result,
                    pH_values=np.array([pH]),
                )
        return results

    valid_pH_array, state_populations, n_states = current_data

    # Process each titratable site
    if patch_info is None or "site" not in patch_info.columns:
        return {}

    for site_id in patch_info["site"].unique():
        site_patches = patch_info[patch_info["site"] == site_id]
        if len(site_patches) == 0:
            continue

        # Get site metadata
        first_row = site_patches.iloc[0]
        segid = str(first_row.get("SEGID", ""))
        resid = str(first_row.get("RESID", ""))
        resname = str(first_row.get("RESNAME", first_row.get("resname", "")))

        # Build substates list and gather microstates for theoretical model
        substates: list[SubstatePopulation] = []
        microstates: list[tuple[str, MicrostateType, float | None]] = []
        charged_state_indices: list[int] = []

        for idx, (_, row) in enumerate(site_patches.iterrows()):
            select_name = str(row.get("SELECT", f"s{site_id}s{idx}"))
            tag = str(row.get("TAG", "NONE")).strip().upper()
            parts = tag.split()

            tag_type = parts[0] if parts else "NONE"
            tag_pKa = None
            if len(parts) >= 2:
                try:
                    tag_pKa = float(parts[1])
                except ValueError:
                    pass

            # Map tag type to MicrostateType
            mtype: MicrostateType = "NONE"
            if tag_type == "UPOS":
                mtype = "UPOS"
                charged_state_indices.append(idx)
            elif tag_type == "UNEG":
                mtype = "UNEG"
                charged_state_indices.append(idx)

            microstates.append((select_name, mtype, tag_pKa))

            # Get simulation populations for this state
            state_pops = state_populations.get(idx, np.zeros(len(valid_pH_array)))

            substates.append(SubstatePopulation(
                state_idx=idx,
                select_name=select_name,
                tag_type=tag_type,
                tag_pKa=tag_pKa,
                pH_values=valid_pH_array.copy(),
                populations=state_pops,
            ))

        # Compute theoretical populations
        if microstates:
            theo_pops = compute_theoretical_populations(valid_pH_array, microstates)
            for substate in substates:
                if substate.select_name in theo_pops:
                    substate.theoretical = theo_pops[substate.select_name]

        # Compute total charged-state population for fitting
        total_charged_pop = np.zeros(len(valid_pH_array))
        for idx in charged_state_indices:
            total_charged_pop += state_populations.get(idx, np.zeros(len(valid_pH_array)))

        # Determine fit type from site composition
        has_upos = any(m[1] == "UPOS" for m in microstates)
        has_uneg = any(m[1] == "UNEG" for m in microstates)
        has_none = any(m[1] == "NONE" for m in microstates)

        fit_type = "auto"
        if has_upos and has_uneg and has_none:
            fit_type = "three_state"
        elif has_upos and has_none:
            fit_type = "basic"
        elif has_uneg and has_none:
            fit_type = "acidic"
        elif has_upos and not has_uneg and not has_none:
            fit_type = "upos_only"
        elif has_uneg and not has_upos and not has_none:
            fit_type = "uneg_only"

        # Get initial pKa guess from TAG values
        pKa_values = [m[2] for m in microstates if m[2] is not None]
        initial_pKa = np.mean(pKa_values) if pKa_values else pH

        # Fit HH curve
        fit_result = fit_hh_curve(
            valid_pH_array,
            total_charged_pop,
            fit_type=fit_type,
            initial_pKa=initial_pKa,
        )

        # Create site result
        site_result = SiteHHResult(
            site_id=str(site_id),
            segid=segid,
            resid=resid,
            resname=resname,
            fit_result=fit_result,
            substates=substates,
            pH_values=valid_pH_array.copy(),
            total_populations=total_charged_pop,
        )
        results[str(site_id)] = site_result

        # Store for legacy plot interface
        legacy_results[f"{segid}:{resname}{resid}"] = fit_result
        populations_data[f"{segid}:{resname}{resid}"] = (valid_pH_array, total_charged_pop)

    # Generate plots — all HH output goes to hh_plots/ subdirectory
    hh_dir = Path(output_dir) / "hh_plots"
    pH_range = (valid_pH_array.min() - 0.5, valid_pH_array.max() + 0.5)

    # Per-site substate plots
    for site_result in results.values():
        plot_site_substates(site_result, pH_range, hh_dir, run_idx)

    # CSV export
    write_hh_csv(results, hh_dir, run_idx)

    return results
