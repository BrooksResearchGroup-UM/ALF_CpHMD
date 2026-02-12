"""Population convergence analysis over ALF runs.

Reads populations.dat from successive analysis directories and plots how
normalized populations evolve, separately for the relaxed (λ > 0.8) and
strict (λ > 0.985) thresholds.  For multi-site systems, each site gets
its own plot with per-site normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Phase → alpha mapping for visual emphasis:
# Phase 1 (exploration) is faint, Phase 3 (production) is fully opaque
_PHASE_ALPHA = {1: 0.5, 2: 0.8, 3: 1.0}


def _read_phases_from_runs(
    input_folder: Path,
    runs: np.ndarray,
) -> np.ndarray:
    """Read phase.dat from each analysisN/ directory.

    Returns:
        1-D int array of phase values (1, 2, or 3) per run.
        Defaults to 1 if phase.dat is missing.
    """
    phases = np.ones(len(runs), dtype=int)
    for i, run_idx in enumerate(runs):
        phase_file = input_folder / f"analysis{run_idx}" / "phase.dat"
        if phase_file.exists():
            try:
                phases[i] = int(np.loadtxt(phase_file))
            except (OSError, ValueError):
                pass
    return phases


def _configure_plot_style():
    """Apply consistent publication-quality plot styling."""
    import matplotlib.pyplot as plt

    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4


def _plot_by_phase(
    ax,
    runs: np.ndarray,
    y: np.ndarray,
    phases: np.ndarray,
    *,
    base_alpha: float = 1.0,
    label: str | None = None,
    **kwargs,
) -> None:
    """Plot a line in contiguous phase segments with per-phase alpha.

    The final alpha is ``base_alpha * phase_alpha``.  Segments overlap by
    1 point at phase boundaries so the line is continuous.
    Only the first segment gets the legend label.
    """
    # Identify contiguous runs of the same phase
    change_points = np.where(np.diff(phases) != 0)[0] + 1
    segments = np.split(np.arange(len(runs)), change_points)

    for seg_idx, seg in enumerate(segments):
        if len(seg) == 0:
            continue
        # Extend segment by 1 into next phase for continuity
        end = min(seg[-1] + 2, len(runs))
        idx = np.arange(seg[0], end)
        phase = phases[seg[0]]
        alpha = base_alpha * _PHASE_ALPHA.get(phase, 1.0)
        ax.plot(
            runs[idx], y[idx],
            alpha=alpha,
            label=label if seg_idx == 0 else None,
            **kwargs,
        )


def read_populations_from_runs(
    input_folder: Path,
    max_run: int,
) -> dict[str, np.ndarray]:
    """Read hit counts from analysis1/ … analysisN/ directories.

    Args:
        input_folder: Parent directory containing analysisN/ subdirectories.
        max_run: Highest run index to consider.

    Returns:
        Dictionary with keys:
          - ``runs``: 1-D int array of run indices that had a valid file
          - ``hits_relaxed``: (n_runs, n_states) hit counts at λ > 0.8
          - ``hits_strict``:  (n_runs, n_states) hit counts at λ > 0.985
    """
    runs: list[int] = []
    hits_relaxed_list: list[np.ndarray] = []
    hits_strict_list: list[np.ndarray] = []

    for run_idx in range(0, max_run + 1):
        pop_file = input_folder / f"analysis{run_idx}" / "populations.dat"
        if not pop_file.exists():
            continue

        try:
            data = np.loadtxt(pop_file, comments="#")
        except Exception:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Column layout (0-indexed):
        #   0: State  1: Raw(>0.8)  2: Norm(>0.8)  3: Hits(>0.8)
        #   4: Raw(>0.985)  5: Norm(>0.985)  6: Hits(>0.985)
        hits_relaxed_list.append(data[:, 3])
        hits_strict_list.append(data[:, 6])
        runs.append(run_idx)

    if not runs:
        return {}

    return {
        "runs": np.array(runs),
        "hits_relaxed": np.array(hits_relaxed_list),  # (n_runs, n_states)
        "hits_strict": np.array(hits_strict_list),
    }


def _normalize_per_site(
    hits: np.ndarray, nsubs: list[int]
) -> list[np.ndarray]:
    """Normalize hit counts per site.

    Args:
        hits: (n_runs, n_states) hit count array.
        nsubs: Number of substates per site, e.g. [2, 2].

    Returns:
        List of (n_runs, nsubs[i]) normalized population arrays, one per site.
    """
    site_pops = []
    offset = 0
    for ns in nsubs:
        site_hits = hits[:, offset : offset + ns]
        totals = site_hits.sum(axis=1, keepdims=True)
        # Avoid division by zero
        totals = np.where(totals > 0, totals, 1.0)
        site_pops.append(site_hits / totals)
        offset += ns
    return site_pops


def plot_population_convergence(
    runs: np.ndarray,
    site_pops: np.ndarray,
    threshold_label: str,
    site_label: str,
    output_path: Path,
    phases: np.ndarray | None = None,
) -> None:
    """Create a single convergence plot for one site and threshold.

    Args:
        runs: 1-D array of run indices.
        site_pops: (n_runs, n_substates) normalized populations for this site.
        threshold_label: e.g. "λ > 0.8".
        site_label: e.g. "Site 0" or "" for single-site systems.
        output_path: Full path for the output PNG file.
        phases: Optional 1-D array of phase values (1/2/3) per run.
            Controls line transparency — phase 1 is faint, phase 3 is opaque.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping population convergence plots")
        return

    _configure_plot_style()

    n_substates = site_pops.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Set1.colors
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*"]

    for s in range(n_substates):
        color = colors[s % len(colors)]
        marker = markers[s % len(markers)]

        if phases is not None:
            # Plot each contiguous phase segment separately for different alpha
            _plot_by_phase(
                ax, runs, site_pops[:, s], phases,
                color=color, marker=marker, linewidth=2, markersize=6,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"State {s}",
            )
        else:
            ax.plot(
                runs,
                site_pops[:, s],
                marker=marker,
                color=color,
                linewidth=2,
                markersize=6,
                markeredgecolor='black',
                markeredgewidth=0.5,
                label=f"State {s}",
            )

    # Ideal equal-population line
    ideal = 1.0 / n_substates
    ax.axhline(ideal, color="black", linestyle="--", linewidth=1.5, alpha=0.6,
               label=f"Ideal (1/{n_substates})")

    # Final frac_diff annotation
    final_pops = site_pops[-1]
    frac_diff = (final_pops.max() - final_pops.min()) * 100
    ax.annotate(
        f"Final diff = {frac_diff:.1f}%",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        fontsize=11,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.9),
    )

    title = f"Population Convergence ({threshold_label})"
    if site_label:
        title += f" — {site_label}"

    from matplotlib.ticker import MaxNLocator

    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Normalized Population", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=9, ncol=max(1, n_substates // 5))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(runs[0] - 0.5, runs[-1] + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_population_plots(
    input_folder: Path,
    max_run: int,
    output_dir: Path,
    nsubs: list[int] | np.ndarray | None = None,
) -> None:
    """Generate population convergence plots, one per site per threshold.

    Args:
        input_folder: Parent directory containing analysisN/ subdirectories.
        max_run: Highest run index to consider.
        output_dir: Directory for output PNG files (created if needed).
        nsubs: Number of substates per site, e.g. [2, 2].
            If None, treats all states as a single site.
    """
    run_data = read_populations_from_runs(input_folder, max_run)
    if not run_data:
        return

    if len(run_data["runs"]) < 1:
        return

    runs = run_data["runs"]
    n_states = run_data["hits_relaxed"].shape[1]

    # Default: single site with all states
    if nsubs is None:
        nsubs = [n_states]
    nsubs = list(nsubs)

    relaxed_per_site = _normalize_per_site(run_data["hits_relaxed"], nsubs)
    strict_per_site = _normalize_per_site(run_data["hits_strict"], nsubs)

    # Read per-run phase info for transparency
    phases = _read_phases_from_runs(input_folder, runs)

    n_sites = len(nsubs)
    multi_site = n_sites > 1

    for thresh_key, thresh_label, site_pops_list in [
        ("relaxed", "λ > 0.8", relaxed_per_site),
        ("strict", "λ > 0.985", strict_per_site),
    ]:
        for site_idx, site_pops in enumerate(site_pops_list):
            if multi_site:
                site_label = f"Site {site_idx}"
                fname = f"populations_{thresh_key}_site{site_idx}.png"
            else:
                site_label = ""
                fname = f"populations_{thresh_key}.png"

            plot_population_convergence(
                runs=runs,
                site_pops=site_pops,
                threshold_label=thresh_label,
                site_label=site_label,
                output_path=output_dir / fname,
                phases=phases,
            )

    print(f"Population convergence plots saved to {output_dir}")
