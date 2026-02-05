"""RMSD convergence plots for ALF bias landscape monitoring.

Plots per-site G-file RMSD over ALF iterations, showing how the bias
landscape stabilizes.  Works with both convergence modes:
- ``rmsd``:  plots from precomputed RMSDState history
- ``population``:  computes RMSD on the fly from analysis directories

Two plot types:
1. Per-site summary: one RMSD line per site (aggregated)
2. Per-pair detail: one line per lambda pair, colored by pair index
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _configure_plot_style():
    """Apply consistent publication-quality plot styling."""
    import matplotlib.pyplot as plt

    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4


def _collect_rmsd_from_dirs(
    input_folder: Path,
    max_run: int,
    nsubs: list[int],
    lag: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-site RMSD across analysis directories.

    Returns:
        (runs, rmsds, coverages) where rmsds/coverages are (n_runs, n_sites).
    """
    from cphmd.core.rmsd_convergence import compute_site_rmsd

    runs: list[int] = []
    rmsds: list[list[float]] = []
    coverages: list[list[float]] = []

    for run_idx in range(1, max_run + 1):
        analysis_dir = input_folder / f"analysis{run_idx}"
        ref_dir = input_folder / f"analysis{run_idx - lag}"
        if not (analysis_dir / "multisite").is_dir():
            continue
        if not (ref_dir / "multisite").is_dir():
            continue

        site_rmsds, site_covs = compute_site_rmsd(analysis_dir, ref_dir, nsubs)
        runs.append(run_idx)
        rmsds.append(site_rmsds)
        coverages.append(site_covs)

    if not runs:
        return np.array([]), np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)

    return np.array(runs), np.array(rmsds), np.array(coverages)


def _collect_pairwise_rmsd_from_dirs(
    input_folder: Path,
    max_run: int,
    nsubs: list[int],
    lag: int = 5,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Compute per-pair RMSD across analysis directories.

    Returns:
        (runs, pair_data_per_run) where pair_data_per_run[run_idx] is a list
        of PairwiseRMSD objects (one per site).
    """
    from cphmd.core.rmsd_convergence import compute_pairwise_rmsd

    runs: list[int] = []
    pair_data: list[list] = []  # list of PairwiseRMSD lists

    for run_idx in range(1, max_run + 1):
        analysis_dir = input_folder / f"analysis{run_idx}"
        ref_dir = input_folder / f"analysis{run_idx - lag}"
        if not (analysis_dir / "multisite").is_dir():
            continue
        if not (ref_dir / "multisite").is_dir():
            continue

        site_pairs = compute_pairwise_rmsd(analysis_dir, ref_dir, nsubs)
        runs.append(run_idx)
        pair_data.append(site_pairs)

    return np.array(runs), pair_data


def plot_rmsd_convergence(
    runs: np.ndarray,
    rmsds: np.ndarray,
    coverages: np.ndarray,
    nsubs: list[int],
    lag: int,
    output_path: Path,
) -> None:
    """Create a per-site RMSD convergence plot.

    Shows RMSD between consecutive runs (previous vs current comparison).

    Args:
        runs: 1-D array of run indices.
        rmsds: (n_runs, n_sites) per-site RMSD values.
        coverages: (n_runs, n_sites) per-site coverage fractions.
        nsubs: Number of substates per site.
        lag: Lag used for RMSD computation (shown in label).
        output_path: Full path for the output PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping RMSD convergence plots")
        return

    _configure_plot_style()

    n_sites = rmsds.shape[1]
    fig, axes = plt.subplots(
        n_sites, 1,
        figsize=(8, 3 * n_sites + 1),
        sharex=True,
        squeeze=False,
    )

    colors = plt.cm.Set1.colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']

    for s in range(n_sites):
        ax = axes[s, 0]
        color = colors[s % len(colors)]
        marker = markers[s % len(markers)]

        # Replace inf with NaN for plotting
        site_rmsd = rmsds[:, s].copy()
        site_rmsd[~np.isfinite(site_rmsd)] = np.nan

        # Main RMSD line with markers (semilog to resolve small values)
        ax.semilogy(
            runs, site_rmsd,
            marker=marker, color=color, linewidth=2, markersize=6,
            markeredgecolor='black', markeredgewidth=0.5,
            label=f"RMSD vs run-{lag}",
        )

        # Coverage on secondary axis (subtle fill)
        ax2 = ax.twinx()
        ax2.fill_between(
            runs, coverages[:, s],
            alpha=0.1, color=color,
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Coverage", fontsize=10, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=9)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color('grey')
        ax2.spines['right'].set_linewidth(0.5)

        # Annotate final RMSD
        last_finite = site_rmsd[np.isfinite(site_rmsd)]
        if len(last_finite) > 0:
            ax.annotate(
                f"Final RMSD = {last_finite[-1]:.2f}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                fontsize=10, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9),
            )

        site_label = f"Site {s+1} ({nsubs[s]} substates)"
        ax.set_ylabel("RMSD (kcal/mol)", fontsize=11)
        ax.set_title(site_label, fontsize=12, fontweight='bold', loc="left")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

        # Clean up spines
        ax.spines['top'].set_visible(False)

    axes[-1, 0].set_xlabel("Run", fontsize=12)
    title = f"G-file RMSD Convergence (vs previous run)"
    if lag != 1:
        title = f"G-file RMSD Convergence (lag={lag})"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_rmsd_convergence(
    runs: np.ndarray,
    pair_data: list[list],
    nsubs: list[int],
    lag: int,
    output_path: Path,
    top_n: int = 5,
) -> None:
    """Create per-pair RMSD convergence plot with individual lines per lambda pair.

    Each site gets one subplot. Pairwise profiles are shown as colored lines,
    individual substate profiles as grey dashed background. The top N worst
    pairs at the final iteration are highlighted with thicker lines and labels.

    Args:
        runs: 1-D array of run indices.
        pair_data: List (per run) of lists (per site) of PairwiseRMSD objects.
        nsubs: Number of substates per site.
        lag: Lag used for RMSD computation.
        output_path: Full path for the output PNG file.
        top_n: Number of worst pairs to highlight.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except ImportError:
        print("matplotlib not available, skipping pairwise RMSD plots")
        return

    if len(runs) < 2 or not pair_data:
        return

    _configure_plot_style()

    n_sites = len(nsubs)
    fig, axes = plt.subplots(
        n_sites, 1,
        figsize=(10, 4 * n_sites + 1),
        sharex=True,
        squeeze=False,
    )

    for s in range(n_sites):
        ax = axes[s, 0]
        ns = nsubs[s]
        n_pairs = ns * (ns - 1) // 2

        # Build time series: (n_runs, n_pairs) for pairwise, (n_runs, ns) for individual
        pair_rmsd = np.full((len(runs), n_pairs), np.nan)
        indiv_rmsd = np.full((len(runs), ns), np.nan)
        pair_labels = []

        for r_idx, site_list in enumerate(pair_data):
            if s >= len(site_list):
                continue
            site = site_list[s]

            for k, (sub_i, rmsd, _cov) in enumerate(site.individual):
                if np.isfinite(rmsd):
                    indiv_rmsd[r_idx, k] = rmsd

            for k, (i, j, rmsd, _cov) in enumerate(site.pairwise):
                if np.isfinite(rmsd):
                    pair_rmsd[r_idx, k] = rmsd
                if r_idx == 0:
                    pair_labels.append(f"{i}-{j}")

        # Plot individual substate profiles as grey dashed background
        for k in range(ns):
            ax.semilogy(
                runs, indiv_rmsd[:, k],
                color="grey", linewidth=0.5, alpha=0.4, linestyle="--",
            )

        # Determine top N worst pairs at the last run
        last_vals = pair_rmsd[-1, :]
        finite_mask = np.isfinite(last_vals)
        if finite_mask.any():
            worst_indices = np.argsort(last_vals[finite_mask])[::-1][:top_n]
            # Map back to original indices
            finite_indices = np.where(finite_mask)[0]
            highlight_set = set(finite_indices[worst_indices])
        else:
            highlight_set = set()

        # Color + linestyle cycling for many pairs
        if n_pairs <= 20:
            tab20 = plt.get_cmap("tab20")
            colors_20 = [tab20(i) for i in range(20)]
        else:
            hsv = plt.get_cmap("hsv")
            n_colors = 20
            colors_20 = [hsv(i / n_colors) for i in range(n_colors)]
        linestyles = ["-", "--", "-.", ":"]

        # Plot all pair lines (i < j only — upper triangle)
        for k in range(n_pairs):
            is_worst = k in highlight_set
            color = colors_20[k % len(colors_20)]
            ls = linestyles[(k // len(colors_20)) % len(linestyles)]
            ax.semilogy(
                runs, pair_rmsd[:, k],
                color=color,
                linestyle=ls if not is_worst else "-",
                linewidth=2.5 if is_worst else 0.8,
                alpha=1.0 if is_worst else 0.35,
                label=pair_labels[k] if is_worst else None,
            )

        site_label = f"Site {s+1} ({ns} substates, {n_pairs} pairs)"
        ax.set_ylabel("RMSD (kcal/mol)", fontsize=11)
        ax.set_title(site_label, fontsize=12, fontweight='bold', loc="left")
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if highlight_set:
            ax.legend(
                loc="upper right", fontsize=8,
                title=f"Top {min(top_n, len(highlight_set))} worst pairs",
                title_fontsize=9,
            )

    axes[-1, 0].set_xlabel("Run", fontsize=12)
    title = f"Per-pair RMSD Convergence (vs previous run)"
    if lag != 1:
        title = f"Per-pair RMSD Convergence (lag={lag})"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_rmsd_convergence_plots(
    input_folder: Path,
    max_run: int,
    output_dir: Path,
    nsubs: list[int],
    lag: int = 1,
    rmsd_state: object | None = None,
) -> None:
    """Generate RMSD convergence plots from analysis directories or precomputed state.

    Produces two plots:
    1. ``rmsd_convergence.png`` — per-site summary (one line per site)
    2. ``rmsd_pairwise.png`` — per-pair detail (one line per lambda pair)

    By default uses lag=1, showing RMSD between each run and its previous run
    (previous vs current comparison).

    Args:
        input_folder: Parent directory containing analysisN/ subdirectories.
        max_run: Highest run index to consider.
        output_dir: Directory for output PNG files (created if needed).
        nsubs: Number of substates per site.
        lag: Lag for RMSD computation. Default 1 compares consecutive runs.
        rmsd_state: Optional RMSDState with precomputed history.
    """
    # --- Per-site summary plot ---
    if rmsd_state is not None and hasattr(rmsd_state, "rmsd_history"):
        history = rmsd_state.rmsd_history
        cov_history = rmsd_state.coverage_history
        run_indices = rmsd_state.run_indices
        if len(history) < 2:
            return
        runs = np.array(run_indices)
        rmsds = np.array(history)
        coverages = np.array(cov_history)
        effective_lag = lag
    else:
        runs, rmsds, coverages = _collect_rmsd_from_dirs(
            input_folder, max_run, nsubs, lag,
        )
        effective_lag = lag

    if len(runs) < 2:
        return

    plot_rmsd_convergence(
        runs=runs,
        rmsds=rmsds,
        coverages=coverages,
        nsubs=nsubs,
        lag=effective_lag,
        output_path=output_dir / "rmsd_convergence.png",
    )
    print(f"RMSD convergence plot saved to {output_dir / 'rmsd_convergence.png'}")

    # --- Per-pair detail plot ---
    pair_runs, pair_data = _collect_pairwise_rmsd_from_dirs(
        input_folder, max_run, nsubs, effective_lag,
    )
    if len(pair_runs) >= 2:
        plot_pairwise_rmsd_convergence(
            runs=pair_runs,
            pair_data=pair_data,
            nsubs=nsubs,
            lag=effective_lag,
            output_path=output_dir / "rmsd_pairwise.png",
        )
        print(f"Pairwise RMSD plot saved to {output_dir / 'rmsd_pairwise.png'}")
