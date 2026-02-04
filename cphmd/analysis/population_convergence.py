"""Population convergence analysis over ALF runs.

Reads populations.dat from successive analysis directories and plots how
normalized populations evolve, separately for the relaxed (λ > 0.8) and
strict (λ > 0.985) thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_populations_from_runs(
    input_folder: Path,
    max_run: int,
) -> dict[str, np.ndarray]:
    """Read normalized populations from analysis1/ … analysisN/ directories.

    Args:
        input_folder: Parent directory containing analysisN/ subdirectories.
        max_run: Highest run index to consider.

    Returns:
        Dictionary with keys:
          - ``runs``: 1-D int array of run indices that had a valid file
          - ``relaxed``: (n_runs, n_states) normalized populations at λ > 0.8
          - ``strict``:  (n_runs, n_states) normalized populations at λ > 0.985
    """
    runs: list[int] = []
    relaxed_list: list[np.ndarray] = []
    strict_list: list[np.ndarray] = []

    for run_idx in range(1, max_run + 1):
        pop_file = input_folder / f"analysis{run_idx}" / "populations.dat"
        if not pop_file.exists():
            continue

        try:
            data = np.loadtxt(pop_file, comments="#")
        except Exception:
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Column layout (0-indexed after State column):
        #   0: State  1: Raw(>0.8)  2: Norm(>0.8)  3: Hits(>0.8)
        #   4: Raw(>0.985)  5: Norm(>0.985)  6: Hits(>0.985)
        relaxed_list.append(data[:, 2])
        strict_list.append(data[:, 5])
        runs.append(run_idx)

    if not runs:
        return {}

    return {
        "runs": np.array(runs),
        "relaxed": np.array(relaxed_list),   # (n_runs, n_states)
        "strict": np.array(strict_list),
    }


def plot_population_convergence(
    run_data: dict[str, np.ndarray],
    threshold_label: str,
    key: str,
    output_path: Path,
) -> None:
    """Create a single convergence plot for one threshold.

    Args:
        run_data: Output of :func:`read_populations_from_runs`.
        threshold_label: Human-readable threshold description, e.g. "λ > 0.8".
        key: Which population array to use (``"relaxed"`` or ``"strict"``).
        output_path: Full path for the output PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping population convergence plots")
        return

    runs = run_data["runs"]
    pops = run_data[key]  # (n_runs, n_states)
    n_states = pops.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Distinct color cycle
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for state_idx in range(n_states):
        color = cmap(state_idx % 10)
        marker = markers[state_idx % len(markers)]
        ax.plot(
            runs,
            pops[:, state_idx],
            marker=marker,
            color=color,
            linewidth=1.5,
            markersize=5,
            label=f"State {state_idx}",
        )

    # Ideal equal-population line
    ideal = 1.0 / n_states
    ax.axhline(ideal, color="grey", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Ideal (1/{n_states})")

    # Final frac_diff annotation
    final_pops = pops[-1]
    frac_diff = (final_pops.max() - final_pops.min()) * 100
    ax.annotate(
        f"Final diff = {frac_diff:.1f}%",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Normalized population", fontsize=12)
    ax.set_title(f"Population convergence ({threshold_label})", fontsize=14)
    ax.legend(loc="best", fontsize=9, ncol=max(1, n_states // 5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(runs[0] - 0.5, runs[-1] + 0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_population_plots(
    input_folder: Path,
    max_run: int,
    output_dir: Path,
) -> None:
    """Generate both population convergence plots.

    Args:
        input_folder: Parent directory containing analysisN/ subdirectories.
        max_run: Highest run index to consider.
        output_dir: Directory for output PNG files (created if needed).
    """
    run_data = read_populations_from_runs(input_folder, max_run)
    if not run_data:
        return

    # Need at least 2 runs for a meaningful convergence plot
    if len(run_data["runs"]) < 2:
        return

    plot_population_convergence(
        run_data,
        threshold_label="λ > 0.8",
        key="relaxed",
        output_path=output_dir / "populations_relaxed.png",
    )
    plot_population_convergence(
        run_data,
        threshold_label="λ > 0.985",
        key="strict",
        output_path=output_dir / "populations_strict.png",
    )
    print(f"Population convergence plots saved to {output_dir}")
