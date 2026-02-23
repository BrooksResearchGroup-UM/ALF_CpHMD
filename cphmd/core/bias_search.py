"""
Bias Search Module for ALF CpHMD

Analyzes ALF simulation results to find optimal bias parameters
by evaluating population distributions across substituents.

This module identifies the best ALF run based on:
- Average population fraction above cutoff
- Population balance across substituents (penalizes imbalance)

Usage:
    # Programmatic
    from cphmd.core.bias_search import BiasSearchConfig, run_bias_search
    config = BiasSearchConfig(input_folder="my_system", cutoff=0.97)
    result = run_bias_search(config)

    # CLI
    python -m cphmd.core.bias_search -i my_system -c 0.97 -v
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BiasSearchConfig:
    """Configuration for bias search analysis.

    Attributes:
        input_folder: Path to simulation directory with analysis folders
        cutoff: Lambda cutoff for population counting (default: 0.97)
        adjustment: Bias adjustment type: '+' (positive), '-' (negative), '0' (none)
        alpha: Penalty factor for population imbalance in scoring
        temperature: Simulation temperature in Kelvin
        verbose: Print detailed output
        plot: Generate convergence plot
        output_dir: Directory for output variables file (default: 'variables')
    """

    input_folder: str
    cutoff: float = 0.97
    adjustment: str = "0"  # '+', '-', or '0'
    alpha: float = 10.0
    temperature: float = 298.15
    verbose: bool = False
    plot: bool = True
    output_dir: str = "variables"
    min_run: int | None = None  # Only consider analysis folders >= this run number
    max_run: int | None = None  # Only consider analysis folders <= this run number

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        if self.adjustment not in ["+", "-", "0"]:
            raise ValueError(f"Invalid adjustment type: {self.adjustment}")


@dataclass
class BiasSearchResult:
    """Results from bias search analysis.

    Attributes:
        best_iteration: Iteration number of the best run
        best_score: Score of the best run
        top5_iterations: Top 5 iteration numbers
        top5_scores: Top 5 scores
        variables_file: Path to generated variables file
        plot_file: Path to generated plot (if any)
        all_populations: Population data for all iterations
    """

    best_iteration: int
    best_score: float
    top5_iterations: list[int]
    top5_scores: list[float]
    variables_file: Path | None
    plot_file: Path | None
    all_populations: np.ndarray


def _find_analysis_folders(
    input_folder: Path,
    min_run: int | None = None,
    max_run: int | None = None,
) -> list[str]:
    """Find valid analysis folders with data.

    Args:
        input_folder: Path to simulation directory
        min_run: Only include folders with run number >= min_run
        max_run: Only include folders with run number <= max_run
    """
    folders = []
    for f in os.listdir(input_folder):
        if "analysis" in f:
            suffix = f.split("analysis")[1]
            if suffix.isdigit():
                run_num = int(suffix)
                if min_run is not None and run_num < min_run:
                    continue
                if max_run is not None and run_num > max_run:
                    continue
                folder_path = input_folder / f
                if (folder_path / "data").exists():
                    folders.append(f)

    return sorted(folders, key=lambda x: int(x.split("analysis")[1]))


def _process_folder(
    input_folder: Path, folder: str, cutoff: float
) -> tuple[int, np.ndarray, str]:
    """Process a single analysis folder.

    Returns:
        Tuple of (iteration_number, population_fractions, log_string)
    """
    iteration = int(folder.split("analysis")[1])
    data_path = input_folder / folder / "data"

    # Find and load Lambda files (prefer .parquet, fall back to .dat)
    from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values
    data_fpaths = find_lambda_files(data_path)
    if not data_fpaths:
        return None

    all_data = []
    for fpath in data_fpaths:
        dat = read_lambda_values(fpath)
        all_data.append(dat)

    data = np.concatenate(all_data, axis=0)
    num_rows = data.shape[0]

    # Calculate population fractions
    col_counts = np.sum(data > cutoff, axis=0) / num_rows
    col_means = np.mean(data, axis=0)

    # Format log string
    pct_counts = np.round(col_counts * 100, 2)
    pct_means = np.round(col_means * 100, 2)

    log_str = (
        f"Run {iteration}:\n"
        f"  Files processed: {len(data_fpaths)}\t Rows: {num_rows}\n"
        f"  Fraction > {cutoff}: {pct_counts} %\n"
        f"  Column means: {pct_means} %\n"
    )

    return (iteration, col_counts, log_str)


def _compute_scores(
    populations: list[np.ndarray], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scores for each run.

    Score = average_population - alpha * (max - min)

    Higher scores indicate better runs (high average, low imbalance).
    """
    scores = []
    for row in populations:
        avg = np.mean(row)
        diff = np.max(row) - np.min(row)
        score = avg - alpha * diff
        scores.append(score)

    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]

    return scores, sorted_indices


def _compute_adjustments(
    populations: np.ndarray, temperature: float, adjustment_type: str
) -> np.ndarray:
    """Compute bias adjustments based on population ratios.

    Uses Boltzmann weighting: adjustment = RT * ln(p_i / p_mean)
    """
    R = 0.0019872041  # kcal/(mol*K)
    RT = R * temperature

    if np.any(populations <= 0):
        print("Warning: Data contains non-positive values, using zero adjustments.")
        return np.zeros_like(populations)

    mean_pop = np.mean(populations)
    if mean_pop == 0:
        print("Warning: Mean population is zero, using zero adjustments.")
        return np.zeros_like(populations)

    adjustments = np.log(populations / mean_pop)
    adjustments -= adjustments[0]  # Normalize to first value

    if adjustment_type == "-":
        adjustments = -adjustments
    elif adjustment_type == "0":
        adjustments = np.zeros_like(adjustments)
    # '+' keeps adjustments as-is

    return adjustments * RT


def _update_variables_file(
    src_file: Path, dst_file: Path, adjustments: np.ndarray
) -> None:
    """Update variables file with bias adjustments."""
    new_lines = []
    adj_idx = 0

    with open(src_file, "r") as f:
        for line in f:
            match = re.match(
                r"^set\s+(lams\d+s\d+)\s*=\s*([-+]?\d*\.?\d+)", line.strip()
            )
            if match:
                var_name, number = match.groups()
                if adj_idx < len(adjustments):
                    old_val = float(number)
                    new_val = np.round(old_val + adjustments[adj_idx], 3)
                    new_lines.append(f"set {var_name} = {new_val:>8.3f}\n")
                    adj_idx += 1
                    continue
            new_lines.append(line)

    with open(dst_file, "w") as f:
        f.writelines(new_lines)


def _create_plot(
    iterations: list[int],
    populations: list[np.ndarray],
    top5_indices: np.ndarray,
    cutoff: float,
    output_path: Path,
) -> None:
    """Create convergence plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each substituent
    n_subs = len(populations[0])
    for sub in range(n_subs):
        ax.plot(
            iterations,
            [row[sub] for row in populations],
            label=f"Substituent {sub}",
            marker="o",
        )

    # Plot mean line
    mean_values = [np.mean(row) for row in populations]
    overall_mean = np.mean(mean_values)
    ax.axhline(y=overall_mean, label="Mean", linewidth=3, color="k", linestyle=":")

    # Highlight top runs
    for rank, idx in enumerate(top5_indices[:5]):
        iter_val = iterations[idx]
        label = f"Top Run (Rank {rank+1})" if rank == 0 else None
        ax.axvline(x=iter_val, color="red", linestyle="--", alpha=0.7, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Fraction of values > {cutoff}")
    ax.set_title("ALF Bias Search - Population Convergence")
    ax.set_ylim(0, 1 / n_subs * 1.5)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_bias_search(config: BiasSearchConfig) -> BiasSearchResult:
    """Run bias search analysis.

    Analyzes ALF simulation results to find optimal bias parameters.

    Args:
        config: BiasSearchConfig with analysis parameters

    Returns:
        BiasSearchResult with best run info and generated files
    """
    input_folder = config.input_folder

    # Find analysis folders
    folders = _find_analysis_folders(input_folder, config.min_run, config.max_run)
    if not folders:
        raise ValueError(
            f"No valid analysis folders found in {input_folder}. "
            "Ensure analysis folders contain 'data' subdirectories with Lambda files."
        )

    # Process folders (can use parallel processing for large datasets)
    results = []
    for folder in folders:
        result = _process_folder(input_folder, folder, config.cutoff)
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("No valid data found in analysis folders")

    # Sort by iteration
    results = sorted(results, key=lambda x: x[0])
    iterations, all_populations, logs = zip(*results)
    iterations = list(iterations)
    all_populations = list(all_populations)

    # Compute scores
    scores, sorted_indices = _compute_scores(all_populations, config.alpha)

    # Get top 5
    top5_indices = sorted_indices[:5]
    top5_iterations = [iterations[i] for i in top5_indices]
    top5_scores = [scores[i] for i in top5_indices]

    best_idx = top5_indices[0]
    best_iteration = iterations[best_idx]
    best_score = scores[best_idx]

    # Print results
    print(f"\nBias Search Results for {input_folder}")
    print("=" * 60)
    print(f"Analyzed {len(folders)} runs with cutoff = {config.cutoff}")
    print("\nTop 5 runs:")
    for rank, (iter_val, score) in enumerate(zip(top5_iterations, top5_scores)):
        pop = all_populations[iterations.index(iter_val)]
        avg_pct = np.round(np.mean(pop) * 100, 2)
        diff_pct = np.round((np.max(pop) - np.min(pop)) * 100, 2)
        marker = " <-- BEST" if rank == 0 else ""
        print(
            f"  Rank {rank+1}: Run {iter_val} - "
            f"avg={avg_pct}%, diff={diff_pct}%, score={score:.4f}{marker}"
        )

    if config.verbose:
        print("\nDetailed logs:")
        for log in logs:
            print(log)

    # Generate output files
    variables_file = None
    plot_file = None

    # Copy and update variables file
    src_var_file = input_folder / f"variables{best_iteration}.inp"
    if src_var_file.exists():
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Name based on system name
        system_name = input_folder.name
        dst_var_file = output_dir / f"var-{system_name}.inp"

        # Compute adjustments
        best_populations = all_populations[best_idx]
        adjustments = _compute_adjustments(
            best_populations, config.temperature, config.adjustment
        )

        if config.adjustment != "0":
            print(f"\nBias adjustments (kcal/mol): {np.round(adjustments, 3)}")

        # Copy and update
        shutil.copy(src_var_file, dst_var_file)
        _update_variables_file(dst_var_file, dst_var_file, adjustments)
        variables_file = dst_var_file
        print(f"\nVariables file saved to: {variables_file}")

    # Generate plot
    if config.plot:
        plot_file = input_folder / "bias_search.png"
        _create_plot(
            iterations, all_populations, sorted_indices, config.cutoff, plot_file
        )
        print(f"Plot saved to: {plot_file}")

    # Save detailed log
    log_file = input_folder / "bias_search.log"
    with open(log_file, "w") as f:
        f.write("Bias Search Analysis\n")
        f.write(f"Input: {input_folder}\n")
        f.write(f"Cutoff: {config.cutoff}\n")
        f.write(f"Temperature: {config.temperature} K\n")
        f.write(f"Alpha (imbalance penalty): {config.alpha}\n\n")

        for log in logs:
            f.write(log + "\n")

        f.write("\nTop 5 runs:\n")
        for rank, (iter_val, score) in enumerate(zip(top5_iterations, top5_scores)):
            pop = all_populations[iterations.index(iter_val)]
            f.write(
                f"Rank {rank+1}: Run {iter_val}, "
                f"avg={np.mean(pop)*100:.2f}%, score={score:.4f}\n"
            )

    return BiasSearchResult(
        best_iteration=best_iteration,
        best_score=best_score,
        top5_iterations=top5_iterations,
        top5_scores=top5_scores,
        variables_file=variables_file,
        plot_file=plot_file,
        all_populations=np.array(all_populations),
    )


def main():
    """CLI entry point for bias search."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze ALF simulation results to find optimal bias parameters"
    )
    parser.add_argument("-i", "--input_folder", required=True, help="Input folder")
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.97,
        help="Lambda cutoff for population counting (default: 0.97)",
    )
    parser.add_argument(
        "-a",
        "--adjustment",
        choices=["+", "0", "-"],
        default="0",
        help="Bias adjustment type: + (positive), - (negative), 0 (none)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in Kelvin (default: 298.15)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Penalty factor for population imbalance (default: 10.0)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="variables",
        help="Output directory for variables file (default: variables)",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=None,
        help="Only consider analysis folders with run number >= this value",
    )
    parser.add_argument(
        "--max-run",
        type=int,
        default=None,
        help="Only consider analysis folders with run number <= this value",
    )

    args = parser.parse_args()

    # Read temperature from prep/temp if available
    temp_file = Path(args.input_folder) / "prep" / "temp"
    temperature = args.temperature
    if temp_file.exists():
        try:
            content = temp_file.read_text().strip()
            if content:
                temperature = float(content)
                print(f"Temperature from {temp_file}: {temperature} K")
        except ValueError:
            print(f"Warning: Invalid temperature in {temp_file}, using {temperature} K")

    try:
        config = BiasSearchConfig(
            input_folder=args.input_folder,
            cutoff=args.cutoff,
            adjustment=args.adjustment,
            temperature=temperature,
            alpha=args.alpha,
            verbose=args.verbose,
            plot=not args.no_plot,
            output_dir=args.output_dir,
            min_run=args.min_run,
            max_run=args.max_run,
        )

        result = run_bias_search(config)
        print(f"\nBest run: {result.best_iteration} (score: {result.best_score:.4f})")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
