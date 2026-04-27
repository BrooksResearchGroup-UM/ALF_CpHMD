"""Compact ALF optimization dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

EWMA_ALPHA = 0.1
PHASE_BG = {1: "#eef2ff", 2: "#ecfdf3", 3: "#fff1f2"}


def generate_alf_dashboard(
    input_folder: Path,
    *,
    max_run: int | None = None,
    output_dir: Path | None = None,
    nsubs: Sequence[int] | None = None,
    title: str | None = None,
    fmt: str = "png",
) -> Path | None:
    """Generate a single-dashboard summary of ALF analysis directories.

    The dashboard is intentionally run-local and overwrites a stable output
    file, so repeated ALF analysis cycles refresh the same default figure.
    """
    input_folder = Path(input_folder)
    if max_run is None:
        max_run = _latest_analysis_index(input_folder)
    if max_run is None:
        return None

    if output_dir is None:
        output_dir = input_folder / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = read_phases(input_folder, max_run)
    nsubs_list = [int(v) for v in nsubs] if nsubs is not None else None

    b_iters, b_vals = collect_bias_trajectory(input_folder, "b_sum.dat", max_run)
    c_iters, c_vals = collect_bias_trajectory(input_folder, "c_sum.dat", max_run)
    x_iters, x_vals = collect_bias_trajectory(input_folder, "x_sum.dat", max_run)
    s_iters, s_vals = collect_bias_trajectory(input_folder, "s_sum.dat", max_run)

    b_nonzero, b_labels = _b_without_block_zero(b_vals)
    c_upper, c_labels = _c_upper_triangle(c_vals)
    pop_iters, pop_rmsds, pop_transitions = collect_population_trajectory(
        input_folder,
        max_run,
        nsubs_list,
    )
    latest_rmsd, latest_pops, latest_trans = read_population_uniformity(
        input_folder,
        max_run,
        nsubs_list,
    )

    pred_b, std_b = _ewma_predict_phase23(b_iters, b_vals, phases)
    pred_c_raw, std_c_raw = _ewma_predict_phase23(c_iters, c_vals, phases)
    pred_x, std_x = _ewma_predict_phase23(x_iters, x_vals, phases)
    pred_s, std_s = _ewma_predict_phase23(s_iters, s_vals, phases)
    pred_c, std_c = _upper_prediction(pred_c_raw, std_c_raw)

    conf = _confidence_score(pop_rmsds, latest_rmsd, latest_trans)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(14, 24),
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 0.4, 0.6]},
    )
    fig.suptitle(title or input_folder.name, fontsize=12, fontweight="bold")

    _plot_bias_panel(axes[0], b_iters, b_nonzero, phases, "b_sum (linear bias)", b_labels)
    _plot_bias_panel(axes[1], c_iters, c_upper, phases, "c_sum (upper triangle)", c_labels)
    _plot_bias_panel(axes[2], x_iters, x_vals, phases, "x_sum (skew bias)")
    _plot_bias_panel(axes[3], s_iters, s_vals, phases, "s_sum (endpoint bias)")
    _plot_population_panel(axes[4], pop_iters, pop_rmsds, phases, nsubs_list)
    _plot_transition_panel(axes[5], input_folder, pop_iters, pop_transitions, phases)
    _plot_summary_panel(
        axes[6],
        max_run=max_run,
        phase=phases.get(max_run, 0),
        transitions=latest_trans,
        rmsd=latest_rmsd,
        populations=latest_pops,
        confidence=conf,
    )
    _plot_bias_table(axes[7], pred_b, std_b, pred_c, std_c, pred_x, std_x, pred_s, std_s)

    for ax in axes[:6]:
        ax.set_xlabel("ALF iteration", fontsize=7)
        ax.tick_params(labelsize=7)

    output_path = output_dir / f"alf_dashboard.{fmt}"
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _latest_analysis_index(input_folder: Path) -> int | None:
    indices = [
        int(path.name[8:])
        for path in input_folder.glob("analysis[0-9]*")
        if path.is_dir() and path.name[8:].isdigit()
    ]
    return max(indices) if indices else None


def read_phases(input_folder: Path, max_run: int) -> dict[int, int]:
    phases: dict[int, int] = {}
    for run_idx in range(max_run + 1):
        path = input_folder / f"analysis{run_idx}" / "phase.dat"
        if not path.exists():
            continue
        try:
            phases[run_idx] = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
    return phases


def collect_bias_trajectory(
    input_folder: Path,
    filename: str,
    max_run: int,
) -> tuple[list[int], list[np.ndarray]]:
    runs: list[int] = []
    values: list[np.ndarray] = []
    for run_idx in range(max_run + 1):
        path = input_folder / f"analysis{run_idx}" / filename
        if not path.exists():
            continue
        try:
            data = np.loadtxt(path)
        except (OSError, ValueError):
            continue
        runs.append(run_idx)
        values.append(np.asarray(data, dtype=float).ravel())
    return runs, values


def _b_without_block_zero(values: list[np.ndarray]) -> tuple[list[np.ndarray], list[str]]:
    if not values:
        return [], []
    trimmed = [value[1:] if value.size > 1 else value for value in values]
    nblocks = trimmed[0].size
    return trimmed, [f"b[{idx + 1}]" for idx in range(nblocks)]


def _square_size(values: np.ndarray) -> int:
    n = int(np.sqrt(values.size))
    return n if n * n == values.size else 0


def _extract_upper_triangle(values: np.ndarray, n: int) -> np.ndarray:
    matrix = values.reshape(n, n)
    return np.array([matrix[i, j] for i in range(n) for j in range(i + 1, n)])


def _upper_triangle_labels(n: int) -> list[str]:
    return [f"c({i + 1},{j + 1})" for i in range(n) for j in range(i + 1, n)]


def _c_upper_triangle(values: list[np.ndarray]) -> tuple[list[np.ndarray], list[str]]:
    if not values:
        return [], []
    n = _square_size(values[0])
    if n == 0:
        return [], []
    upper = [_extract_upper_triangle(value, n) for value in values if value.size == n * n]
    return upper, _upper_triangle_labels(n)


def read_population_uniformity(
    input_folder: Path,
    run_idx: int,
    nsubs: Sequence[int] | None,
) -> tuple[float, list[float], int]:
    pop_path = input_folder / f"analysis{run_idx}" / "populations.dat"
    if not pop_path.exists():
        return float("nan"), [], read_transitions(input_folder, run_idx)
    try:
        data = np.loadtxt(pop_path, comments="#")
    except (OSError, ValueError):
        return float("nan"), [], read_transitions(input_folder, run_idx)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        return float("nan"), [], read_transitions(input_folder, run_idx)

    populations = np.asarray(data[:, 5], dtype=float)
    targets = _population_targets(len(populations), nsubs)
    rmsd = float(np.sqrt(np.mean((populations - targets) ** 2)))
    return rmsd, populations.tolist(), read_transitions(input_folder, run_idx)


def _population_targets(nstates: int, nsubs: Sequence[int] | None) -> np.ndarray:
    if not nsubs or sum(nsubs) != nstates:
        return np.full(nstates, 1.0 / max(nstates, 1), dtype=float)
    targets: list[float] = []
    for ns in nsubs:
        targets.extend([1.0 / max(int(ns), 1)] * int(ns))
    return np.asarray(targets, dtype=float)


def read_transitions(input_folder: Path, run_idx: int) -> int:
    path = input_folder / f"analysis{run_idx}" / "transitions.dat"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^#\s*Total transitions:\s*([0-9]+)", text, flags=re.MULTILINE)
    if match is not None:
        return int(match.group(1))
    try:
        data = np.loadtxt(path, comments="#")
    except (OSError, ValueError):
        return 0
    if data.ndim == 2:
        return int(np.sum(data) - np.trace(data))
    return int(np.sum(data))


def collect_population_trajectory(
    input_folder: Path,
    max_run: int,
    nsubs: Sequence[int] | None,
) -> tuple[list[int], list[float], list[int]]:
    runs: list[int] = []
    rmsds: list[float] = []
    transitions: list[int] = []
    for run_idx in range(1, max_run + 1):
        rmsd, _, total_transitions = read_population_uniformity(input_folder, run_idx, nsubs)
        if np.isnan(rmsd):
            continue
        runs.append(run_idx)
        rmsds.append(rmsd)
        transitions.append(total_transitions)
    return runs, rmsds, transitions


def _physical_frames(input_folder: Path, run_idx: int) -> int:
    path = input_folder / f"analysis{run_idx}" / "populations.dat"
    if not path.exists():
        return 1
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"total physical:\s*([0-9]+)", text)
    return int(match.group(1)) if match is not None else 1


def _ewma_smooth(values: Sequence[float], alpha: float = EWMA_ALPHA) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _ewma_predict_phase23(
    runs: list[int],
    values: list[np.ndarray],
    phases: dict[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    phase23 = [value for run_idx, value in zip(runs, values) if phases.get(run_idx, 1) >= 2]
    if not phase23:
        return None, None
    ema = phase23[0].copy()
    compatible: list[np.ndarray] = [phase23[0]]
    for value in phase23[1:]:
        if value.shape != ema.shape:
            continue
        compatible.append(value)
        ema = EWMA_ALPHA * value + (1.0 - EWMA_ALPHA) * ema
    std = np.std(np.vstack(compatible), axis=0) if len(compatible) > 1 else np.zeros_like(ema)
    return ema, std


def _upper_prediction(
    values: np.ndarray | None,
    std: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if values is None:
        return None, None
    n = _square_size(values)
    if n == 0:
        return values, std
    upper = _extract_upper_triangle(values, n)
    upper_std = _extract_upper_triangle(std, n) if std is not None else None
    return upper, upper_std


def _confidence_score(rmsds: list[float], latest_rmsd: float, transitions: int) -> float:
    if len(rmsds) < 2 or np.isnan(latest_rmsd):
        return 0.0
    pop_score = max(0.0, 1.0 - latest_rmsd / 0.2)
    recent = rmsds[-20:]
    stability = 1.0 if np.var(recent) < 0.001 else 0.5
    transition_score = min(transitions / 100.0, 1.0)
    return float(pop_score * stability * transition_score)


def _add_phase_shading(ax, phases: dict[int, int], runs: Sequence[int]) -> None:
    if not phases or not runs:
        return
    lower = min(runs)
    upper = max(runs)
    for phase in (1, 2, 3):
        phase_runs = [
            run_idx
            for run_idx, value in sorted(phases.items())
            if value == phase and lower <= run_idx <= upper
        ]
        if not phase_runs:
            continue
        start = phase_runs[0]
        prev = start
        for run_idx in phase_runs[1:] + [None]:
            if run_idx is not None and run_idx <= prev + 1:
                prev = run_idx
                continue
            ax.axvspan(
                start - 0.5,
                prev + 0.5,
                alpha=0.15,
                color=PHASE_BG.get(phase, "white"),
                zorder=0,
            )
            if run_idx is not None:
                start = prev = run_idx


def _plot_bias_panel(
    ax,
    runs: list[int],
    values: list[np.ndarray],
    phases: dict[int, int],
    title: str,
    labels: list[str] | None = None,
) -> None:
    ax.set_title(title, fontsize=9, fontweight="bold")
    if not runs or not values:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    ncols = values[0].size
    colors = plt.cm.tab10.colors
    _add_phase_shading(ax, phases, runs)
    for idx in range(ncols):
        column = [value[idx] for value in values if value.size == ncols]
        if len(column) != len(runs):
            continue
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else f"[{idx}]"
        ax.plot(runs, column, "-", linewidth=0.7, alpha=0.45, color=color)
        phase_runs: list[int] = []
        phase_values: list[float] = []
        for run_idx, value in zip(runs, column):
            if phases.get(run_idx, 1) >= 2:
                phase_runs.append(run_idx)
                phase_values.append(value)
        if phase_values:
            ax.plot(
                phase_runs,
                _ewma_smooth(phase_values),
                "-",
                linewidth=2,
                color=color,
                label=label,
            )
    ax.set_ylabel("kcal/mol", fontsize=7)
    ax.legend(fontsize=6, loc="best", ncol=min(max(ncols, 1), 4))
    ax.grid(True, alpha=0.2)


def _plot_population_panel(
    ax,
    runs: list[int],
    rmsds: list[float],
    phases: dict[int, int],
    nsubs: Sequence[int] | None,
) -> None:
    ax.set_title("Population balance (strict endpoint)", fontsize=9, fontweight="bold")
    if not runs:
        ax.text(0.5, 0.5, "No population data", transform=ax.transAxes, ha="center")
        return
    _add_phase_shading(ax, phases, runs)
    ax.axhline(0.0, color="green", linewidth=1, alpha=0.5, label="Target")
    ax.plot(runs, rmsds, "ko-", markersize=3, linewidth=1, label="RMSD vs target")
    if len(rmsds) > 1:
        ax.plot(runs, _ewma_smooth(rmsds), "k-", linewidth=2.5, alpha=0.7, label="EWMA")
    target_text = "per-site target" if nsubs and len(nsubs) > 1 else "uniform target"
    ax.set_ylabel(target_text, fontsize=8)
    ax.set_ylim(0, max(0.3, max(rmsds) * 1.1))
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.2)


def _plot_transition_panel(
    ax,
    input_folder: Path,
    runs: list[int],
    transitions: list[int],
    phases: dict[int, int],
) -> None:
    ax.set_title("Normalized state transitions per iteration", fontsize=9, fontweight="bold")
    if not runs or not transitions:
        ax.text(0.5, 0.5, "No transition data", transform=ax.transAxes, ha="center")
        return
    _add_phase_shading(ax, phases, runs)
    normalized = [
        total / max(_physical_frames(input_folder, run_idx), 1) * 1000.0
        for run_idx, total in zip(runs, transitions)
    ]
    ax.plot(runs, normalized, "ko-", markersize=3, linewidth=1, label="Transitions/1000 frames")
    if len(normalized) > 1:
        ax.plot(runs, _ewma_smooth(normalized), "k-", linewidth=2.5, alpha=0.7, label="EWMA")
    ax.set_ylabel("Transitions / 1000 frames", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.2)


def _plot_summary_panel(
    ax,
    *,
    max_run: int,
    phase: int,
    transitions: int,
    rmsd: float,
    populations: list[float],
    confidence: float,
) -> None:
    ax.axis("off")
    rmsd_text = f"{rmsd:.4f}" if not np.isnan(rmsd) else "N/A"
    pop_text = " ".join(f"{value:.3f}" for value in populations) if populations else "N/A"
    text = (
        f"Phase:{phase} | Iter:{max_run} | Trans:{transitions} | "
        f"Pop RMSD:{rmsd_text} | Pops:[{pop_text}] | Conf:{confidence:.2f}"
    )
    bg = "#ccffcc" if confidence > 0.5 else "#ffffcc" if confidence > 0.0 else "#ffcccc"
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.9,
            boxstyle="round,pad=0.02",
            facecolor=bg,
            edgecolor="gray",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
    )


def _format_prediction(values: np.ndarray | None, std: np.ndarray | None) -> str:
    if values is None:
        return "N/A"
    parts: list[str] = []
    for idx, value in enumerate(values):
        if abs(float(value)) < 0.0005:
            continue
        item = f"[{idx}]{float(value):+.2f}"
        if std is not None and idx < len(std):
            item += f" +/-{float(std[idx]):.2f}"
        parts.append(item)
    return "  ".join(parts) if parts else "all zero"


def _format_c_prediction(values: np.ndarray | None, std: np.ndarray | None) -> str:
    if values is None:
        return "N/A"
    n = int((1 + np.sqrt(1 + 8 * len(values))) / 2)
    labels = _upper_triangle_labels(n) if n * (n - 1) // 2 == len(values) else []
    parts: list[str] = []
    for idx, value in enumerate(values):
        label = labels[idx] if idx < len(labels) else f"c[{idx}]"
        item = f"{label}{float(value):+.2f}"
        if std is not None and idx < len(std):
            item += f" +/-{float(std[idx]):.2f}"
        parts.append(item)
    return "  ".join(parts) if parts else "all zero"


def _plot_bias_table(
    ax,
    pred_b: np.ndarray | None,
    std_b: np.ndarray | None,
    pred_c: np.ndarray | None,
    std_c: np.ndarray | None,
    pred_x: np.ndarray | None,
    std_x: np.ndarray | None,
    pred_s: np.ndarray | None,
    std_s: np.ndarray | None,
) -> None:
    ax.axis("off")
    ax.set_title("Predicted biases (EWMA alpha=0.1, Phase 2/3)", fontsize=9, fontweight="bold")
    text = "\n".join(
        [
            f"b:  {_format_prediction(pred_b, std_b)}",
            f"c:  {_format_c_prediction(pred_c, std_c)}",
            f"x:  {_format_prediction(pred_x, std_x)}",
            f"s:  {_format_prediction(pred_s, std_s)}",
        ]
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.85,
            boxstyle="round,pad=0.02",
            facecolor="#f8f8f8",
            edgecolor="gray",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.04,
        0.85,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontfamily="monospace",
    )
