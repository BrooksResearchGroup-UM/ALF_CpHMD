"""pKa data loading layer.

Discovers production parquet files, builds a site map from patches.dat,
loads lambda data grouped by simulation and pH, and applies boolean
cutoff thresholds for population analysis.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _parquet():
    import pyarrow.parquet as pq

    return pq

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _get_parquet_metadata(file_path: str | Path) -> dict[str, str]:
    """Read schema-level metadata from a single parquet file.

    Returns a dict with ``file_path`` plus all decoded key-value pairs
    embedded in the Arrow schema metadata.
    """
    try:
        pq = _parquet()
        parquet_file = pq.ParquetFile(str(file_path))
        raw = parquet_file.schema_arrow.metadata
        if raw:
            meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
        else:
            meta = {}
        return {"file_path": str(file_path), **meta}
    except Exception as exc:
        return {"file_path": str(file_path), "error": str(exc)}


def discover_parquets(folder: Path | str, n_jobs: int = 1) -> pd.DataFrame:
    """Find all ``.parquet`` files recursively and extract schema metadata.

    Parameters
    ----------
    folder : Path
        Root directory to search.
    n_jobs : int
        Number of threads for parallel I/O (default 1 = sequential).

    Returns
    -------
    pd.DataFrame
        One row per file with columns: ``file_path``, ``pH``,
        ``Simulation``, ``nsubsites``, etc.
    """
    folder = Path(folder)
    parquet_files: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith(".parquet"):
                parquet_files.append(os.path.join(root, fname))

    if not parquet_files:
        return pd.DataFrame()

    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            metadata_list = list(pool.map(_get_parquet_metadata, parquet_files))
    else:
        metadata_list = [_get_parquet_metadata(f) for f in parquet_files]

    return pd.DataFrame(metadata_list)


# ---------------------------------------------------------------------------
# Site-map adapter
# ---------------------------------------------------------------------------


def build_site_map(patches_path: Path | str) -> pd.DataFrame:
    """Read ``patches.dat`` and return a site-map DataFrame.

    The SELECT column encodes the site/subsite indices as ``s{i}s{j}``.
    This function extracts those indices and returns a DataFrame with
    columns: ``select``, ``segid``, ``resid``, ``patch``, ``tag``,
    ``site``, ``sub``.

    Parameters
    ----------
    patches_path : Path
        Path to ``patches.dat`` (CSV with header
        ``SELECT,SEGID,RESID,PATCH,TAG``).
    """
    df = pd.read_csv(str(patches_path))
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Extract site and sub indices from the SELECT column
    extracted = df["select"].str.extract(r"s(\d+)s(\d+)")
    df["site"] = extracted[0].astype(int)
    df["sub"] = extracted[1].astype(int)

    return df[["select", "segid", "resid", "patch", "tag", "site", "sub"]]


def get_site_columns(site_map: pd.DataFrame, site_index: int) -> list[str]:
    """Return the ``s{i}s{j}`` column names for a given site index.

    Parameters
    ----------
    site_map : pd.DataFrame
        Output of :func:`build_site_map`.
    site_index : int
        1-based site index.

    Returns
    -------
    list[str]
        Sorted list of SELECT labels belonging to this site.
    """
    mask = site_map["site"] == site_index
    return sorted(site_map.loc[mask, "select"].tolist())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_parquet_columns(file_path: str, columns: list[str]) -> pd.DataFrame:
    """Read specific columns from a parquet file into a DataFrame."""
    try:
        pq = _parquet()
        table = pq.read_table(file_path, columns=columns)
        return table.to_pandas()
    except Exception:
        return pd.DataFrame()


def load_lambda_data(
    metadata_df: pd.DataFrame,
    columns: list[str],
    n_jobs: int = 1,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load lambda columns from parquets, grouped by simulation and pH.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Output of :func:`discover_parquets` (must have ``Simulation``,
        ``pH``, and ``file_path`` columns).
    columns : list[str]
        Column names to read from each parquet (e.g.
        ``["time", "s1s1", "s1s2"]``).
    n_jobs : int
        Number of threads for parallel reads.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        ``{simulation: {pH: DataFrame}}``
    """
    all_data: dict[str, dict[str, pd.DataFrame]] = {}

    for (sim, ph), group in metadata_df.groupby(["Simulation", "pH"]):
        files = sorted(group["file_path"].tolist())

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                frames = list(
                    pool.map(lambda f: _read_parquet_columns(f, columns), files)
                )
        else:
            frames = [_read_parquet_columns(f, columns) for f in files]

        frames = [f for f in frames if not f.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        all_data.setdefault(sim, {})[ph] = combined

    return all_data


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------


def apply_cutoff(
    data: dict[str, dict[str, pd.DataFrame]],
    cutoff: float = 0.97,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Convert lambda values to boolean (``lambda > cutoff``).

    Rearranges the nesting from ``{sim: {pH: df}}`` to
    ``{pH: {sim: df}}`` for downstream fitting convenience.

    Parameters
    ----------
    data : dict
        Nested dict ``{simulation: {pH: DataFrame}}``.
    cutoff : float
        Threshold; values strictly above this become ``True``.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        ``{pH: {simulation: DataFrame}}`` with boolean columns.
    """
    rearranged: dict[str, dict[str, pd.DataFrame]] = {}

    for sim, ph_data in data.items():
        for ph, df in ph_data.items():
            bool_df = df.apply(lambda col: (col > cutoff).astype(bool))
            rearranged.setdefault(ph, {})[sim] = bool_df

    return rearranged


# ---------------------------------------------------------------------------
# Equilibration skip
# ---------------------------------------------------------------------------


def skip_equilibration(
    data: dict[str, dict[str, pd.DataFrame]],
    timesteps: dict[str, dict[str, pd.DataFrame]],
    skip_ps: float,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]:
    """Remove initial equilibration frames based on simulation time.

    Parameters
    ----------
    data : dict
        ``{pH: {simulation: DataFrame}}`` with boolean state columns.
    timesteps : dict
        ``{pH: {simulation: DataFrame}}`` each having a ``time`` column (ps).
    skip_ps : float
        Amount of simulation time (in picoseconds) to discard from the start.

    Returns
    -------
    tuple
        ``(trimmed_data, trimmed_timesteps)`` with the same nesting.
    """

    new_data: dict[str, dict[str, pd.DataFrame]] = {}
    new_ts: dict[str, dict[str, pd.DataFrame]] = {}

    for ph, sim_dict in data.items():
        new_data[ph] = {}
        new_ts[ph] = {}
        for sim, df in sim_dict.items():
            ts_df = timesteps[ph][sim]
            time_vals = ts_df["time"].values
            mask = time_vals >= skip_ps
            new_data[ph][sim] = df.loc[mask].reset_index(drop=True)
            new_ts[ph][sim] = ts_df.loc[mask].reset_index(drop=True)

    return new_data, new_ts


# ---------------------------------------------------------------------------
# Sub-bin population estimation
# ---------------------------------------------------------------------------


def _process_single_bin(
    bin_data: np.ndarray, sub_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population mean and std for one time bin via sub-binning.

    Parameters
    ----------
    bin_data : np.ndarray
        2-D boolean array ``(n_frames, n_states)``.
    sub_bins : int
        Number of sub-divisions for error estimation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_per_state, std_per_state)``
    """
    import numpy as np

    n_frames = len(bin_data)

    if n_frames < sub_bins:
        total = bin_data.sum(axis=0).astype(float)
        denom = total.sum()
        if denom > 0:
            return total / denom, np.zeros(bin_data.shape[1], dtype=float)
        return np.zeros(bin_data.shape[1], dtype=float), np.zeros(
            bin_data.shape[1], dtype=float
        )

    sub_size = n_frames // sub_bins
    truncated = sub_size * sub_bins
    reshaped = bin_data[:truncated].reshape(sub_bins, sub_size, -1)

    sub_pops = reshaped.sum(axis=1).astype(float)
    row_totals = sub_pops.sum(axis=1, keepdims=True)
    np.divide(sub_pops, row_totals, out=sub_pops, where=row_totals > 0)

    return sub_pops.mean(axis=0), sub_pops.std(axis=0, ddof=1)


def compute_populations(
    data: dict[str, dict[str, pd.DataFrame]],
    timesteps: dict[str, dict[str, pd.DataFrame]],
    bin_size_ps: float = 1000.0,
    sub_bins: int = 5,
    accumulate: bool = False,
    n_jobs: int = 1,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]
]:
    """Compute per-bin populations with sub-bin error estimation.

    Parameters
    ----------
    data : dict
        ``{pH: {simulation: DataFrame}}`` with boolean state columns.
    timesteps : dict
        ``{pH: {simulation: DataFrame}}`` each having a ``time`` column (ps).
    bin_size_ps : float
        Width of each time bin in picoseconds.
    sub_bins : int
        Number of sub-bins within each bin for error estimation.
    accumulate : bool
        If ``True``, each bin includes all data from the start up to that
        bin boundary (cumulative).
    n_jobs : int
        Number of parallel workers (default 1 = sequential).

    Returns
    -------
    tuple
        ``(populations, errors)`` each ``{pH: {sim: DataFrame}}``.
    """
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    populations: dict[str, dict[str, pd.DataFrame]] = {}
    errors: dict[str, dict[str, pd.DataFrame]] = {}

    for ph, sim_dict in data.items():
        populations[ph] = {}
        errors[ph] = {}

        for sim, df in sim_dict.items():
            time_vals = timesteps[ph][sim]["time"].values
            min_t, max_t = time_vals.min(), time_vals.max()
            bin_edges = np.arange(min_t, max_t + bin_size_ps, bin_size_ps)

            data_arr = df.to_numpy()

            if not accumulate:
                grouped = [
                    data_arr[
                        (time_vals >= bin_edges[i]) & (time_vals < bin_edges[i + 1])
                    ]
                    for i in range(len(bin_edges) - 1)
                ]
            else:
                order = np.argsort(time_vals)
                sorted_data = data_arr[order]
                sorted_times = time_vals[order]
                grouped = [
                    sorted_data[: np.searchsorted(sorted_times, edge, side="right")]
                    for edge in bin_edges[1:]
                ]

            grouped = [g for g in grouped if len(g) > 0]

            if n_jobs > 1:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    results = list(
                        pool.map(
                            _process_single_bin,
                            grouped,
                            [sub_bins] * len(grouped),
                        )
                    )
            else:
                results = [_process_single_bin(g, sub_bins) for g in grouped]

            bin_pops = [r[0] for r in results]
            bin_errs = [r[1] for r in results]

            populations[ph][sim] = pd.DataFrame(bin_pops, columns=df.columns)
            errors[ph][sim] = pd.DataFrame(bin_errs, columns=df.columns)

    return populations, errors


# ---------------------------------------------------------------------------
# Total (global) population
# ---------------------------------------------------------------------------


def compute_total_population(
    data: dict[str, dict[str, pd.DataFrame]],
    bin_size: int = 500_000,
    n_jobs: int = 1,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Compute global population per pH by pooling all simulations.

    Parameters
    ----------
    data : dict
        ``{pH: {simulation: DataFrame}}`` with boolean state columns.
    bin_size : int
        Number of rows per large bin for error estimation.
    n_jobs : int
        Reserved for future parallel support.

    Returns
    -------
    tuple
        ``(populations, errors)`` each ``{pH: pd.Series}``.
    """
    populations: dict[str, pd.Series] = {}
    errors_out: dict[str, pd.Series] = {}

    for ph, sim_dict in data.items():
        frames = list(sim_dict.values())
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)

        total_sum = combined.sum().sum()
        if total_sum == 0:
            populations[ph] = pd.Series(0.0, index=combined.columns)
            errors_out[ph] = pd.Series(0.0, index=combined.columns)
            continue

        n_bins = max(1, len(combined) // bin_size)
        actual_bin = len(combined) // n_bins

        bin_pops = []
        for i in range(n_bins):
            chunk = combined.iloc[i * actual_bin : (i + 1) * actual_bin]
            chunk_total = chunk.sum().sum()
            if chunk_total == 0:
                bin_pops.append(pd.Series(0.0, index=combined.columns))
            else:
                bin_pops.append(chunk.sum(axis=0) / chunk_total)

        pop_df = pd.DataFrame(bin_pops)
        populations[ph] = pop_df.mean(axis=0)
        if len(pop_df) > 1:
            errors_out[ph] = pop_df.std(axis=0, ddof=1)
        else:
            errors_out[ph] = pd.Series(0.0, index=combined.columns)

    return populations, errors_out


# ---------------------------------------------------------------------------
# Prepare data for bootstrap / HH fitting
# ---------------------------------------------------------------------------


def prepare_fit_data(
    populations: dict[str, dict[str, pd.Series]],
    errors: dict[str, dict[str, pd.Series]],
    state_columns: list[str],
) -> tuple[np.ndarray, dict[str, list[float]], dict[str, list[float]]]:
    """Reshape per-sim populations into arrays suitable for curve fitting.

    Parameters
    ----------
    populations : dict
        ``{pH: {simulation: pd.Series}}``.
    errors : dict
        ``{pH: {simulation: pd.Series}}``.
    state_columns : list[str]
        State column names to include (e.g. ``["s1s1"]``).

    Returns
    -------
    tuple
        ``(pH_array, pop_per_pH, err_per_pH)`` where *pH_array* is a
        sorted float array of unique pH values, and the dicts map
        ``{pH_str: [per-sim values]}``.
    """
    import numpy as np

    combined_pop: dict[str, list[float]] = {}
    combined_err: dict[str, list[float]] = {}

    for ph, sim_dict in populations.items():
        for sim, pop_series in sim_dict.items():
            combined_pop.setdefault(ph, []).append(float(pop_series[state_columns[0]]))
            if errors is not None:
                combined_err.setdefault(ph, []).append(
                    float(errors[ph][sim][state_columns[0]])
                )

    pH_vals = np.array(sorted(map(float, combined_pop.keys())))
    pop_per_pH = {str(p): combined_pop[str(p)] for p in pH_vals}
    err_per_pH = {str(p): combined_err.get(str(p), []) for p in pH_vals}

    return pH_vals, pop_per_pH, err_per_pH
