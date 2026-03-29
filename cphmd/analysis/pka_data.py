"""pKa data loading layer.

Discovers production parquet files, builds a site map from patches.dat,
loads lambda data grouped by simulation and pH, and applies boolean
cutoff thresholds for population analysis.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _get_parquet_metadata(file_path: str | Path) -> dict[str, str]:
    """Read schema-level metadata from a single parquet file.

    Returns a dict with ``file_path`` plus all decoded key-value pairs
    embedded in the Arrow schema metadata.
    """
    try:
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
