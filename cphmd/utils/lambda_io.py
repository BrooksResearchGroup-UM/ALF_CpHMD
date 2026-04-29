"""Lambda file I/O utilities.

This module provides functions for reading and writing lambda trajectory files
from CpHMD/MSLD simulations. Supports Apache Parquet (.parquet), text (.dat),
and legacy CHARMM binary (.lmd) formats.
"""

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class LambdaFileMetadata:
    """Metadata from a lambda file header.

    Attributes:
        nfile: Total number of dynamics steps in lambda file
        npriv: Number of steps preceding this run
        nsavl: Save frequency for lambda in file
        nblocks: Total number of blocks (env + subsite blocks)
        nsitemld: Number of substitution sites (R-groups)
        delta_t: Time step in ps
        title: Title from trajectory file
        temp: Temperature used in lambda dynamics thermostat
    """

    nfile: int
    npriv: int
    nsavl: int
    nblocks: int
    nsitemld: int
    delta_t: float
    title: str
    temp: float


@lru_cache(maxsize=1)
def _legacy_lmd_module():
    module_name = "cphmd.analysis.legacy_lmd_io"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    legacy_path = Path(__file__).resolve().parents[1] / "analysis" / "legacy_lmd_io.py"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy lambda reader from {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_lambda_parquet(filepath: str | Path) -> np.ndarray:
    """Read lambda data from Parquet file.

    Args:
        filepath: Path to .parquet file

    Returns:
        Lambda data array with timestamps in first column
    """
    import pyarrow.parquet as pq

    filepath = Path(filepath)
    table = pq.read_table(str(filepath))
    return np.column_stack([col.to_numpy() for col in table.columns])


def write_lambda_parquet(
    filepath: str | Path,
    lambda_data: np.ndarray,
    column_names: list[str] | None = None,
    nsubs: list[int] | None = None,
    compression: str = "snappy",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write lambda data to Parquet file.

    Args:
        filepath: Output path
        lambda_data: Array with shape (nsteps, ncols) - first col is time
        column_names: Optional column names (overrides nsubs-based naming)
        nsubs: Number of substituents per site for s{site}s{subsite} naming
        compression: Parquet compression (snappy, gzip, lz4, zstd)
        metadata: Optional key-value metadata to embed in the parquet schema
                  (e.g. pH, temperature, simulation ID)

    Returns:
        Path to written file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if column_names is None:
        n_lambdas = lambda_data.shape[1] - 1
        if nsubs is not None:
            column_names = ["time"]
            for site_idx, n_sub in enumerate(nsubs):
                for sub_idx in range(n_sub):
                    column_names.append(f"s{site_idx + 1}s{sub_idx + 1}")
        else:
            # Single-site fallback
            column_names = ["time"] + [f"s1s{i + 1}" for i in range(n_lambdas)]

    table = pa.table(
        {column_names[i]: pa.array(lambda_data[:, i]) for i in range(lambda_data.shape[1])}
    )

    if metadata:
        byte_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
        existing = table.schema.metadata or {}
        existing.update(byte_metadata)
        table = table.replace_schema_metadata(existing)

    pq.write_table(table, str(filepath), compression=compression)

    return filepath


def read_lambda(filepath: str | Path) -> np.ndarray:
    """Read lambda file (auto-detect format).

    Args:
        filepath: Path to lambda file (.lmd, .parquet, or .dat)

    Returns:
        Lambda data array
    """
    filepath = Path(filepath)

    if filepath.suffix == ".lmd":
        data, _ = _legacy_lmd_module().read_legacy_lmd_binary(filepath)
        return data
    elif filepath.suffix == ".parquet":
        return read_lambda_parquet(filepath)
    elif filepath.suffix == ".dat":
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    else:
        raise ValueError(f"Unknown lambda file format: {filepath.suffix}")


def read_lambda_values(filepath: str | Path) -> np.ndarray:
    """Read lambda values only (no time column) from any supported format.

    Handles .parquet (named columns, drops 'time'), .dat (strips col 0),
    and .lmd (binary, strips col 0).

    Args:
        filepath: Path to lambda file (.parquet, .dat, or .lmd)

    Returns:
        Lambda values array with shape (nsteps, nblocks) — no time column.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(str(filepath))
        # Drop 'time' column if present (named columns make this unambiguous)
        if "time" in table.column_names:
            table = table.drop("time")
        return np.column_stack([col.to_numpy() for col in table.columns])
    elif filepath.suffix == ".lmd":
        data, _ = _legacy_lmd_module().read_legacy_lmd_binary(filepath)
        # Binary reader prepends time as column 0
        return data[:, 1:]
    elif filepath.suffix == ".dat":
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Text files have time in column 0
        if data.shape[1] > 1:
            return data[:, 1:]
        return data
    else:
        raise ValueError(f"Unknown lambda file format: {filepath.suffix}")


def get_lambda_frame_count(filepath: str | Path, skip_e: int = 1) -> int:
    """Get the number of frames in a lambda file without loading data.

    For parquet files, reads row count from metadata (zero I/O on data pages).
    For .lmd binary files, computes from file size and record layout.
    For .dat text files, falls back to counting lines.

    Args:
        filepath: Path to lambda file (.parquet, .lmd, or .dat).
        skip_e: Subsample interval (every Nth frame). Default 1 = all.

    Returns:
        Number of frames after applying skip_e subsampling.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".parquet":
        import pyarrow.parquet as pq

        meta = pq.read_metadata(str(filepath))
        total_rows = meta.num_rows
    elif filepath.suffix == ".lmd":
        total_rows = _legacy_lmd_module().get_legacy_lmd_frame_count(filepath, skip_e)
    elif filepath.suffix == ".dat":
        with open(filepath) as f:
            total_rows = sum(1 for line in f if line.strip())
    else:
        raise ValueError(f"Unknown lambda file format: {filepath.suffix}")

    if skip_e <= 1:
        return total_rows
    # Match the (skip_e-1)::skip_e slicing pattern used by _load_simulation_data
    return len(range(skip_e - 1, total_rows, skip_e))


def read_lambda_columns(
    filepath: str | Path,
    columns: list[int] | None = None,
    skip_e: int = 1,
) -> np.ndarray:
    """Read lambda values with optional column selection and subsampling.

    When columns is None, equivalent to read_lambda_values with skip_e applied.
    When specified, reads only those column indices (0-based, after time removal).
    For parquet: uses pyarrow's column selection for zero-copy selective read.
    For .lmd/.dat: falls back to full read + column indexing.

    Args:
        filepath: Path to lambda file (.parquet, .dat, or .lmd).
        columns: Column indices to read (None = all). 0-based into lambda columns
                 (time column already excluded).
        skip_e: Subsample interval (every Nth frame). Default 1 = all.

    Returns:
        Lambda values array with shape (nframes_after_skip, ncols_selected).
    """
    filepath = Path(filepath)

    if filepath.suffix == ".parquet" and columns is not None:
        import pyarrow.parquet as pq

        table = pq.read_table(str(filepath))
        # Drop time column if present
        if "time" in table.column_names:
            table = table.drop("time")
        col_names = [table.column_names[c] for c in columns]
        subset = table.select(col_names)
        data = np.column_stack([col.to_numpy() for col in subset.columns])
    else:
        # Full read, then select columns
        data = read_lambda_values(filepath)
        if columns is not None:
            data = data[:, columns]

    # Apply skip_e subsampling
    if skip_e > 1:
        data = data[(skip_e - 1) :: skip_e, :]

    return data


def find_lambda_files(data_dir: Path, pattern: str = "Lambda.*.*") -> list[Path]:
    """Find lambda data files, preferring .parquet, then .dat, then .lmd.

    Args:
        data_dir: Directory containing lambda files.
        pattern: Base glob pattern without extension.

    Returns:
        Sorted list of lambda file paths.
    """
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob(f"{pattern}.parquet"))
    if parquet_files:
        return parquet_files
    # Fallback to text format for old runs
    dat_files = sorted(data_dir.glob(f"{pattern}.dat"))
    if dat_files:
        return dat_files
    return sorted(data_dir.glob(f"{pattern}.lmd"))


def parse_lambda_filename(filepath: str | Path) -> tuple[int, int]:
    """Parse a Lambda filename into rerun and replica indices."""
    filepath = Path(filepath)
    parts = filepath.stem.split(".")
    if len(parts) != 3 or parts[0] != "Lambda":
        raise ValueError(
            f"invalid Lambda filename '{filepath.name}'; "
            "expected Lambda.<rerun_idx>.<replica_idx>"
        )
    try:
        return int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"invalid Lambda filename '{filepath.name}'; "
            "rerun and replica indices must be integers"
        ) from exc


def group_lambda_files_by_replica(
    data_dir: Path,
    pattern: str = "Lambda.*.*",
) -> dict[int, list[Path]]:
    """Group lambda files by replica index.

    Files are sorted by their first numeric filename field and then by name.
    """
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for filepath in find_lambda_files(data_dir, pattern):
        analysis_idx, replica_idx = parse_lambda_filename(filepath)
        grouped.setdefault(replica_idx, []).append((analysis_idx, filepath))
    return {
        replica_idx: [path for _, path in sorted(entries, key=lambda item: (item[0], item[1].name))]
        for replica_idx, entries in sorted(grouped.items())
    }


def convert_lambda_to_parquet(
    input_path: str | Path, output_path: str | Path | None = None, compression: str = "snappy"
) -> Path:
    """Convert binary lambda file to Parquet format.

    Args:
        input_path: Path to .lmd file
        output_path: Output path (defaults to same name with .parquet)
        compression: Parquet compression

    Returns:
        Path to output file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)

    lambda_data, _ = _legacy_lmd_module().read_legacy_lmd_binary(input_path)
    return write_lambda_parquet(output_path, lambda_data, compression=compression)


def concatenate_lambda_files(
    filepaths: list[str | Path], output_path: str | Path | None = None
) -> np.ndarray:
    """Concatenate multiple lambda files.

    Args:
        filepaths: List of lambda files (can mix .lmd, .parquet, and .dat)
        output_path: Optional output path for combined parquet

    Returns:
        Combined lambda data array
    """
    all_data = []
    for fp in filepaths:
        data = read_lambda(fp)
        all_data.append(data)

    combined = np.vstack(all_data)

    if output_path is not None:
        write_lambda_parquet(output_path, combined)

    return combined


def get_lambda_columns_for_sites(
    nsubs: list[int],
    site_names: list[str] | None = None,
) -> list[str]:
    """Generate column names for lambda values based on site structure.

    Args:
        nsubs: Number of substituents per site
        site_names: Optional site names (unused, kept for API compatibility)

    Returns:
        List of column names like ["time", "s1s1", "s1s2", "s2s1", ...]
    """
    columns = ["time"]

    for site_idx, n_sub in enumerate(nsubs):
        for sub_idx in range(n_sub):
            columns.append(f"s{site_idx + 1}s{sub_idx + 1}")

    return columns


# CLI entry point
def main():
    """Command-line interface for lambda file conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert lambda files to Parquet")
    parser.add_argument("-i", "--input", required=True, nargs="+", help="Input lambda files (.lmd)")
    parser.add_argument("-o", "--output", help="Output directory or file")
    parser.add_argument(
        "-c",
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "lz4", "zstd"],
        help="Compression method",
    )
    parser.add_argument(
        "--concat", action="store_true", help="Concatenate all inputs into single output"
    )

    args = parser.parse_args()

    if args.concat:
        # Concatenate mode
        output = args.output or "combined.parquet"
        combined = concatenate_lambda_files(args.input, output)
        print(f"Combined {len(args.input)} files -> {output}")
        print(f"  Total steps: {len(combined)}")
    else:
        # Individual conversion mode
        for input_file in args.input:
            input_path = Path(input_file)
            if args.output:
                output_dir = Path(args.output)
                if output_dir.is_dir():
                    output_path = output_dir / input_path.with_suffix(".parquet").name
                else:
                    output_path = output_dir
            else:
                output_path = input_path.with_suffix(".parquet")

            output = convert_lambda_to_parquet(input_path, output_path, args.compression)
            print(f"Converted: {input_path} -> {output}")


if __name__ == "__main__":
    main()
