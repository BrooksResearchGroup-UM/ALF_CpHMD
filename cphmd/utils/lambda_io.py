"""Lambda file I/O utilities.

This module provides functions for reading and writing lambda trajectory files
from CpHMD/MSLD simulations. Supports both CHARMM binary (.lmd) and
Apache Parquet (.parquet) formats.

Key Features:
- Read CHARMM Fortran binary lambda files
- Read/write Apache Parquet format (fast, compressed)
- Convert between formats
- Extract metadata from simulation logs
"""

from dataclasses import dataclass
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


def read_lambda_binary(filepath: str | Path) -> tuple[np.ndarray, LambdaFileMetadata]:
    """Read CHARMM binary lambda file.

    Args:
        filepath: Path to .lmd file

    Returns:
        Tuple of (lambda_data, metadata) where lambda_data has shape (nsteps, nblocks)
        with first column being timestamps in ps

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted or has unexpected format
    """
    from scipy.io import FortranFile

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Lambda file not found: {filepath}")

    fp = FortranFile(str(filepath), 'r')

    # Read header and icntrl array
    header = fp.read_record([('hdr', np.bytes_, 4), ('icntrl', np.int32, 20)])
    icntrl = header['icntrl'][0][:]

    nfile = icntrl[0]      # Total dynamics steps
    npriv = icntrl[1]      # Steps preceding this run
    nsavl = icntrl[2]      # Save frequency
    nblocks = icntrl[6]    # Total blocks
    nsitemld = icntrl[10]  # Number of R-groups

    # Time step in ps (convert from AKMA)
    delta4 = fp.read_record(dtype=np.float32) * 4.888821477E-2
    delta_t = float(delta4[0]) if hasattr(delta4, '__len__') else float(delta4)

    # Title
    title_rec = fp.read_record(dtype=[('h', np.int32), ('title', 'S80')])
    title = title_rec['title'][0].decode().strip()

    # Skip unused records
    _ = fp.read_record(dtype=np.int32)   # nbiasv
    _ = fp.read_record(dtype=np.float32) # junk
    _ = fp.read_record(dtype=np.int32)   # isitemld (block->site mapping)

    # Temperature
    temp = float(fp.read_record(dtype=np.float32)[0])

    # Skip more unused data
    _ = fp.read_record(dtype=np.float32)

    # Bulk-read lambda values: each frame is a (lambda, theta) record pair.
    # Fortran records: [4B marker][payload][4B marker]
    # Lambda has nblocks float32s, theta has (nblocks-1) float32s.
    from numpy.lib.stride_tricks import as_strided

    pos = fp._fp.tell()
    fp.close()

    lambda_rec_bytes = nblocks * 4 + 8           # lambda payload + 2 markers
    theta_rec_bytes = (nblocks - 1) * 4 + 8      # theta payload + 2 markers
    frame_bytes = lambda_rec_bytes + theta_rec_bytes

    with open(str(filepath), 'rb') as f:
        f.seek(pos)
        buf = f.read()

    actual_steps = len(buf) // frame_bytes
    if actual_steps > 0:
        raw = np.frombuffer(buf[:actual_steps * frame_bytes], dtype=np.float32)
        # Lambda data starts at byte 4 (skip first marker) within each frame.
        # Stride between frames = frame_bytes.
        Lambda = as_strided(
            raw[1:],  # skip first marker (1 float32 = 4 bytes)
            shape=(actual_steps, nblocks),
            strides=(frame_bytes, 4),
        ).copy()[:, 1:]     # copy to contiguous array, drop env block (col 0)
    else:
        Lambda = np.zeros((0, nblocks - 1))

    # Calculate timestamps
    timestart = npriv * delta_t * nsavl
    timestep = nsavl * delta_t
    timestamps = timestart + np.arange(len(Lambda)) * timestep

    # Prepend timestamps as first column
    lambda_data = np.column_stack([timestamps, Lambda])
    lambda_data = np.round(lambda_data, 3)

    metadata = LambdaFileMetadata(
        nfile=nfile,
        npriv=npriv,
        nsavl=nsavl,
        nblocks=nblocks,
        nsitemld=nsitemld,
        delta_t=delta_t,
        title=title,
        temp=temp,
    )

    return lambda_data, metadata


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
    return np.column_stack(
        [col.to_numpy() for col in table.columns]
    )


def write_lambda_parquet(
    filepath: str | Path,
    lambda_data: np.ndarray,
    column_names: list[str] | None = None,
    compression: str = "snappy"
) -> Path:
    """Write lambda data to Parquet file.

    Args:
        filepath: Output path
        lambda_data: Array with shape (nsteps, ncols) - first col is time
        column_names: Optional column names (defaults to time, λ1, λ2, ...)
        compression: Parquet compression (snappy, gzip, lz4, zstd)

    Returns:
        Path to written file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if column_names is None:
        n_lambdas = lambda_data.shape[1] - 1
        column_names = ["time"] + [f"λ{i+1}" for i in range(n_lambdas)]

    table = pa.table(
        {column_names[i]: pa.array(lambda_data[:, i]) for i in range(lambda_data.shape[1])}
    )
    pq.write_table(table, str(filepath), compression=compression)

    return filepath


def read_lambda(filepath: str | Path) -> np.ndarray:
    """Read lambda file (auto-detect format).

    Args:
        filepath: Path to lambda file (.lmd or .parquet)

    Returns:
        Lambda data array
    """
    filepath = Path(filepath)

    if filepath.suffix == '.lmd':
        data, _ = read_lambda_binary(filepath)
        return data
    elif filepath.suffix == '.parquet':
        return read_lambda_parquet(filepath)
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

    if filepath.suffix == '.parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(str(filepath))
        # Drop 'time' column if present (named columns make this unambiguous)
        if "time" in table.column_names:
            table = table.drop("time")
        return np.column_stack(
            [col.to_numpy() for col in table.columns]
        )
    elif filepath.suffix == '.lmd':
        data, _ = read_lambda_binary(filepath)
        # Binary reader prepends time as column 0
        return data[:, 1:]
    elif filepath.suffix == '.dat':
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Text files have time in column 0
        if data.shape[1] > 1:
            return data[:, 1:]
        return data
    else:
        raise ValueError(f"Unknown lambda file format: {filepath.suffix}")


def find_lambda_files(data_dir: Path, pattern: str = "Lambda.*.*") -> list[Path]:
    """Find lambda data files, preferring .parquet over .dat (backwards compat).

    Args:
        data_dir: Directory containing lambda files.
        pattern: Base glob pattern without extension.

    Returns:
        Sorted list of lambda file paths (.parquet preferred, .dat fallback).
    """
    parquet_files = sorted(data_dir.glob(f"{pattern}.parquet"))
    if parquet_files:
        return parquet_files
    # Fallback to text format for old runs
    return sorted(data_dir.glob(f"{pattern}.dat"))


def convert_lambda_to_parquet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    compression: str = "snappy"
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
        output_path = input_path.with_suffix('.parquet')
    else:
        output_path = Path(output_path)

    lambda_data, _ = read_lambda_binary(input_path)
    return write_lambda_parquet(output_path, lambda_data, compression=compression)


def concatenate_lambda_files(
    filepaths: list[str | Path],
    output_path: str | Path | None = None
) -> np.ndarray:
    """Concatenate multiple lambda files.

    Args:
        filepaths: List of lambda files (can mix .lmd and .parquet)
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
    site_names: list[str] | None = None
) -> list[str]:
    """Generate column names for lambda values based on site structure.

    Args:
        nsubs: Number of substituents per site
        site_names: Optional site names (e.g., ["GLU35", "ASP52"])

    Returns:
        List of column names like ["time", "GLU35_0", "GLU35_1", ...]
    """
    columns = ["time"]

    for site_idx, n_sub in enumerate(nsubs):
        site_name = site_names[site_idx] if site_names else f"site{site_idx+1}"
        for sub_idx in range(n_sub):
            columns.append(f"{site_name}_{sub_idx}")

    return columns


# CLI entry point
def main():
    """Command-line interface for lambda file conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert lambda files to Parquet")
    parser.add_argument("-i", "--input", required=True, nargs="+",
                        help="Input lambda files (.lmd)")
    parser.add_argument("-o", "--output", help="Output directory or file")
    parser.add_argument("-c", "--compression", default="snappy",
                        choices=["snappy", "gzip", "lz4", "zstd"],
                        help="Compression method")
    parser.add_argument("--concat", action="store_true",
                        help="Concatenate all inputs into single output")

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
                    output_path = output_dir / input_path.with_suffix('.parquet').name
                else:
                    output_path = output_dir
            else:
                output_path = input_path.with_suffix('.parquet')

            output = convert_lambda_to_parquet(
                input_path, output_path, args.compression
            )
            print(f"Converted: {input_path} -> {output}")


if __name__ == "__main__":
    main()
