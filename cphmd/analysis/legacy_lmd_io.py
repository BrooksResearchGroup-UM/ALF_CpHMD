"""Legacy CHARMM .lmd binary lambda file helpers."""

from pathlib import Path

import numpy as np

from cphmd.utils.lambda_io import LambdaFileMetadata


def _read_legacy_lmd_header(filepath: Path) -> tuple[dict[str, int | float | str], int]:
    from scipy.io import FortranFile

    fp = FortranFile(str(filepath), "r")

    header = fp.read_record([("hdr", np.bytes_, 4), ("icntrl", np.int32, 20)])
    icntrl = header["icntrl"][0][:]

    nfile = int(icntrl[0])
    npriv = int(icntrl[1])
    nsavl = int(icntrl[2])
    nblocks = int(icntrl[6])
    nsitemld = int(icntrl[10])

    delta4 = fp.read_record(dtype=np.float32) * 4.888821477e-2
    delta_t = float(delta4[0]) if hasattr(delta4, "__len__") else float(delta4)

    title_rec = fp.read_record(dtype=[("h", np.int32), ("title", "S80")])
    title = title_rec["title"][0].decode().strip()

    _ = fp.read_record(dtype=np.int32)
    _ = fp.read_record(dtype=np.float32)
    _ = fp.read_record(dtype=np.int32)
    temp = float(fp.read_record(dtype=np.float32)[0])
    _ = fp.read_record(dtype=np.float32)

    pos = fp._fp.tell()
    fp.close()

    return (
        {
            "nfile": nfile,
            "npriv": npriv,
            "nsavl": nsavl,
            "nblocks": nblocks,
            "nsitemld": nsitemld,
            "delta_t": delta_t,
            "title": title,
            "temp": temp,
        },
        pos,
    )


def read_legacy_lmd_binary(filepath: str | Path) -> tuple[np.ndarray, LambdaFileMetadata]:
    """Read a legacy CHARMM binary lambda file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Lambda file not found: {filepath}")

    metadata_dict, pos = _read_legacy_lmd_header(filepath)
    nblocks = int(metadata_dict["nblocks"])

    from numpy.lib.stride_tricks import as_strided

    lambda_rec_bytes = nblocks * 4 + 8
    theta_rec_bytes = (nblocks - 1) * 4 + 8
    frame_bytes = lambda_rec_bytes + theta_rec_bytes

    with open(str(filepath), "rb") as f:
        f.seek(pos)
        buf = f.read()

    actual_steps = len(buf) // frame_bytes
    if actual_steps > 0:
        raw = np.frombuffer(buf[: actual_steps * frame_bytes], dtype=np.float32)
        lambda_values = as_strided(
            raw[1:],
            shape=(actual_steps, nblocks),
            strides=(frame_bytes, 4),
        ).copy()[:, 1:]
    else:
        lambda_values = np.zeros((0, nblocks - 1))

    timestart = int(metadata_dict["npriv"]) * float(metadata_dict["delta_t"])
    timestep = int(metadata_dict["nsavl"]) * float(metadata_dict["delta_t"])
    timestamps = timestart + np.arange(len(lambda_values)) * timestep

    lambda_data = np.column_stack([timestamps, lambda_values])
    metadata = LambdaFileMetadata(
        nfile=int(metadata_dict["nfile"]),
        npriv=int(metadata_dict["npriv"]),
        nsavl=int(metadata_dict["nsavl"]),
        nblocks=nblocks,
        nsitemld=int(metadata_dict["nsitemld"]),
        delta_t=float(metadata_dict["delta_t"]),
        title=str(metadata_dict["title"]),
        temp=float(metadata_dict["temp"]),
    )

    return lambda_data, metadata


def get_legacy_lmd_frame_count(filepath: str | Path, skip_e: int = 1) -> int:
    """Return the number of frames in a legacy CHARMM binary lambda file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Lambda file not found: {filepath}")

    metadata_dict, pos = _read_legacy_lmd_header(filepath)
    nblocks = int(metadata_dict["nblocks"])

    file_size = filepath.stat().st_size
    data_bytes = file_size - pos
    lambda_rec = nblocks * 4 + 8
    theta_rec = (nblocks - 1) * 4 + 8
    frame_bytes = lambda_rec + theta_rec
    total_rows = data_bytes // frame_bytes

    if skip_e <= 1:
        return total_rows
    return len(range(skip_e - 1, total_rows, skip_e))
