from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class LambdaPrecision(str, Enum):
    FULL = "full"
    SHRINKER = "shrinker"


def _format_nsubsites(nsubsites: tuple[int, ...]) -> str:
    return "[" + " ".join(str(value) for value in nsubsites) + "]"


def _format_scalar(value: object) -> str:
    return str(value)


@dataclass
class ShrinkerMetadata:
    ph: float
    nblocks: int
    nsites: int
    nsubsites: tuple[int, ...] | list[int]
    lambda_scale: int
    replica_label: int | None = None
    title: str | None = None
    simulation: str | None = None
    name: str | None = None
    temperature: float | None = None
    time_step: float | None = None
    time_start: float | None = None
    time_end: float | None = None
    save_frequency: int | None = None
    start_step: int | None = None
    total_steps: int | None = None
    end_step: int | None = None

    def __post_init__(self) -> None:
        if self.nblocks <= 0:
            raise ValueError("nblocks must be positive")
        if self.nsites <= 0:
            raise ValueError("nsites must be positive")
        if self.lambda_scale <= 0:
            raise ValueError("lambda_scale must be positive")

        self.nsubsites = tuple(int(value) for value in self.nsubsites)
        if len(self.nsubsites) != self.nblocks:
            raise ValueError("nsubsites length must equal nblocks")
        if self.nsubsites[0] != 0:
            raise ValueError("nsubsites must start with 0 for the environment block")
        if any(value < 1 or value > self.nsites for value in self.nsubsites[1:]):
            raise ValueError("nsubsites values after the environment block must be in [1, nsites]")
        if self.replica_label is not None:
            self.replica_label = int(self.replica_label)
            if self.replica_label < 0:
                raise ValueError("replica_label must be non-negative")

    def to_parquet_kv(self) -> dict[str, str]:
        metadata = {
            "pH": _format_scalar(self.ph),
            "ph": _format_scalar(self.ph),
            "nblocks": _format_scalar(self.nblocks),
            "nsites": _format_scalar(self.nsites),
            "nsubsites": _format_nsubsites(self.nsubsites),
            "lambda_scale": _format_scalar(self.lambda_scale),
            "lambda_precision": "full" if self.lambda_scale == 1 else "shrinker",
        }

        optional_fields = {
            "replica_label": self.replica_label,
            "Title": self.title,
            "Simulation": self.simulation,
            "Name": self.name,
            "Temperature": self.temperature,
            "temperature": self.temperature,
            "Time Step": self.time_step,
            "Time Start": self.time_start,
            "Time End": self.time_end,
            "Save Frequency": self.save_frequency,
            "Start Step": self.start_step,
            "Total Steps": self.total_steps,
            "End Step": self.end_step,
        }
        for key, value in optional_fields.items():
            if value is not None:
                metadata[key] = _format_scalar(value)

        return metadata


def _validate_column_headers(column_headers: list[str]) -> None:
    for header in column_headers:
        if len(header.split()) != 3:
            raise ValueError(
                "each column header must split into exactly three tokens: SEGID RESID RESN"
            )


def _build_table(
    lambda_matrix: np.ndarray,
    timestamps: np.ndarray,
    column_headers: list[str],
    metadata: ShrinkerMetadata,
    precision: LambdaPrecision,
):
    import pyarrow as pa

    lambda_matrix = np.asarray(lambda_matrix)
    timestamps = np.asarray(timestamps, dtype=np.float32)

    arrays = [pa.array(timestamps)]
    names = ["time"]

    if precision is LambdaPrecision.FULL:
        if metadata.lambda_scale != 1:
            raise ValueError("FULL precision requires metadata.lambda_scale == 1")
        for idx, header in enumerate(column_headers):
            arrays.append(pa.array(np.asarray(lambda_matrix[:, idx], dtype=np.float32)))
            names.append(header)
    elif precision is LambdaPrecision.SHRINKER:
        if metadata.lambda_scale != 10000:
            raise ValueError("SHRINKER precision requires metadata.lambda_scale == 10000")

        raw = np.asarray(lambda_matrix, dtype=np.float64)
        out_of_range = (raw < 0.0) | (raw > 1.0)
        if np.any(out_of_range):
            raise ValueError(
                f"{int(np.count_nonzero(out_of_range))} lambda value(s) out of range for "
                "SHRINKER precision; expected all values in [0, 1]"
            )

        scaled = np.rint(raw * metadata.lambda_scale)
        clipped = np.clip(scaled, 0, metadata.lambda_scale).astype(np.uint16)
        for idx, header in enumerate(column_headers):
            arrays.append(pa.array(clipped[:, idx], type=pa.uint16()))
            names.append(header)
    else:
        raise ValueError(f"Unsupported precision: {precision!r}")

    return pa.table(arrays, names=names)


def write_segment_parquet(
    path: str | Path,
    lambda_matrix: np.ndarray,
    timestamps: np.ndarray,
    column_headers: list[str],
    metadata: ShrinkerMetadata,
    *,
    precision: LambdaPrecision,
) -> Path:
    lambda_matrix = np.asarray(lambda_matrix)
    timestamps = np.asarray(timestamps)

    if lambda_matrix.ndim != 2:
        raise ValueError("lambda_matrix must be 2D")
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1D")

    nrows, ncols = lambda_matrix.shape
    if timestamps.shape[0] != nrows:
        raise ValueError("timestamps length must match lambda_matrix rows")
    if len(column_headers) != ncols:
        raise ValueError("column_headers length must match lambda_matrix columns")
    if metadata.nblocks != ncols + 1:
        raise ValueError(
            "metadata.nblocks must equal the number of lambda columns plus the "
            "environment block"
        )
    if len(metadata.nsubsites) != ncols + 1:
        raise ValueError(
            "metadata.nsubsites length must equal lambda columns plus the environment block"
        )

    _validate_column_headers(column_headers)

    table = _build_table(lambda_matrix, timestamps, column_headers, metadata, precision)
    encoded_metadata = {
        key.encode("utf-8"): value.encode("utf-8")
        for key, value in metadata.to_parquet_kv().items()
    }
    table = table.replace_schema_metadata(encoded_metadata)

    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path


__all__ = [
    "LambdaPrecision",
    "ShrinkerMetadata",
    "write_segment_parquet",
]
