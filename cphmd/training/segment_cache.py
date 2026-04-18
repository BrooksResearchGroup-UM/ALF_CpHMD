from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SegmentCache:
    max_segments: int
    segment_ids: tuple[int, ...] = ()
    lambda_arrays: tuple[np.ndarray, ...] = ()
    bias_arrays: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if self.max_segments <= 0:
            raise ValueError("max_segments must be positive")
        if not (
            len(self.segment_ids) == len(self.lambda_arrays) == len(self.bias_arrays)
        ):
            raise ValueError("segment cache arrays must have matching lengths")

    def append(
        self,
        segment_idx: int,
        lambda_matrix: np.ndarray,
        bias_matrix: np.ndarray,
    ) -> "SegmentCache":
        ids = (*self.segment_ids, int(segment_idx))[-self.max_segments :]
        lambdas = (
            *self.lambda_arrays,
            np.asarray(lambda_matrix, dtype=np.float32).copy(),
        )[-self.max_segments :]
        biases = (
            *self.bias_arrays,
            np.asarray(bias_matrix, dtype=np.float32).copy(),
        )[-self.max_segments :]
        return SegmentCache(self.max_segments, ids, lambdas, biases)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.tmp")
        payload = {
            "schema_version": np.array([1], dtype=np.int32),
            "max_segments": np.array([self.max_segments], dtype=np.int32),
            "segment_ids": np.asarray(self.segment_ids, dtype=np.int32),
        }
        for idx, arr in enumerate(self.lambda_arrays):
            payload[f"lambda_{idx}"] = arr
        for idx, arr in enumerate(self.bias_arrays):
            payload[f"bias_{idx}"] = arr
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(tmp, out)
        return out

    @classmethod
    def read(cls, path: str | Path) -> "SegmentCache":
        src = Path(path)
        with np.load(src, allow_pickle=False) as data:
            schema = int(data["schema_version"][0])
            if schema != 1:
                raise ValueError(f"unsupported segment cache schema {schema}")
            max_segments = int(data["max_segments"][0])
            ids = tuple(int(value) for value in data["segment_ids"])
            lambdas = tuple(
                data[f"lambda_{idx}"].astype(np.float32) for idx in range(len(ids))
            )
            biases = tuple(
                data[f"bias_{idx}"].astype(np.float32) for idx in range(len(ids))
            )
        return cls(max_segments, ids, lambdas, biases)
