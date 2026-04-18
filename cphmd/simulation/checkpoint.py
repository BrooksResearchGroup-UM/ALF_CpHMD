from __future__ import annotations

import json
import os
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.utils.native_fingerprint import compute


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint cannot be resumed safely."""


class CheckpointManager:
    def __init__(
        self,
        ctx: RunContext,
        *,
        native_modules: Iterable[ModuleType],
        pycharmm_version: str | None = None,
    ):
        self.ctx = ctx
        self.native_modules = tuple(native_modules)
        self.pycharmm_version = pycharmm_version or metadata.version("pycharmm")

    def resume_or_fresh(self) -> tuple[LoopState, dict[str, Any]]:
        path = self.ctx.checkpoint_path
        if not path.exists():
            return LoopState(), {}

        payload = json.loads(path.read_text())
        self._validate(payload)
        state = LoopState(**payload["loop_state"])
        return state, payload.get("rng_state", {})

    def write(self, state: LoopState, *, rng_state: dict[str, Any]) -> Path:
        payload = {
            "schema_version": 1,
            "loop_state": asdict(state),
            "rng_state": rng_state,
            "pycharmm_version": self.pycharmm_version,
            "config_hash": self.ctx.config_hash,
            "native_api_fingerprint": compute(self.native_modules),
        }
        path = self.ctx.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
        return path

    def write_final(self, state: LoopState, *, rng_state: dict[str, Any]) -> Path:
        return self.write(state, rng_state=rng_state)

    @property
    def segment_cache_path(self) -> Path:
        return self.ctx.rank_dir / "checkpoint_segment_cache.npz"

    @property
    def bias_snapshot_path(self) -> Path:
        return self.ctx.rank_dir / "checkpoint_bias.npz"

    def write_training_sidecars(self, *, cache=None, bias_snapshot=None) -> None:
        if cache is not None:
            cache.write(self.segment_cache_path)
        if bias_snapshot is not None:
            self._write_bias_snapshot(bias_snapshot)

    def read_segment_cache(self, *, max_segments: int):
        from cphmd.training.segment_cache import SegmentCache

        if not self.segment_cache_path.exists():
            return SegmentCache(max_segments=max_segments)
        return SegmentCache.read(self.segment_cache_path)

    def read_bias_snapshot(self, *, nsubs):
        from cphmd.training.bias_snapshot import BiasSnapshot

        if not self.bias_snapshot_path.exists():
            return None
        with np.load(self.bias_snapshot_path, allow_pickle=False) as data:
            schema = int(data["schema_version"][0])
            if schema != 1:
                raise ValueError(f"unsupported bias snapshot schema {schema}")
            return BiasSnapshot.from_arrays(
                b=data["b"],
                c=data["c"],
                x=data["x"],
                s=data["s"],
                nsubs=tuple(nsubs),
            )

    def _write_bias_snapshot(self, bias_snapshot) -> Path:
        path = self.bias_snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.array([1], dtype=np.int32),
                b=bias_snapshot.b,
                c=bias_snapshot.c,
                x=bias_snapshot.x,
                s=bias_snapshot.s,
            )
        os.replace(tmp, path)
        return path

    def _validate(self, payload: dict[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "pycharmm_version": self.pycharmm_version,
            "config_hash": self.ctx.config_hash,
            "native_api_fingerprint": compute(self.native_modules),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CheckpointMismatchError(
                    f"checkpoint {key} mismatch: expected {value!r}, found {payload.get(key)!r}"
                )
