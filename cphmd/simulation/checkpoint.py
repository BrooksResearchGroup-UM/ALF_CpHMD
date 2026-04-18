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

    def write_training_sidecars(self, *, cache=None, bias_snapshot=None, state: LoopState) -> None:
        metadata = self._sidecar_metadata(state)
        if cache is not None:
            cache.write(self.segment_cache_path, metadata=metadata)
        if bias_snapshot is not None:
            self._write_bias_snapshot(bias_snapshot, state=state)

    def read_segment_cache(self, *, max_segments: int, state: LoopState | None = None):
        from cphmd.training.segment_cache import SegmentCache

        if not self.segment_cache_path.exists():
            return SegmentCache(max_segments=max_segments)
        if state is not None:
            with np.load(self.segment_cache_path, allow_pickle=False) as data:
                self._validate_sidecar_metadata(data, state)
        return SegmentCache.read(self.segment_cache_path)

    def read_bias_snapshot(self, *, nsubs, state: LoopState | None = None):
        from cphmd.training.bias_snapshot import BiasSnapshot

        if not self.bias_snapshot_path.exists():
            return None
        with np.load(self.bias_snapshot_path, allow_pickle=False) as data:
            schema = int(data["schema_version"][0])
            if schema != 1:
                raise ValueError(f"unsupported bias snapshot schema {schema}")
            if state is not None:
                self._validate_sidecar_metadata(data, state)
            return BiasSnapshot.from_arrays(
                b=data["b"],
                c=data["c"],
                x=data["x"],
                s=data["s"],
                nsubs=tuple(nsubs),
            )

    def _write_bias_snapshot(self, bias_snapshot, *, state: LoopState) -> Path:
        path = self.bias_snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.array([1], dtype=np.int32),
                **self._sidecar_metadata(state),
                b=bias_snapshot.b,
                c=bias_snapshot.c,
                x=bias_snapshot.x,
                s=bias_snapshot.s,
            )
        os.replace(tmp, path)
        return path

    def _sidecar_metadata(self, state: LoopState) -> dict[str, np.ndarray]:
        return {
            "loop_segment_idx": np.array([state.segment_idx], dtype=np.int64),
            "loop_cycle_idx": np.array([state.cycle_idx], dtype=np.int64),
            "loop_phase": np.array([state.phase], dtype=np.int64),
            "config_hash": np.array([self.ctx.config_hash]),
        }

    def _validate_sidecar_metadata(self, data, state: LoopState) -> None:
        expected = {
            "loop_segment_idx": state.segment_idx,
            "loop_cycle_idx": state.cycle_idx,
            "loop_phase": state.phase,
        }
        display_names = {
            "loop_segment_idx": "segment_idx",
            "loop_cycle_idx": "cycle_idx",
            "loop_phase": "phase",
        }
        for key, value in expected.items():
            display_name = display_names[key]
            if key not in data:
                raise CheckpointMismatchError(f"sidecar {display_name} missing")
            found = int(data[key][0])
            if found != value:
                raise CheckpointMismatchError(
                    f"sidecar {display_name} mismatch: expected {value!r}, found {found!r}"
                )
        if "config_hash" not in data:
            raise CheckpointMismatchError("sidecar config_hash missing")
        found_hash = str(data["config_hash"][0])
        if found_hash != self.ctx.config_hash:
            raise CheckpointMismatchError(
                f"sidecar config_hash mismatch: expected {self.ctx.config_hash!r}, "
                f"found {found_hash!r}"
            )

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
