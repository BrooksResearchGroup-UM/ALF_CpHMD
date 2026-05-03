from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np

from cphmd.simulation.context import LoopState, RunContext
from cphmd.utils.native_fingerprint import compute
from cphmd.utils.pycharmm_version import resolve_pycharmm_version_string


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint cannot be resumed safely."""


class CheckpointManager:
    def __init__(
        self,
        ctx: RunContext,
        *,
        native_modules: Iterable[ModuleType],
        pycharmm_version: str | None = None,
        require_charmm_restart: bool = False,
        force_iteration_restart: bool = False,
    ):
        self.ctx = ctx
        self.native_modules = tuple(native_modules)
        self.pycharmm_version = (
            pycharmm_version if pycharmm_version is not None else resolve_pycharmm_version_string()
        )
        self.native_api_fingerprint = compute(self.native_modules)
        self.require_charmm_restart = bool(require_charmm_restart)
        self.force_iteration_restart = bool(force_iteration_restart)
        self._resume_charmm_restart_path: Path | None = None
        self._resume_coordinate_checkpoint_path: Path | None = None
        self._resume_requires_fresh_start = False
        self._resume_recovery_reason: str | None = None
        self._native_api_fingerprint_mismatch: tuple[str, str] | None = None

    def resume_or_fresh(self) -> tuple[LoopState, dict[str, Any]]:
        path = self.ctx.checkpoint_path
        if not path.exists():
            return LoopState(), {}

        payload = json.loads(path.read_text())
        self._validate(payload)
        state = LoopState(**payload["loop_state"])
        state = self._validate_charmm_resume_payload(payload, state)
        if (
            self._native_api_fingerprint_mismatch is not None
            and not self._resume_requires_fresh_start
        ):
            expected, found = self._native_api_fingerprint_mismatch
            raise CheckpointMismatchError(
                f"checkpoint native_api_fingerprint mismatch: expected {expected!r}, "
                f"found {found!r}"
            )
        return state, payload.get("rng_state", {})

    def write(
        self,
        state: LoopState,
        *,
        rng_state: dict[str, Any],
        charmm_restart: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "schema_version": 1,
            "loop_state": asdict(state),
            "rng_state": rng_state,
            "master_seed": self.ctx.master_seed,
            "pycharmm_version": self.pycharmm_version,
            "config_hash": self.ctx.config_hash,
            "native_api_fingerprint": self.native_api_fingerprint,
        }
        if charmm_restart is not None:
            payload["charmm_restart"] = charmm_restart
        path = self.ctx.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
        return path

    def write_final(
        self,
        state: LoopState,
        *,
        rng_state: dict[str, Any],
        charmm_restart: dict[str, Any] | None = None,
    ) -> Path:
        return self.write(state, rng_state=rng_state, charmm_restart=charmm_restart)

    def write_status_summary(self, state: LoopState, *, run_state: str = "running") -> Path:
        payload = {
            "schema_version": 1,
            "run_dir": str(self.ctx.run_dir),
            "state": run_state,
            "segments": state.segment_idx,
            "ranks": {
                f"rep{self.ctx.rank:02d}": {
                    "segment_idx": state.segment_idx,
                    "phase": state.phase,
                    "rex_attempted": list(state.rex_attempted),
                    "rex_accepted": list(state.rex_accepted),
                }
            },
        }
        path = self.ctx.run_dir / "status_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
        return path

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

    def charmm_restart_metadata(
        self,
        state: LoopState,
        path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        path = Path(path) if path is not None else self.ctx.restart_path
        if not path.exists():
            return None
        stat = path.stat()
        return {
            "path": self._relative_to_run_dir(path),
            "segment_idx": state.segment_idx,
            "run_idx": state.run_idx,
            "cycle_idx": state.cycle_idx,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def resume_charmm_restart_path(self) -> Path | None:
        return self._resume_charmm_restart_path

    def resume_coordinate_checkpoint_path(self) -> Path | None:
        return self._resume_coordinate_checkpoint_path

    def resume_requires_fresh_start(self) -> bool:
        return self._resume_requires_fresh_start

    def resume_recovery_reason(self) -> str | None:
        return self._resume_recovery_reason

    def prune_charmm_restarts(self, active_restart: dict[str, Any] | None) -> None:
        if active_restart is None:
            return
        active_path = self._charmm_restart_path(active_restart)
        for path in self.ctx.rank_dir.glob("checkpoint_segment_*.restart"):
            if path != active_path:
                path.unlink(missing_ok=True)

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
                **self._bias_topology_metadata(),
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

    def _bias_topology_metadata(self) -> dict[str, np.ndarray]:
        nsubs = [
            sum(1 for value in self.ctx.nsubsites[1:] if value == site)
            for site in range(1, self.ctx.nsites + 1)
        ]
        return {
            "bias_nsubs": np.asarray(nsubs, dtype=np.int32),
            "lambda_headers": np.asarray(self.ctx.lambda_headers),
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
            "master_seed": self.ctx.master_seed,
            "config_hash": self.ctx.config_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CheckpointMismatchError(
                    f"checkpoint {key} mismatch: expected {value!r}, found {payload.get(key)!r}"
                )
        found_fingerprint = payload.get("native_api_fingerprint")
        if found_fingerprint != self.native_api_fingerprint:
            self._native_api_fingerprint_mismatch = (
                self.native_api_fingerprint,
                found_fingerprint,
            )

    def _validate_charmm_resume_payload(
        self,
        payload: dict[str, Any],
        state: LoopState,
    ) -> LoopState:
        if not self.require_charmm_restart:
            return state
        if state.segment_idx <= 0 and state.run_idx <= 0 and state.cycle_idx <= 0:
            return state
        if self.force_iteration_restart:
            return self._restart_iteration_from_sidecar(
                state,
                "CHARMM restart disabled for this run",
            )
        if not payload.get("charmm_restart"):
            return self._restart_iteration_from_sidecar(
                state,
                "checkpoint lacks CHARMM restart metadata",
            )
        restart = payload["charmm_restart"]
        path = self._charmm_restart_path(restart)
        if not path.exists():
            return self._restart_iteration_from_sidecar(
                state,
                f"CHARMM restart file is missing: {path}",
            )
        if int(restart.get("segment_idx", -1)) != state.segment_idx:
            return self._restart_iteration_from_sidecar(
                state,
                "CHARMM restart segment mismatch: expected "
                f"{state.segment_idx}, found {restart.get('segment_idx')!r}",
            )
        expected_size = restart.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            return self._restart_iteration_from_sidecar(
                state,
                "CHARMM restart file size changed",
            )
        self._resume_charmm_restart_path = path
        return state

    def _restart_iteration_from_sidecar(self, state: LoopState, reason: str) -> LoopState:
        sidecar_state = self._sidecar_loop_state()
        if sidecar_state is None:
            raise CheckpointMismatchError(
                f"{reason}; no ALF bias sidecar is available for iteration restart"
            )
        if sidecar_state.segment_idx > state.segment_idx:
            raise CheckpointMismatchError(
                f"{reason}; ALF sidecar segment {sidecar_state.segment_idx} is newer "
                f"than checkpoint segment {state.segment_idx}"
            )
        if sidecar_state.cycle_idx > state.cycle_idx:
            raise CheckpointMismatchError(
                f"{reason}; ALF sidecar cycle {sidecar_state.cycle_idx} is newer "
                f"than checkpoint cycle {state.cycle_idx}"
            )
        replica_label = state.replica_label
        if (
            self.ctx.rex_enabled
            and sidecar_state.segment_idx != state.segment_idx
        ):
            replica_label = self._replica_label_from_segment(sidecar_state.segment_idx)
            if replica_label is None:
                raise CheckpointMismatchError(
                    f"{reason}; cannot recover replica_label for ALF boundary "
                    f"segment {sidecar_state.segment_idx}"
                )

        self._resume_charmm_restart_path = None
        self._resume_coordinate_checkpoint_path = self._late_recovery_coordinate_path(
            sidecar_state,
            reason,
        )
        self._resume_requires_fresh_start = True
        self._resume_recovery_reason = reason
        return replace(
            state,
            segment_idx=sidecar_state.segment_idx,
            run_idx=sidecar_state.run_idx,
            cycle_idx=sidecar_state.cycle_idx,
            phase=sidecar_state.phase,
            stop_requested=False,
            replica_label=replica_label,
            integrator_seed=None,
        )

    def _late_recovery_coordinate_path(self, state: LoopState, reason: str) -> Path | None:
        warmup_cycles = int(self.ctx.startup_minimization_segments)
        path = self.ctx.coordinate_path_for_segment(state.segment_idx)
        if path.exists():
            return path
        if warmup_cycles <= 0 or state.cycle_idx <= warmup_cycles:
            return None
        raise CheckpointMismatchError(
            f"{reason}; coordinate checkpoint is missing for ALF cycle "
            f"{state.cycle_idx}: {path}"
        )

    def _sidecar_loop_state(self) -> LoopState | None:
        if not self.bias_snapshot_path.exists():
            return None
        with np.load(self.bias_snapshot_path, allow_pickle=False) as data:
            self._validate_sidecar_config_hash(data)
            return LoopState(
                segment_idx=int(data["loop_segment_idx"][0]),
                run_idx=int(data["loop_segment_idx"][0]),
                cycle_idx=int(data["loop_cycle_idx"][0]),
                phase=int(data["loop_phase"][0]),
            )

    def _validate_sidecar_config_hash(self, data) -> None:
        if "config_hash" not in data:
            raise CheckpointMismatchError("sidecar config_hash missing")
        found_hash = str(data["config_hash"][0])
        if found_hash != self.ctx.config_hash:
            raise CheckpointMismatchError(
                f"sidecar config_hash mismatch: expected {self.ctx.config_hash!r}, "
                f"found {found_hash!r}"
            )

    def _replica_label_from_segment(self, segment_idx: int) -> int | None:
        path = self.ctx.segment_path(segment_idx)
        if not path.exists():
            return None
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
        raw = metadata.get(b"replica_label")
        if raw is None:
            return None
        return int(raw.decode("utf-8"))

    def _relative_to_run_dir(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.ctx.run_dir))
        except ValueError:
            return str(path)

    def _charmm_restart_path(self, restart: dict[str, Any]) -> Path:
        raw_path = Path(str(restart.get("path", "")))
        if raw_path.is_absolute():
            return raw_path
        return self.ctx.run_dir / raw_path
