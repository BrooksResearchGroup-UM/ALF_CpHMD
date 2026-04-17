from __future__ import annotations

import json
import os
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

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
