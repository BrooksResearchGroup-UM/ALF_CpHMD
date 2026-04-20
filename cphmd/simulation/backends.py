from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DynamicsBackend(str, Enum):
    BLADE = "blade"
    DOMDEC_CPU = "domdec-cpu"
    DOMDEC_GPU = "domdec-gpu"

    @property
    def requires_gpu(self) -> bool:
        return self in {DynamicsBackend.BLADE, DynamicsBackend.DOMDEC_GPU}

    @property
    def uses_blade(self) -> bool:
        return self is DynamicsBackend.BLADE

    @property
    def uses_domdec(self) -> bool:
        return self in {DynamicsBackend.DOMDEC_CPU, DynamicsBackend.DOMDEC_GPU}


class AnalysisBackend(str, Enum):
    CUDA_WHAM = "cuda-wham"
    DISABLED = "disabled"

    @property
    def requires_gpu(self) -> bool:
        return self is AnalysisBackend.CUDA_WHAM


@dataclass(frozen=True)
class DomdecConfig:
    dlb: bool | str | None = None
    ndir: int | None = None
    split: int | str | None = None
    ppang: float | None = None
    single: bool = False
    double: bool = False
    test: bool = False
    warn_restraints: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "DomdecConfig":
        cfg = dict(data or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"native.domdec has unsupported keys: {unknown}")
        return cls(**cfg)

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "single": self.single,
            "double": self.double,
            "test": self.test,
            "warn_restraints": self.warn_restraints,
        }
        for key in ("dlb", "ndir", "split", "ppang"):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        return kwargs


def parse_dynamics_backend(value: Any) -> DynamicsBackend:
    if isinstance(value, DynamicsBackend):
        return value
    try:
        return DynamicsBackend(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in DynamicsBackend)
        raise ValueError(f"native.dynamics_backend must be one of: {allowed}") from exc


def parse_analysis_backend(value: Any) -> AnalysisBackend:
    if isinstance(value, AnalysisBackend):
        return value
    try:
        return AnalysisBackend(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AnalysisBackend)
        raise ValueError(f"native.analysis_backend must be one of: {allowed}") from exc
