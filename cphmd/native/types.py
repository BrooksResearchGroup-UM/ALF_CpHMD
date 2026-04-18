"""Plain data types crossing the native pyCHARMM boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AtomSelection:
    """CpHMD-owned atom selection descriptor."""

    segid: str | None = None
    resid: int | str | None = None
    resname: str | None = None
    atom_name: str | None = None
    raw: str | None = None


@dataclass(frozen=True)
class AtomRecord:
    segid: str
    resid: int | str
    resname: str
    atom_name: str
    x: float
    y: float
    z: float
    mass: float
    charge: float


@dataclass(frozen=True)
class CellParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    shape: Literal["cubic", "ortho", "hexa", "tetra", "octa", "rhdo", "mono", "tric"]


@dataclass(frozen=True)
class TopologySnapshot:
    atoms: tuple[AtomRecord, ...]
    natom: int
    cell: CellParameters | None = None
