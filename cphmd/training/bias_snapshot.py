from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cphmd.core.bias_constants import CHI_OFFSET, OMEGA_DECAY


@dataclass(frozen=True)
class LDBVTerm:
    block_i: int
    block_j: int
    cls: int
    ref: float
    cforce: float
    npower: int = 0

    def as_tuple(self) -> tuple[int, int, int, float, float, int]:
        return (self.block_i, self.block_j, self.cls, self.ref, self.cforce, self.npower)


@dataclass(frozen=True)
class BiasSnapshot:
    b: np.ndarray
    c: np.ndarray
    x: np.ndarray
    s: np.ndarray
    nsubs: tuple[int, ...]
    intrinsic_biases: tuple[tuple[int, float], ...]
    ldbv_terms: tuple[LDBVTerm, ...]

    @classmethod
    def from_arrays(
        cls,
        *,
        b: np.ndarray,
        c: np.ndarray,
        x: np.ndarray,
        s: np.ndarray,
        nsubs: tuple[int, ...] | list[int],
        chi_offset: float = CHI_OFFSET,
        omega_decay: float = OMEGA_DECAY,
    ) -> "BiasSnapshot":
        nsubs_t = tuple(int(value) for value in nsubs)
        nblocks = sum(nsubs_t)
        b_arr = np.asarray(b, dtype=np.float64)
        if b_arr.shape == (nblocks,):
            b_arr = b_arr.reshape(1, nblocks)
        c_arr = np.asarray(c, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        s_arr = np.asarray(s, dtype=np.float64)
        if b_arr.shape != (1, nblocks):
            raise ValueError(f"b must have shape (1, {nblocks}) or ({nblocks},)")
        for name, arr in (("c", c_arr), ("x", x_arr), ("s", s_arr)):
            if arr.shape != (nblocks, nblocks):
                raise ValueError(f"{name} must have shape ({nblocks}, {nblocks})")

        intrinsic = tuple((idx + 2, float(b_arr[0, idx])) for idx in range(nblocks))
        terms: list[LDBVTerm] = []
        for i in range(nblocks):
            for j in range(i + 1, nblocks):
                value = -float(c_arr[i, j])
                if value != 0.0:
                    terms.append(LDBVTerm(i + 2, j + 2, 6, 0.0, value))
        for i in range(nblocks):
            for j in range(nblocks):
                if i == j:
                    continue
                value = -float(s_arr[i, j])
                if value != 0.0:
                    terms.append(LDBVTerm(i + 2, j + 2, 8, float(chi_offset), value))
        for i in range(nblocks):
            for j in range(nblocks):
                if i == j:
                    continue
                value = -float(x_arr[i, j])
                if value != 0.0:
                    terms.append(LDBVTerm(i + 2, j + 2, 10, float(omega_decay), value))

        return cls(
            b=b_arr.copy(),
            c=c_arr.copy(),
            x=x_arr.copy(),
            s=s_arr.copy(),
            nsubs=nsubs_t,
            intrinsic_biases=intrinsic,
            ldbv_terms=tuple(terms),
        )

    @classmethod
    def from_analysis_dir(
        cls,
        analysis_dir: str | Path,
        *,
        nsubs: tuple[int, ...] | list[int],
        chi_offset: float = CHI_OFFSET,
        omega_decay: float = OMEGA_DECAY,
    ) -> "BiasSnapshot":
        root = Path(analysis_dir)
        return cls.from_arrays(
            b=np.loadtxt(root / "b_sum.dat"),
            c=np.loadtxt(root / "c_sum.dat"),
            x=np.loadtxt(root / "x_sum.dat"),
            s=np.loadtxt(root / "s_sum.dat"),
            nsubs=tuple(nsubs),
            chi_offset=chi_offset,
            omega_decay=omega_decay,
        )
