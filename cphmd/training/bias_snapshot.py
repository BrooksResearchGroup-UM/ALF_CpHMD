from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHI_OFFSET = 0.017
OMEGA_DECAY = -5.56


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

    @classmethod
    def from_block_str(
        cls,
        block_path: str | Path,
        *,
        nsubs: tuple[int, ...] | list[int],
        chi_offset: float = CHI_OFFSET,
        omega_decay: float = OMEGA_DECAY,
    ) -> "BiasSnapshot":
        """Parse LDIN/LDBV bias state from a CHARMM ``block.str`` file."""
        nsubs_t = tuple(int(value) for value in nsubs)
        nblocks = sum(nsubs_t)
        b = np.zeros((1, nblocks), dtype=np.float64)
        c = np.zeros((nblocks, nblocks), dtype=np.float64)
        x = np.zeros((nblocks, nblocks), dtype=np.float64)
        s = np.zeros((nblocks, nblocks), dtype=np.float64)

        for raw_line in Path(block_path).read_text().splitlines():
            line = raw_line.split("!", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            keyword = tokens[0].upper()
            if keyword == "LDIN" and len(tokens) >= 6:
                block_id = int(tokens[1])
                if block_id <= 1:
                    continue
                idx = block_id - 2
                if 0 <= idx < nblocks:
                    b[0, idx] = float(tokens[5])
            elif keyword == "LDBV" and len(tokens) >= 7:
                block_i = int(tokens[2]) - 2
                block_j = int(tokens[3]) - 2
                cls_id = int(tokens[4])
                cforce = float(tokens[6])
                if not (0 <= block_i < nblocks and 0 <= block_j < nblocks):
                    continue
                if cls_id == 6:
                    c[block_i, block_j] = -cforce
                elif cls_id == 8:
                    s[block_i, block_j] = -cforce
                elif cls_id == 10:
                    x[block_i, block_j] = -cforce

        return cls.from_arrays(
            b=b,
            c=c,
            x=x,
            s=s,
            nsubs=nsubs_t,
            chi_offset=chi_offset,
            omega_decay=omega_decay,
        )
