"""Deprecated compatibility wrappers around the native pyCHARMM system boundary."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cphmd.core import ElecType, VdwType


class _NativeSystemProxy:
    def __getattr__(self, name: str):
        from cphmd.native import system as native_system

        return getattr(native_system, name)


system = _NativeSystemProxy()


@dataclass
class BoxParameters:
    crystal_type: str
    dimensions: list[float]
    angles: list[float] = field(default_factory=lambda: [90.0, 90.0, 90.0])

    @classmethod
    def from_file(cls, box_file: Path | str) -> "BoxParameters":
        lines = Path(box_file).read_text().splitlines()
        return cls(
            crystal_type=lines[0].strip(),
            dimensions=[float(value) for value in lines[1].split()],
            angles=[float(value) for value in lines[2].split()],
        )


@dataclass
class FFTParameters:
    fftx: int
    ffty: int
    fftz: int

    @classmethod
    def from_file(cls, fft_file: Path | str) -> "FFTParameters":
        values = Path(fft_file).read_text().split()
        return cls(fftx=int(values[0]), ffty=int(values[1]), fftz=int(values[2]))


@dataclass
class NonBondedConfig:
    cutnb: float = 14.0
    cutim: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType = "vswitch"
    kappa: float = 0.320
    order: int = 6
    fftx: int | None = None
    ffty: int | None = None
    fftz: int | None = None

    @property
    def use_pme(self) -> bool:
        return self.elec_type in ("pmeex", "pmeon", "pmenn")

    def to_dict(self) -> dict:
        params = {
            "elec": True,
            "atom": True,
            "cdie": True,
            "eps": 1,
            "cutnb": self.cutnb,
            "cutim": self.cutim,
            "ctofnb": self.ctofnb,
            "ctonnb": self.ctonnb,
            "inbfrq": -1,
            "imgfrq": -1,
            "nbxmod": 5,
        }
        params["vswitch" if self.vdw_type == "vswitch" else "vfswitch"] = True
        if self.elec_type in ("pmeex", "pmeon", "pmenn"):
            params.update(
                {
                    "switch": True,
                    "ewald": True,
                    "pmewald": True,
                    "kappa": self.kappa,
                    "order": self.order,
                }
            )
            if self.fftx is not None:
                params.update({"fftx": self.fftx, "ffty": self.ffty, "fftz": self.fftz})
        elif self.elec_type == "fshift":
            params["fshift"] = True
        else:
            params["fswitch"] = True
        return params


def _stream_charmm_script(script: str, prefix: str = "charmm_utils") -> None:
    with tempfile.TemporaryDirectory(prefix=f"{prefix}_") as tmp:
        script_path = Path(tmp) / "script.inp"
        script_path.write_text(script.rstrip() + "\n")
        system.stream_file(script_path)


def read_topology_files(
    toppar_dir: Path | str,
    topology_files: list[str],
    verbose: bool = False,
) -> None:
    toppar_dir = Path(toppar_dir)
    if not verbose:
        system.set_prnlev(-1)
    system.set_bomb_level(-2)
    system.set_warn_level(-1)

    rtf_files = [path for path in topology_files if path.endswith(".rtf")]
    prm_files = [path for path in topology_files if path.endswith(".prm")]
    str_files = [path for path in topology_files if path.endswith(".str")]
    for idx, path in enumerate(rtf_files):
        system.read_rtf(toppar_dir / path, append=idx > 0)
    for idx, path in enumerate(prm_files):
        system.read_param(toppar_dir / path, append=idx > 0)
    for path in str_files:
        system.stream_file(toppar_dir / path)

    system.set_warn_level(5)
    system.set_bomb_level(0)
    if not verbose:
        system.set_prnlev(5)
    system.set_iofmt(extended=True)


def read_structure(psf_file: Path | str, crd_file: Path | str) -> None:
    system.read_psf(psf_file)
    system.read_coor(crd_file)


def setup_crystal(
    box_params: BoxParameters,
    nb_config: NonBondedConfig,
    use_image_centering: bool = True,
) -> None:
    crystal_type = box_params.crystal_type.strip().upper()
    a, b, c = box_params.dimensions
    alpha, beta, gamma = box_params.angles
    system.crystal_free()
    system.crystal_define(
        shape=crystal_type,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    system.crystal_build(cutoff=nb_config.cutim)
    if use_image_centering:
        from cphmd.native.types import AtomSelection

        system.image_setup(byres=True, segid_list=["SOLV", "IONS"])
        system.image_setup(
            byres=False,
            selection=AtomSelection(raw=".not. (segid SOLV .or. segid IONS)"),
        )


def setup_nonbonded(nb_config: NonBondedConfig) -> None:
    system.nbonds_setup(**nb_config.to_dict())


def define_selections(patch_info: pd.DataFrame) -> None:
    lines: list[str] = []
    for _, row in patch_info.iterrows():
        atoms = row["ATOMS"].split()
        atom_clause = " -\n .or. type ".join(atoms)
        lines.append(
            f"DEFine {row['SELECT']} SELEction SEGID {row['SEGID']} "
            f".AND. RESId {row['RESID']} .AND. RESName {row['PATCH']} "
            f".AND. -\n(type {atom_clause}) END"
        )
    _stream_charmm_script("\n".join(lines), "define_selections")


def execute_block_command(block_cmd: str) -> None:
    _stream_charmm_script(block_cmd, "block")


def clear_block() -> None:
    system.clear_block()


def clear_crystal() -> None:
    system.crystal_free()


def clear_noe() -> None:
    system.noe_reset()


def reset_io_unit(unit: int = 91) -> None:
    system.reset_io_unit(unit)


def get_natom() -> int:
    return system.get_natom()


def show_energy() -> None:
    system.energy_show()


def setup_shake(fast: bool = True, bonh: bool = True, tol: float = 1e-7) -> None:
    system.shake_on(fast=fast, bonh=bonh, params=True, tol=tol)


def get_gpu_id(rank: int) -> int:
    local_rank = None
    for env_var in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
    ):
        if env_var in os.environ:
            local_rank = int(os.environ[env_var])
            break
    if local_rank is None:
        local_rank = rank
    visible = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]
    if visible:
        return local_rank % len(visible)
    return local_rank


def enable_blade(gpuid: int = 0) -> None:
    system.blade_on(gpu_id=gpuid, faster=True)


class CHARMMSession:
    def __init__(
        self,
        toppar_dir: Path | str,
        topology_files: list[str],
        verbose: bool = False,
    ):
        self.toppar_dir = Path(toppar_dir)
        self.topology_files = topology_files
        self.verbose = verbose
        self._initialized = False

    def __enter__(self):
        read_topology_files(self.toppar_dir, self.topology_files, self.verbose)
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read_structure(self, psf_file: Path | str, crd_file: Path | str) -> None:
        read_structure(psf_file, crd_file)

    def setup_crystal(self, box_params: BoxParameters, nb_config: NonBondedConfig) -> None:
        setup_crystal(box_params, nb_config)

    def setup_nonbonded(self, nb_config: NonBondedConfig) -> None:
        setup_nonbonded(nb_config)


__all__ = [
    "BoxParameters",
    "FFTParameters",
    "NonBondedConfig",
    "CHARMMSession",
    "read_topology_files",
    "read_structure",
    "setup_crystal",
    "setup_nonbonded",
    "define_selections",
    "execute_block_command",
    "clear_block",
    "clear_crystal",
    "clear_noe",
    "reset_io_unit",
    "get_natom",
    "show_energy",
    "setup_shake",
    "get_gpu_id",
    "enable_blade",
]
