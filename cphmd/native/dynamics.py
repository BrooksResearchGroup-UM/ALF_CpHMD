from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.native.errors import DynamicsRunError, wrap_exception
from cphmd.simulation.backends import DomdecConfig, DynamicsBackend, parse_dynamics_backend

_TABLE_METADATA_COLUMNS = frozenset({"STEP", "TIME"})


@dataclass(frozen=True)
class SegmentResult:
    lambda_matrix: np.ndarray
    bias_matrix: np.ndarray


def run_segment(
    *,
    nsteps: int,
    nsavl: int,
    nsavc: int,
    timestep: float,
    temperature: float,
    gpu_id: int | None,
    start: bool,
    dynamics_backend: DynamicsBackend | str | None = None,
    lambda_headers: tuple[str, ...] | list[str] | None = None,
    iseed: int | None = None,
    blade: bool | None = None,
) -> SegmentResult:
    try:
        import pycharmm

        backend = _resolve_backend(dynamics_backend, blade=blade)
        script_kwargs = dict(
            start=start,
            restart=False,
            timestep=timestep,
            nstep=nsteps,
            nsavl=nsavl,
            nsavc=nsavc,
            nprint=nsavl,
            iprfrq=nsavl,
            firstt=temperature,
            finalt=temperature,
            tstruc=temperature,
            tbath=temperature,
            ichecw=0,
            ihtfrq=0,
            ieqfrq=0,
            iasors=1,
            iasvel=1,
            iscvel=0,
            inbfrq=0,
            ilbfrq=0,
            imgfrq=0,
            ntrfrq=_ntrfrq_for_backend(backend, nsavl=nsavl),
            echeck=-1,
            iuncrd=-1,
            iunwri=-1,
            lambdata=True,
        )
        if backend is DynamicsBackend.BLADE:
            script_kwargs["blade"] = True
        if iseed is not None:
            _set_rng_seeds(pycharmm, int(iseed))
        with tempfile.TemporaryDirectory(prefix="cphmd_lambda_") as tmpdir:
            lambda_parquet_path = Path(tmpdir) / "segment.parquet"
            script_kwargs["lambda_parquet"] = lambda_parquet_path
            script = pycharmm.DynamicsScript(**script_kwargs)
            script.run()
            lambda_matrix = _lambda_matrix_from_parquet(
                lambda_parquet_path,
                expected_columns=len(lambda_headers) if lambda_headers is not None else None,
            )
            return SegmentResult(
                lambda_matrix=lambda_matrix,
                bias_matrix=_bias_matrix_from_table(
                    script.lambdata_bias,
                    allow_missing=True,
                    row_count=lambda_matrix.shape[0],
                ),
            )
    except DynamicsRunError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "running native dynamics segment") from exc


def use_blade(gpu_id: int) -> None:
    try:
        import pycharmm.blade as blade

        # GPUIDS changes take effect only across a BLADE OFF/ON cycle in CHARMM.
        blade.disable()
        blade.enable(gpu_ids=gpu_id)
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "enabling BLaDE") from exc


def use_domdec(
    *,
    gpu: bool,
    gpu_id: int | None,
    config: DomdecConfig | None = None,
) -> None:
    try:
        from cphmd.native import domdec

        domdec.enable(gpu=gpu, gpu_id=gpu_id, config=config or DomdecConfig())
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "enabling DOMDEC") from exc


def preflight_blade_energy() -> None:
    try:
        import pycharmm.blade as blade

        blade.energy(show=False)
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "preflighting BLaDE energy") from exc


def preflight_domdec_energy(*, gpu: bool) -> None:
    try:
        from cphmd.native import domdec

        domdec.energy(gpu=gpu)
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "preflighting DOMDEC energy") from exc


def _resolve_backend(
    dynamics_backend: DynamicsBackend | str | None,
    *,
    blade: bool | None,
) -> DynamicsBackend:
    if dynamics_backend is not None:
        return parse_dynamics_backend(dynamics_backend)
    if blade is None:
        return DynamicsBackend.BLADE
    return DynamicsBackend.BLADE if blade else DynamicsBackend.DOMDEC_CPU


def _ntrfrq_for_backend(backend: DynamicsBackend, *, nsavl: int) -> int:
    if backend is DynamicsBackend.BLADE:
        return 0
    return max(1, int(nsavl))


def enable_fast_routines() -> None:
    import pycharmm.miscom as miscom

    miscom.faster(True)


def _lambda_matrix_from_table(
    table: Any,
    *,
    expected_columns: int | None = None,
) -> np.ndarray:
    if table is None:
        raise DynamicsRunError("lambda collection returned no data")

    columns_by_upper = {str(column).strip().upper(): column for column in table.columns}
    xlm2_column = columns_by_upper.get("XLM2")
    if xlm2_column is not None and len(_data_columns(table)) == 1:
        values = table[xlm2_column].to_numpy(dtype=np.float32)
        if expected_columns is not None:
            if values.shape[0] == expected_columns + 1:
                values = values[1:]
            elif values.shape[0] != expected_columns:
                raise DynamicsRunError(
                    "lambda collection has "
                    f"{values.shape[0]} BLOCK rows; expected {expected_columns} "
                    f"lambda columns or {expected_columns + 1} rows including environment"
                )
        return values.reshape(1, -1)

    values = table.drop(
        columns=_metadata_columns(table),
        errors="ignore",
    ).to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise DynamicsRunError("lambda collection has no lambda columns")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise DynamicsRunError(
            f"lambda collection has {values.shape[1]} lambda columns; expected {expected_columns}"
        )
    return values


def _lambda_matrix_from_parquet(
    path: str | Path,
    *,
    expected_columns: int | None = None,
) -> np.ndarray:
    import pandas as pd

    frame = pd.read_parquet(path)
    values = frame.drop(
        columns=_metadata_columns(frame),
        errors="ignore",
    ).to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise DynamicsRunError("lambda parquet did not return a 2D matrix")
    if values.shape[1] == 0:
        raise DynamicsRunError("lambda parquet has no lambda columns")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise DynamicsRunError(
            f"lambda parquet has {values.shape[1]} lambda columns; expected {expected_columns}"
        )
    return values


def _bias_matrix_from_table(
    table: Any,
    *,
    allow_missing: bool = False,
    row_count: int | None = None,
) -> np.ndarray:
    if table is None:
        if allow_missing:
            return np.zeros((int(row_count or 0), 0), dtype=np.float32)
        raise DynamicsRunError("bias collection returned no data")
    return table.drop(columns=_metadata_columns(table), errors="ignore").to_numpy(dtype=np.float32)


def _metadata_columns(table: Any) -> list[Any]:
    return [
        column for column in table.columns if str(column).strip().upper() in _TABLE_METADATA_COLUMNS
    ]


def _data_columns(table: Any) -> list[Any]:
    return [column for column in table.columns if column not in _metadata_columns(table)]


def _rng_seed_vector(pycharmm_module: Any, seed: int) -> list[int]:
    dynamics_module = _pycharmm_dynamics_module(pycharmm_module)
    get_nrand = getattr(dynamics_module, "get_nrand", None)
    nrand = int(get_nrand()) if get_nrand is not None else 1
    if nrand <= 0:
        raise DynamicsRunError(f"pyCHARMM reported invalid RNG seed count {nrand}")
    base = seed & 0x7FFFFFFF
    return [((base + idx * 104729) & 0x7FFFFFFF) or 1 for idx in range(nrand)]


def _set_rng_seeds(pycharmm_module: Any, seed: int) -> None:
    dynamics_module = _pycharmm_dynamics_module(pycharmm_module)
    setter = getattr(dynamics_module, "set_rngseeds", None)
    if setter is None:
        setter = getattr(dynamics_module, "set_iseed", None)
    if setter is None:
        raise DynamicsRunError("pyCHARMM dynamics module does not expose RNG seed setter")
    seeds = _rng_seed_vector(pycharmm_module, seed)
    ok = setter(seeds)
    if ok is False:
        raise DynamicsRunError("pyCHARMM rejected RNG seed vector")


def _pycharmm_dynamics_module(pycharmm_module: Any):
    dynamics_module = getattr(pycharmm_module, "dynamics", None)
    if dynamics_module is not None:
        return dynamics_module
    import pycharmm.dynamics as dynamics_module

    return dynamics_module
