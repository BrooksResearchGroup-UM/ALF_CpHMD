from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cphmd.native.errors import DynamicsRunError, wrap_exception


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
    gpu_id: int,
    blade: bool,
    start: bool,
) -> SegmentResult:
    try:
        import pycharmm

        script = pycharmm.DynamicsScript(
            start=start,
            restart=not start,
            blade=blade,
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
            ntrfrq=0,
            echeck=-1,
            iuncrd=-1,
            iunwri=-1,
            iunldm=-1,
            fill_lambdata=True,
            fill_msldata=True,
        )
        script.run()
        return SegmentResult(
            lambda_matrix=_lambda_matrix_from_table(script.lambdata_bixlamsq),
            bias_matrix=_bias_matrix_from_table(script.lambdata_bias),
        )
    except DynamicsRunError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, DynamicsRunError, "running native dynamics segment") from exc


def use_blade(gpu_id: int) -> None:
    import pycharmm.lingo as lingo

    lingo.charmm_script(f"blade on gpuid {gpu_id}")


def enable_fast_routines() -> None:
    import pycharmm.lingo as lingo

    lingo.charmm_script("faster on")


def _lambda_matrix_from_table(table: Any) -> np.ndarray:
    if table is None:
        raise DynamicsRunError("lambda collection returned no data")
    values = table.drop(columns=["STEP"], errors="ignore").to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise DynamicsRunError("lambda collection has no lambda columns")
    return values


def _bias_matrix_from_table(table: Any) -> np.ndarray:
    if table is None:
        return np.empty((0, 0), dtype=np.float32)
    return table.drop(columns=["STEP"], errors="ignore").to_numpy(dtype=np.float32)
