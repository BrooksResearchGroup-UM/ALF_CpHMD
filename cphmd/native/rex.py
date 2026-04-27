from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cphmd.native.errors import ExchangeTransportError, wrap_exception


def build_exchanger(comm=None, seed: int | None = None):
    try:
        from pycharmm.replica_exchange import Exchanger

        return Exchanger(comm=comm, seed=seed, abort_on_error=False)
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "building pyCHARMM replica exchanger",
        ) from exc


def snapshot_state(
    exchanger,
    label: int,
    ldin_blocks: Sequence[int],
    msld_theta_blocks: Sequence[int] | None = None,
    include_lambdas: bool = True,
    include_ph: bool = True,
    include_temperature: bool = False,
    include_ffix: bool = False,
    metadata: dict[str, Any] | None = None,
):
    try:
        from pycharmm.replica_exchange import State

        state = exchanger.read_state(
            label=label,
            include_lambdas=include_lambdas,
            include_ph=include_ph,
            include_temperature=include_temperature,
            include_ffix=include_ffix,
            ldin_blocks=list(ldin_blocks),
            msld_theta_blocks=list(msld_theta_blocks or ()),
            metadata=metadata or {},
        )
        if not isinstance(state, State):
            raise TypeError("Exchanger.read_state() did not return pycharmm.replica_exchange.State")
        return state
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "reading pyCHARMM replica exchange state",
        ) from exc


def attempt_neighbor_swap(
    exchanger,
    attempt_idx: int,
    state,
    acceptance_fn: Callable[[Any, Any], float],
):
    try:
        return exchanger.attempt_neighbor_exchange(
            attempt_idx,
            state,
            acceptance_fn,
            apply_state=True,
        )
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "attempting pyCHARMM neighbor exchange",
        ) from exc


def ph_lambda_probability(
    lambda_i: np.ndarray,
    lambda_j: np.ndarray,
    signs: np.ndarray,
    ph_i: float,
    ph_j: float,
) -> float:
    try:
        from pycharmm.replica_criteria import ph_lambda_probability as _probability

        return float(_probability(lambda_i, lambda_j, signs, ph_i, ph_j))
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "computing pH/lambda exchange probability",
        ) from exc
