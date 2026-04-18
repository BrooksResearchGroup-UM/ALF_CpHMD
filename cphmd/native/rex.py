from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cphmd.native.errors import ExchangeTransportError, wrap_exception


def build_exchanger(comm=None, seed: int | None = None):
    try:
        from pycharmm.replica_exchange import DirectLabelExchanger

        return DirectLabelExchanger(comm=comm, seed=seed)
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "building direct label exchanger",
        ) from exc


def snapshot_state(
    exchanger,
    label: int,
    ldin_blocks: Sequence[int],
    include_lambdas: bool = True,
    include_ph: bool = True,
    include_temperature: bool = False,
    include_ffix: bool = False,
    metadata: dict[str, Any] | None = None,
):
    try:
        return exchanger.snapshot(
            label=label,
            include_lambdas=include_lambdas,
            include_ph=include_ph,
            include_temperature=include_temperature,
            include_ffix=include_ffix,
            ldin_blocks=list(ldin_blocks),
            metadata=metadata or {},
        )
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "snapshotting direct exchange state",
        ) from exc


def attempt_neighbor_swap(
    exchanger,
    attempt_idx: int,
    state,
    acceptance_fn: Callable[[Any, Any], float],
    apply_state: bool = True,
):
    try:
        return exchanger.attempt_neighbor_exchange(
            attempt_idx,
            state,
            acceptance_fn,
            apply_state=apply_state,
        )
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "attempting direct neighbor exchange",
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


def neighbor_partner(rank: int, size: int, attempt_idx: int) -> int | None:
    try:
        from pycharmm.replica_exchange import neighbor_partner as _neighbor_partner

        return _neighbor_partner(rank, size, attempt_idx)
    except Exception as exc:
        raise wrap_exception(
            exc,
            ExchangeTransportError,
            "computing neighbor exchange partner",
        ) from exc
