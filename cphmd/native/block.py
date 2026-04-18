from __future__ import annotations

from cphmd.native.errors import BiasRebuildError, wrap_exception


def _pyblock():
    import pycharmm.block as pyblock

    return pyblock


def clear_biases(*, pyblock=None) -> None:
    block_mod = pyblock or _pyblock()
    try:
        block_mod.clear_biases()
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "clearing native BLOCK biases") from exc


def add_bias(
    block_i: int,
    block_j: int,
    cls: int,
    ref: float,
    cforce: float,
    npower: int,
    *,
    pyblock=None,
) -> int:
    block_mod = pyblock or _pyblock()
    try:
        return int(block_mod.add_bias(block_i, block_j, cls, ref, cforce, npower))
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "adding native BLOCK bias") from exc


def set_intrinsic_bias(block_id: int, bias: float, *, pyblock=None) -> None:
    block_mod = pyblock or _pyblock()
    try:
        params = block_mod.get_ldin_params_direct(block_id)
        if params is None:
            raise BiasRebuildError(f"LDIN parameters unavailable for block {block_id}")
        ok = block_mod.set_ldin_params_direct(
            block_id,
            float(params["lambda_sq"]),
            float(params["velocity"]),
            float(params["mass"]),
            float(bias),
            float(params.get("friction", 0.0)),
        )
        if not ok:
            raise BiasRebuildError(f"failed to update LDIN bias for block {block_id}")
    except BiasRebuildError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "setting native LDIN bias") from exc
