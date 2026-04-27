from __future__ import annotations

from contextlib import contextmanager, nullcontext

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


@contextmanager
def modify_biases(*, pyblock=None):
    block_mod = pyblock or _pyblock()
    modifier = getattr(block_mod, "modify", None)
    try:
        with (modifier() if modifier is not None else nullcontext()):
            yield
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "modifying native BLOCK biases") from exc


def add_bias(
    block_i: int,
    block_j: int,
    cls: int,
    ref: float,
    cforce: float,
    npower: int,
    index: int | None = None,
    *,
    pyblock=None,
) -> int | None:
    block_mod = pyblock or _pyblock()
    try:
        kwargs = {"index": int(index)} if index is not None else {}
        result = block_mod.add_bias(block_i, block_j, cls, ref, cforce, npower, **kwargs)
        if result is None:
            return None
        return int(result)
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "adding native BLOCK bias") from exc


def set_bias_count(count: int, *, pyblock=None) -> None:
    block_mod = pyblock or _pyblock()
    try:
        block_mod.set_bias_count(int(count))
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "allocating native BLOCK bias slots") from exc


def get_bias_count(*, pyblock=None) -> int | None:
    block_mod = pyblock or _pyblock()
    try:
        biases = getattr(block_mod, "get_biases", None)
        python_count = len(biases()) if biases is not None else None
        getter = getattr(block_mod, "get_nbiasv_direct", None)
        if getter is not None:
            value = getter()
            if value is not None:
                direct_count = int(value)
                if direct_count > 0 or python_count is None:
                    return direct_count
        return python_count
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "reading native BLOCK bias count") from exc


def get_bias_params(index: int, *, pyblock=None) -> dict | None:
    block_mod = pyblock or _pyblock()
    try:
        getter = getattr(block_mod, "get_bias_params_direct", None)
        if getter is not None:
            params = getter(int(index))
            if params is not None:
                return {
                    "index": int(index),
                    "block_i": int(params["block_i"]),
                    "block_j": int(params["block_j"]),
                    "cls": int(params["cls"]),
                    "ref": float(params.get("ref_up", params.get("ref", 0.0))),
                    "cforce": float(params.get("force_const", params.get("cforce", 0.0))),
                    "npower": int(params.get("power", params.get("npower", 0))),
                }
        python_biases = getattr(block_mod, "get_biases", None)
        if python_biases is not None:
            for item in python_biases():
                if int(item.get("index", 0)) == int(index):
                    return {
                        "index": int(index),
                        "block_i": int(item["block_i"]),
                        "block_j": int(item["block_j"]),
                        "cls": int(item["cls"]),
                        "ref": float(item["ref"]),
                        "cforce": float(item["cforce"]),
                        "npower": int(item["npower"]),
                    }
        return None
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "reading native BLOCK bias parameters") from exc


def get_ldin_params(block_id: int, *, pyblock=None) -> dict | None:
    block_mod = pyblock or _pyblock()
    try:
        getter = getattr(block_mod, "get_ldin_params_direct", None)
        if getter is None:
            return None
        params = getter(int(block_id))
        if params is None:
            return None
        return dict(params)
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "reading native LDIN parameters") from exc


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


def set_ldin_params(
    block_id: int,
    lambda_sq: float,
    velocity: float,
    mass: float,
    bias: float,
    friction: float = 0.0,
    *,
    pyblock=None,
) -> None:
    block_mod = pyblock or _pyblock()
    try:
        ok = block_mod.set_ldin_params_direct(
            int(block_id),
            float(lambda_sq),
            float(velocity),
            float(mass),
            float(bias),
            float(friction),
        )
        if not ok:
            raise BiasRebuildError(f"failed to update LDIN parameters for block {block_id}")
    except BiasRebuildError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "setting native LDIN parameters") from exc


def set_ph(ph: float, *, pyblock=None) -> None:
    block_mod = pyblock or _pyblock()
    try:
        setter = getattr(block_mod, "set_ph_direct", None)
        if setter is not None:
            ok = setter(float(ph))
            if ok:
                return
        fallback = getattr(block_mod, "phmd_ph", None)
        if fallback is not None:
            fallback(float(ph))
            return
        raise BiasRebuildError("pyCHARMM BLOCK pH setter is unavailable")
    except BiasRebuildError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "setting native pH") from exc


def get_fnex(*, pyblock=None) -> float:
    block_mod = pyblock or _pyblock()
    try:
        getter = getattr(block_mod, "get_fnex_direct", None)
        if getter is None:
            raise BiasRebuildError("pyCHARMM BLOCK FNEX getter is unavailable")
        value = getter()
        if value is None:
            raise BiasRebuildError("native BLOCK FNEX is unavailable")
        return float(value)
    except BiasRebuildError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "reading native FNEX") from exc


def sync_state(*, pyblock=None) -> None:
    block_mod = pyblock or _pyblock()
    try:
        sync = getattr(block_mod, "sync_state_from_charmm", None)
        if sync is not None:
            ok = sync()
            if not ok:
                raise BiasRebuildError("failed to synchronize native BLOCK state")
        try:
            import pycharmm.blade as blade
        except ImportError:
            return
        is_enabled = getattr(blade, "is_enabled", None)
        blade_sync = getattr(blade, "sync_state_from_charmm", None)
        if is_enabled is not None and blade_sync is not None and is_enabled():
            ok = blade_sync()
            if not ok:
                raise BiasRebuildError("failed to synchronize live BLaDE state")
    except BiasRebuildError:
        raise
    except Exception as exc:
        raise wrap_exception(exc, BiasRebuildError, "synchronizing native BLOCK state") from exc
