"""Compatibility helpers for config dataclass aliases."""

from __future__ import annotations

import warnings
from dataclasses import MISSING, fields
from typing import Any


def warn_deprecated_alias(alias: str, canonical: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{alias} is deprecated; use {canonical} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def init_dataclass_with_aliases(
    instance: object,
    cls: type,
    values: dict[str, Any],
    aliases: dict[str, str],
) -> None:
    values = dict(values)
    for alias, canonical in aliases.items():
        if alias not in values:
            continue
        if canonical in values:
            raise TypeError(f"Cannot pass both {alias!r} and {canonical!r}.")
        warn_deprecated_alias(alias, canonical, stacklevel=4)
        values[canonical] = values.pop(alias)

    for data_field in fields(cls):
        if not data_field.init:
            continue
        name = data_field.name
        if name in values:
            value = values.pop(name)
        elif data_field.default is not MISSING:
            value = data_field.default
        elif data_field.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = data_field.default_factory()  # type: ignore[misc]
        else:
            raise TypeError(f"Missing required argument: {name!r}.")
        object.__setattr__(instance, name, value)

    if values:
        unexpected = next(iter(values))
        raise TypeError(f"Unexpected argument: {unexpected!r}.")


def deprecated_getattr(instance: object, name: str, aliases: dict[str, str]) -> Any:
    if name in aliases:
        canonical = aliases[name]
        warn_deprecated_alias(name, canonical, stacklevel=3)
        return getattr(instance, canonical)
    raise AttributeError(f"{type(instance).__name__!r} object has no attribute {name!r}")


def deprecated_setattr(instance: object, name: str, value: Any, aliases: dict[str, str]) -> None:
    if name in aliases:
        canonical = aliases[name]
        warn_deprecated_alias(name, canonical, stacklevel=3)
        object.__setattr__(instance, canonical, value)
        return
    object.__setattr__(instance, name, value)
