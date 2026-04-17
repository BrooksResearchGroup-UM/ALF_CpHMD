"""Utilities for fingerprinting the public surface of Python modules."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from dataclasses import MISSING, Field
from types import ModuleType
from typing import Iterable

__all__ = [
    "compute",
    "diff",
    "fingerprint_modules",
    "diff_modules",
]

_MISSING = object()


@dataclasses.dataclass(frozen=True)
class _SurfaceEntry:
    module: str
    kind: str
    name: str
    value: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.module, self.kind, self.name)


def _annotation_repr(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return "inspect._empty"
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__", repr(annotation))
        if module == "builtins":
            return qualname
        return f"{module}.{qualname}"
    return repr(annotation)


def _callable_repr(obj: object) -> str:
    if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        module = getattr(obj, "__module__", "")
        qualname = getattr(obj, "__qualname__", repr(obj))
        if module == "builtins":
            return qualname
        return f"{module}.{qualname}"
    return repr(obj)


def _signature_repr(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<no-signature>"


def _default_repr(field: Field) -> str:
    if field.default is not MISSING:
        return repr(field.default)
    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return f"factory:{_callable_repr(field.default_factory)}"
    return "<missing>"


def _iter_exported_names(module: ModuleType) -> list[str]:
    exported = module.__dict__.get("__all__")
    if exported is not None:
        names = [name for name in exported if isinstance(name, str) and not name.startswith("_")]
        return list(dict.fromkeys(names))
    return sorted(name for name in module.__dict__ if not name.startswith("_"))


def _is_module_reexport(module_name: str, value: object, exported: bool) -> bool:
    if not isinstance(value, ModuleType):
        return False
    if exported:
        return True
    return value.__name__.startswith(f"{module_name}.")


def _module_surface(module: ModuleType) -> list[_SurfaceEntry]:
    surface: list[_SurfaceEntry] = []
    exported_names = _iter_exported_names(module)
    explicit_exports = "__all__" in module.__dict__

    for name in exported_names:
        if name.startswith("_"):
            continue

        value = module.__dict__.get(name, _MISSING)
        if value is _MISSING:
            continue
        qualname = f"{module.__name__}.{name}"

        if _is_module_reexport(module.__name__, value, explicit_exports):
            surface.append(
                _SurfaceEntry(
                    module.__name__,
                    "module",
                    qualname,
                    getattr(value, "__name__", repr(value)),
                )
            )
            continue

        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
            if explicit_exports or getattr(value, "__module__", None) == module.__name__:
                surface.append(
                    _SurfaceEntry(
                        module.__name__,
                        "function",
                        qualname,
                        f"signature={_signature_repr(value)}",
                    )
                )
            continue

        if inspect.isclass(value):
            if explicit_exports or getattr(value, "__module__", None) == module.__name__:
                class_signature = f"signature={_signature_repr(value)}"
                bases = (
                    ",".join(base.__qualname__ for base in value.__bases__ if base is not object)
                    or "object"
                )
                payload = f"{class_signature};bases={bases}"
                surface.append(_SurfaceEntry(module.__name__, "class", qualname, payload))

                if dataclasses.is_dataclass(value):
                    for field in dataclasses.fields(value):
                        field_name = f"{qualname}.{field.name}"
                        payload = (
                            f"type={_annotation_repr(field.type)};"
                            f"default={_default_repr(field)}"
                        )
                        surface.append(
                            _SurfaceEntry(
                                module.__name__,
                                "dataclass-field",
                                field_name,
                                payload,
                            )
                        )
            continue

        surface.append(_SurfaceEntry(module.__name__, "constant", qualname, repr(value)))

    surface.sort(key=lambda entry: entry.key + (entry.value,))
    return surface


def _surface_for_modules(modules: Iterable[ModuleType]) -> list[_SurfaceEntry]:
    surface: list[_SurfaceEntry] = []
    for module in sorted(modules, key=lambda item: item.__name__):
        surface.extend(_module_surface(module))
    surface.sort(key=lambda entry: entry.key + (entry.value,))
    return surface


def compute(modules: Iterable[ModuleType]) -> str:
    """Compute a SHA256 digest for the public surface of one or more modules."""
    surface = _surface_for_modules(modules)
    payload = "\n".join(
        f"{entry.module}|{entry.kind}|{entry.name}|{entry.value}" for entry in surface
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def diff(old: Iterable[ModuleType], new: Iterable[ModuleType]) -> dict[str, list[dict[str, str]]]:
    """Return added, removed, and changed public surface entries."""
    old_entries = {entry.key: entry for entry in _surface_for_modules(old)}
    new_entries = {entry.key: entry for entry in _surface_for_modules(new)}

    added = [
        {
            "module": entry.module,
            "kind": entry.kind,
            "name": entry.name,
            "value": entry.value,
        }
        for key, entry in sorted(new_entries.items())
        if key not in old_entries
    ]
    removed = [
        {
            "module": entry.module,
            "kind": entry.kind,
            "name": entry.name,
            "value": entry.value,
        }
        for key, entry in sorted(old_entries.items())
        if key not in new_entries
    ]
    changed = [
        {
            "module": new_entries[key].module,
            "kind": new_entries[key].kind,
            "name": new_entries[key].name,
            "old": old_entries[key].value,
            "new": new_entries[key].value,
        }
        for key in sorted(old_entries.keys() & new_entries.keys())
        if old_entries[key].value != new_entries[key].value
    ]

    return {"added": added, "removed": removed, "changed": changed}


fingerprint_modules = compute
diff_modules = diff
