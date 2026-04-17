"""Native bootstrap error types."""

from __future__ import annotations

from typing import TypeVar

_ErrorT = TypeVar("_ErrorT", bound=BaseException)


class CpHMDNativeError(RuntimeError):
    """Base class for native CpHMD bootstrap failures."""


class SystemLoadError(CpHMDNativeError):
    """Raised when the native simulation system cannot be loaded."""


class BlockStateError(CpHMDNativeError):
    """Raised when BLOCK state cannot be established or restored."""


class DynamicsRunError(CpHMDNativeError):
    """Raised when a dynamics segment fails."""


class ExchangeTransportError(CpHMDNativeError):
    """Raised when replica exchange transport fails."""


class BiasRebuildError(CpHMDNativeError):
    """Raised when a bias table or bias state cannot be rebuilt."""


# Deprecated compatibility alias; prefer ``BiasRebuildError`` for new native code.
BiasRebuildFailure = BiasRebuildError


class DirectStateError(CpHMDNativeError):
    """Raised when direct state transitions fail."""


class MPIInitError(CpHMDNativeError):
    """Raised when MPI cannot be initialized or mapped safely."""


def wrap_exception(original: BaseException, target: type[_ErrorT], context: str) -> _ErrorT:
    """Wrap ``original`` in ``target`` while preserving ``__cause__``."""
    wrapped = target(f"{context}: {original}")
    wrapped.__cause__ = original
    return wrapped
