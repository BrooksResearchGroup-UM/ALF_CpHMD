"""Deterministic RNG helpers for phase-invariant workflows."""

from __future__ import annotations

import copy
import hashlib

import numpy as np

DOMAINS = ("rex", "dynamics", "gimp_mc", "velocity_init", "production_chunk")

__all__ = [
    "DOMAINS",
    "derive_seed",
    "make_rng",
    "get_rng_state",
    "make_rng_from_state",
]


def derive_seed(master: int, domain: str, *scope: int) -> int:
    """Derive a deterministic 64-bit seed for a named RNG domain."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain!r}")

    scope_text = ":".join(str(int(value)) for value in scope)
    digest = hashlib.sha256(f"{master}:{domain}:{scope_text}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def make_rng(master: int, domain: str, rank: int) -> np.random.Generator:
    """Build a deterministic NumPy Generator for a domain and MPI rank."""
    return np.random.default_rng(derive_seed(master, domain, rank))


def get_rng_state(rng: np.random.Generator) -> dict:
    """Return a deep copy of a Generator's bit-generator state."""
    return copy.deepcopy(rng.bit_generator.state)


def make_rng_from_state(state: dict) -> np.random.Generator:
    """Restore a Generator from a copied bit-generator state."""
    state_copy = copy.deepcopy(state)
    bit_generator_name = state_copy.get("bit_generator")
    if not isinstance(bit_generator_name, str):
        raise ValueError("state is missing bit_generator metadata")

    try:
        bit_generator_cls = getattr(np.random, bit_generator_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported bit generator: {bit_generator_name!r}") from exc

    bit_generator = bit_generator_cls()
    bit_generator.state = state_copy
    return np.random.Generator(bit_generator)
