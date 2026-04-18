"""Current-tree invariant baselines for import and RNG audits."""

from __future__ import annotations

PYCHARMM_IMPORT_ALLOWLIST = (
    "cphmd/analysis/volume.py",
    "cphmd/core/alf_runner.py",
    "cphmd/core/charmm_utils.py",
    "cphmd/core/dynamics_runner.py",
    "cphmd/core/production_runner.py",
    "cphmd/core/replica_exchange.py",
)

BARE_RNG_ALLOWLIST = (
    "cphmd/analysis/dca.py",
    "cphmd/analysis/pka_fitting.py",
    "cphmd/core/block_builder.py",
    "cphmd/core/dynamics_runner.py",
    "cphmd/core/entropy.py",
    "cphmd/core/replica_exchange.py",
    "cphmd/core/variance.py",
)
