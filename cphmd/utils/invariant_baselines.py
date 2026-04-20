"""Current-tree invariant baselines for import and RNG audits.

The remaining RNG exception is the seeded Numba Monte Carlo implementation in
``cphmd/core/entropy.py``. Replacing that generator changes G_imp numerical
baselines and belongs in a dedicated regression phase, not the Phase 8
promotion cleanup.
"""

from __future__ import annotations

PYCHARMM_IMPORT_ALLOWLIST: tuple[str, ...] = ()

BARE_RNG_ALLOWLIST: tuple[str, ...] = ("cphmd/core/entropy.py",)
