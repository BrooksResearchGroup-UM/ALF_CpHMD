"""Initial bias estimation from single-point energy perturbation.

Estimates ALF linear biases (b) and quadratic barriers (c) by evaluating
CHARMM energies at pure-state and midpoint lambda configurations on a
minimized structure. Used when presets are unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_b_from_endpoints(
    endpoint_energies: dict[int, np.ndarray],
    nsubs: list[int],
) -> np.ndarray:
    """Compute linear biases (b) from endpoint energies.

    For each site, b[i] = -(E[i] - E_mean) where E_mean is the mean
    energy across all substates of that site. Centering ensures sum(b) = 0.

    Args:
        endpoint_energies: {site_index: array of energies per substate}
        nsubs: Number of substates per site.

    Returns:
        b array with shape (1, nblocks) matching ALF convention.
    """
    nblocks = sum(nsubs)
    b = np.zeros(nblocks)

    offset = 0
    for site_idx, n in enumerate(nsubs):
        energies = endpoint_energies[site_idx]
        e_mean = np.mean(energies)
        b[offset : offset + n] = -(energies - e_mean)
        offset += n

    return b.reshape(1, -1)
