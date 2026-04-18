"""Signal helpers for graceful segment-boundary shutdown."""

from __future__ import annotations

import signal
from dataclasses import dataclass


@dataclass
class StopRequested:
    stop_requested: bool = False


def install_clean_shutdown_handler() -> StopRequested:
    """Install SIGTERM/SIGINT handlers that only set a stop flag.

    pyCHARMM/native dynamics is not interrupted by this helper. The simulation
    loop observes the token between segments and writes a clean checkpoint then.
    """

    token = StopRequested()

    def _handler(signum, frame):
        token.stop_requested = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return token
