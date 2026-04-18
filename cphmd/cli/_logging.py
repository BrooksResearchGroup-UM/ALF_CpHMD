"""Logging helpers for native MPI runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import TracebackType


class _FlushOnWarningFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if record.levelno >= logging.WARNING:
            self.flush()


def install_rank_logger(*, rank: int, run_dir: Path, level: str = "INFO") -> logging.Logger:
    """Install append-only per-rank logging.

    File handlers are intentionally buffered; callers flush at segment
    boundaries to avoid per-record metadata pressure on shared filesystems.
    """

    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cphmd.rank{rank:02d}")
    logger.setLevel(_coerce_level(level))
    logger.propagate = False
    _clear_file_handlers(logger)

    rank_handler = _FlushOnWarningFileHandler(
        log_dir / f"rank{rank:02d}.log",
        mode="a",
        delay=True,
    )
    rank_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [rank %(rank)s] %(message)s")
    )
    rank_handler.addFilter(_RankFilter(rank))
    logger.addHandler(rank_handler)

    if rank == 0:
        coordinator = _FlushOnWarningFileHandler(
            log_dir / "coordinator.log",
            mode="a",
            delay=True,
        )
        coordinator.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        coordinator.addFilter(_RankFilter(rank))
        logger.addHandler(coordinator)

    return logger


def install_excepthook(
    *,
    logger: logging.Logger | None = None,
    rank: int = 0,
    run_dir: Path | None = None,
):
    if logger is None:
        if run_dir is None:
            raise ValueError("run_dir is required when logger is not provided")
        logger = install_rank_logger(rank=rank, run_dir=run_dir)

    previous = sys.excepthook

    def _hook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        flush_rank_loggers(logger)
        previous(exc_type, exc, tb)

    sys.excepthook = _hook
    return previous


def flush_rank_loggers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        flush = getattr(handler, "flush", None)
        if flush is not None:
            flush()


def redirect_charmm_output(*args, **kwargs) -> None:
    """Placeholder boundary for CHARMM unit-6 redirection.

    The actual pyCHARMM file handle setup must happen after MPI and pyCHARMM
    initialization. The CLI calls this hook only after entering native runtime.
    """

    from cphmd.native import system

    output_unit = kwargs.get("unit", 6)
    system.set_output_unit(output_unit)


class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def _coerce_level(level: str) -> int:
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"unknown log level {level!r}")
    return value


def _clear_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        close = getattr(handler, "close", None)
        if close is not None:
            close()
