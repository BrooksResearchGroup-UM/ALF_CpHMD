"""cphmd.simulation -- segment loop, checkpoint, archive, shrinker I/O."""

from .archiver import Archiver
from .checkpoint import CheckpointManager, CheckpointMismatchError
from .context import LoopHooks, LoopState, RunContext
from .shrinker import LambdaPrecision, ShrinkerMetadata, write_segment_parquet

__all__ = [
    "Archiver",
    "CheckpointManager",
    "CheckpointMismatchError",
    "LambdaPrecision",
    "LoopHooks",
    "LoopState",
    "RunContext",
    "ShrinkerMetadata",
    "write_segment_parquet",
]
