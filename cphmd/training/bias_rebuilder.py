from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cphmd.training.bias_snapshot import BiasSnapshot


@dataclass(frozen=True)
class BiasRebuilder:
    native_block: Any | None = None

    def apply(self, snapshot: BiasSnapshot) -> None:
        native = self._native()
        native.clear_biases()
        for block_id, bias in snapshot.intrinsic_biases:
            native.set_intrinsic_bias(block_id, bias)
        for term in snapshot.ldbv_terms:
            native.add_bias(*term.as_tuple())

    def _native(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block
