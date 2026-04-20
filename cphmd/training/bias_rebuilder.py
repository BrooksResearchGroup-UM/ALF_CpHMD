from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from cphmd.training.bias_snapshot import BiasSnapshot


@dataclass(frozen=True)
class BiasRebuilder:
    native_block: Any | None = None

    def apply(self, snapshot: BiasSnapshot) -> None:
        native = self._native()
        sync_state = getattr(native, "sync_state", None)
        if sync_state is not None:
            sync_state()
        modify_biases = getattr(native, "modify_biases", None)
        with (modify_biases() if modify_biases is not None else nullcontext()):
            native.clear_biases()
            for block_id, bias in snapshot.intrinsic_biases:
                native.set_intrinsic_bias(block_id, bias)
            ldbv_terms = tuple(snapshot.ldbv_terms)
            set_bias_count = getattr(native, "set_bias_count", None)
            if set_bias_count is not None:
                set_bias_count(len(ldbv_terms))
            for index, term in enumerate(ldbv_terms, start=1):
                native.add_bias(*term.as_tuple(), index=index)
        if sync_state is not None:
            sync_state()

    def _native(self):
        if self.native_block is not None:
            return self.native_block
        from cphmd.native import block

        return block
