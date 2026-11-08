"""2:4 MEISSA on the scheduler-driven BF16 path."""

import numpy as np

from systolic_array.systolic_array_meissa import SystolicArrayMEISSA


class StructuredMEISSA(SystolicArrayMEISSA):
    """Two physical rows per four logical K positions, with per-column indices."""

    def __init__(self, size, *, mux_latency=0, use_mixed_adder=False):
        if size < 8 or size % 8 or mux_latency < 0:
            raise ValueError("expected a row count divisible by eight and a nonnegative mux latency")
        super().__init__(size, dtype="bf16", mul_latency=1 + mux_latency,
                         use_mixed_adder=use_mixed_adder, use_native=False)
        self.weight_width = size + size // 8
        self._act = np.zeros((2 * size, size), dtype=np.float32)
        self.indices = np.tile(np.arange(size, dtype=np.uint8)[:, None] % 2, (1, size))
        self._group_base = 4 * (np.arange(size)[:, None] // 2)
        self._pending_indices = None

    def enqueue(self, activations, dtype=None, *, count_algo=True):
        if len(activations) != 2 * self.size:
            raise ValueError("sparse activation width must be twice the physical row count")
        if not self.can_accept():
            return False
        self._pending_act = self._cast(activations, dtype)
        self._drained = False
        self._pending_count_algo = bool(count_algo)
        if count_algo:
            self.algo_counts["act_rows"] += 1
        return True

    def enqueue_weights(self, weights, dtype=None, *, count_algo=True):
        if len(weights) != self.weight_width:
            raise ValueError("sparse weights must include packed position indices")
        words = np.asarray(weights[self.size:], dtype=np.float32).view(np.uint32) >> 16
        indices = ((words[:, None] >> np.arange(0, 16, 2, dtype=np.uint32)) & 3).ravel()
        if np.any(indices[::2] == indices[1::2]):
            raise ValueError("the two indices in a 2:4 group must be distinct")
        accepted = super().enqueue_weights(weights[:self.size], dtype, count_algo=count_algo)
        if accepted:
            self._pending_indices = indices
        return accepted

    def _shift_weights(self):
        if self.weight_en and self._pending_wgt is not None:
            self.indices[:, 1:] = self.indices[:, :-1]
            self.indices[:, 0] = self._pending_indices
        super()._shift_weights()
        self._pending_indices = None

    def _advance_trees(self):
        logical = self._act
        self._act = logical[self._group_base + self.indices, self._col_idx[None, :]]
        try:
            super()._advance_trees()
        finally:
            self._act = logical
