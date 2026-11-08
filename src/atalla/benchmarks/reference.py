"""Untimed BF16 oracle for the 16-row mixed MEISSA reduction.

Inputs already contain exactly representable BF16 values. Each group of four
FP32 products/sums is added in FP64 and rounded once to FP32, as in the array.
"""

import numpy as np

from base.dtype import bf16_round


def reference_product(activations, weights, indices=None):
    if indices is None:
        products = activations[:, :, None] * weights[None, :, :]
    else:
        group_bases = 4 * (np.arange(len(indices))[:, None] // 2)
        products = activations[:, group_bases + indices] * weights[None, :, :]
    rows, _, cols = products.shape
    partial = products.reshape(rows, 4, 4, cols).sum(axis=2, dtype=np.float64).astype(np.float32)
    return bf16_round(partial.sum(axis=1, dtype=np.float64).astype(np.float32))
