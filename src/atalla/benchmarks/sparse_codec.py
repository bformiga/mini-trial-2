"""2:4 weight encoding along K and BF16 words for the Atalla DRAM image."""

import numpy as np

from base.dtype import bf16_round


def prune_2of4(weights):
    """Return two values and their ascending positions for each group/column."""
    k, n = weights.shape
    if k % 4 or not np.isfinite(weights).all():
        raise ValueError("2:4 weights must be finite with K divisible by four")
    groups = weights.reshape(k // 4, 4, n)
    # Stable ranking breaks magnitude ties in favor of the lower K position.
    positions = np.argsort(-np.abs(groups), axis=1, kind="stable")[:, :2, :]
    positions.sort(axis=1)
    values = np.take_along_axis(groups, positions, axis=1)
    return values.reshape(k // 2, n), positions.reshape(k // 2, n).astype(np.uint8)


def decompress_2of4(values, indices):
    if (values.ndim != 2 or values.shape[0] % 2 or indices.shape != values.shape
            or not np.isfinite(values).all()):
        raise ValueError("compressed values and indices must have matching [K/2,N] shapes")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices > 3):
        raise ValueError("indices must be integers in 0..3")
    if np.any(indices[0::2] >= indices[1::2]):
        raise ValueError("the two positions per group must be distinct and ascending")
    rows, n = values.shape
    weights = np.zeros((2 * rows, n), dtype=np.float32)
    addresses = 4 * (np.arange(rows)[:, None] // 2) + indices
    weights[addresses, np.arange(n)[None, :]] = values
    return weights


def encode_bf16(values):
    return (bf16_round(values).view(np.uint32) >> 16).astype("<u2").tobytes()


def decode_bf16(payload, shape):
    return (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
