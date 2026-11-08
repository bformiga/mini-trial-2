"""Regression checks on the full scheduler and memory path."""

import numpy as np
import pytest

from atalla.benchmarks.scheduled_gemm import SIZE, run_block
from atalla.benchmarks.sparse_codec import decompress_2of4, prune_2of4
from base.dtype import bf16_round


def test_prunes_k_per_column_with_stable_ties():
    weights = np.array([[2, 0], [-2, 6], [2, 0], [0, -7]], dtype=np.float32)
    values, indices = prune_2of4(weights)
    np.testing.assert_array_equal(indices, [[0, 1], [1, 3]])
    np.testing.assert_array_equal(decompress_2of4(values, indices),
                                  [[2, 0], [-2, 6], [0, 0], [0, -7]])
    with pytest.raises(ValueError, match="distinct and ascending"):
        decompress_2of4(values, np.zeros_like(indices))


@pytest.mark.parametrize("sparse", [False, True])
def test_rectangular_gemm_runs_all_k_passes_and_writes_dram(sparse):
    rng = np.random.default_rng(5)
    a = bf16_round(rng.normal(size=(3, 36)))
    dense = bf16_round(rng.normal(size=(36, SIZE)))
    values, indices = prune_2of4(dense)
    w = values if sparse else decompress_2of4(values, indices)
    out, expected, report, _ = run_block(a, w, indices=indices if sparse else None, timeline=True)
    np.testing.assert_array_equal(out, expected)
    assert report["correctness_passed"]
    assert report["array_passes"] == (2 if sparse else 3)
    assert report["metadata_bytes"] == (128 if sparse else 0)
    assert sum(report["vlsu_active_count_cycles"].values()) == report["cycles"]
    assert report["unit_busy"]["vlsu1"]["cycles"] > 0
    assert report["instruction_mix"]["vreg_store"] == len(a)


def test_weight_wait_distinguishes_slow_dma_from_array_compute():
    a = np.ones((1, 32), dtype=np.float32)
    values = np.tile(np.array([[1], [-2]], dtype=np.float32), (8, SIZE))
    indices = np.tile(np.array([[1], [3]], dtype=np.uint8), (8, SIZE))
    _, _, fast, _ = run_block(a, values, indices=indices)
    out, _, slow, _ = run_block(a, values, indices=indices, dram_latency=80, timeline=True)
    assert np.all(out == -8)
    assert slow["weight_wait_by_stage"]["dma_to_scratchpad"] > fast["weight_wait_cycles"]
    assert 0 < slow["weight_wait_array_idle_cycles"] <= slow["weight_wait_cycles"]
    assert len(slow["timeline"]["weight_wait_cycles"]) == slow["weight_wait_cycles"]
