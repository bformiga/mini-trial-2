"""Reproducible synthetic GEMMs adapted from the student benchmark generator."""

import argparse
import json
from pathlib import Path

import numpy as np

from base.dtype import bf16_round
from .sparse_codec import decompress_2of4, prune_2of4

WORKLOADS = {
    "resnet50_conv1": (12544, 64, 160),
    "llama3_q_proj": (32, 1024, 4096),
    "llama3_gate_proj": (256, 1024, 4096),
}


def generate(name, output_dir, seed):
    m, n, k = WORKLOADS[name]
    ordinal = list(WORKLOADS).index(name)
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    a = bf16_round(rng.standard_normal((m, k), dtype=np.float32))
    w = bf16_round(rng.standard_normal((k, n), dtype=np.float32))
    values, indices = prune_2of4(w)
    pruned = decompress_2of4(values, indices)
    metadata = dict(workload=name, shape_MNK=[m, n, k], seed=seed,
                    workload_seed_index=ordinal, numpy_version=np.__version__,
                    dtype="bf16", sparsity_axis="K", synthetic=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    arrays = dict(A=a, W_dense=w, W_pruned=pruned, W_values=values, W_indices=indices,
                  metadata_json=np.asarray(json.dumps(metadata)),
                  C_dense_fp32=a @ w, C_pruned_fp32=a @ pruned)
    np.savez(path, **arrays)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    for name in args.workload or WORKLOADS:
        print(generate(name, args.output_dir, args.seed))
