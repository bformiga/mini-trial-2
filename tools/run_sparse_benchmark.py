"""Run a generated workload on full Atalla's dense and 2:4 MEISSA paths."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atalla.benchmarks.scheduled_gemm import MAX_ROWS, SIZE, run_block
from atalla.benchmarks.sparse_codec import decompress_2of4
from base.dtype import bf16_round


def run(a, w, indices, args, mode):
    cycles = weight_wait = idle_wait = payload = metadata = passes = blocks = 0
    dram_bytes = dram_bursts = later_wait = 0
    busy_cycles = Counter()
    concurrency = Counter()
    breakdown = Counter()
    wait_stages = Counter()
    output = np.empty((a.shape[0], w.shape[1]), dtype=np.float32)
    reference = np.empty_like(output)
    error = 0.0
    correct = True
    for i in range(0, a.shape[0], args.block_rows):
        ab = a[i:i + args.block_rows]
        for j in range(0, w.shape[1], SIZE):
            wb = w[:, j:j + SIZE]
            columns = wb.shape[1]
            col_padding = ((0, 0), (0, SIZE - columns))
            wb = np.pad(wb, col_padding)
            positions = None
            if indices is not None:
                positions = np.pad(indices[:, j:j + SIZE], col_padding)
                positions[1::2, columns:] = 1
            first = blocks == 0
            out, expected, detail, activity = run_block(
                ab, wb, indices=positions, mux_latency=args.mux_latency,
                dram_latency=args.dram_latency, burst_bytes=args.burst_bytes,
                timeline=bool(args.trace_dir and first), activity=bool(args.trace_dir and first),
                cycle_limit=args.cycle_limit)
            output[i:i + len(ab), j:j + columns] = out[:, :columns]
            reference[i:i + len(ab), j:j + columns] = expected[:, :columns]
            cycles += detail["cycles"]
            dram_bytes += detail["memory"]["dram_bytes"]
            dram_bursts += detail["memory"]["dram_bursts"]
            weight_wait += detail["weight_wait_cycles"]
            idle_wait += detail["weight_wait_array_idle_cycles"]
            later_wait += detail["weight_wait_after_first_activation_cycles"]
            payload += detail["weight_payload_bytes"]
            metadata += detail["metadata_bytes"]
            passes += detail["array_passes"]
            blocks += 1
            error = max(error, detail["max_absolute_error"])
            correct &= detail["correctness_passed"]
            busy_cycles.update({key: unit["cycles"] for key, unit in detail["unit_busy"].items()})
            concurrency.update(detail["vlsu_active_count_cycles"])
            breakdown.update(detail["cycle_breakdown"])
            wait_stages.update(detail["weight_wait_by_stage"])
            if first and args.trace_dir:
                from activity_view import render

                args.trace_dir.mkdir(parents=True, exist_ok=True)
                save_json(args.trace_dir / f"{mode}_block0.json", detail)
                (args.trace_dir / f"{mode}_block0.html").write_text(
                    render(activity.trace({"workload": mode, "cycles": detail["cycles"]})),
                    encoding="utf-8")
    return output, reference, dict(
        cycles=cycles, blocks=blocks, array_passes=passes,
        dram_bus_bytes=dram_bytes, dram_bursts=dram_bursts,
        correctness_passed=correct, max_absolute_error=error,
        weight_wait_cycles=weight_wait, weight_wait_array_idle_cycles=idle_wait,
        weight_wait_after_first_activation_cycles=later_wait,
        weight_wait_by_stage=dict(wait_stages), weight_payload_bytes=payload,
        metadata_bytes=metadata, vlsu_active_count_cycles=dict(sorted(concurrency.items())),
        unit_busy={key: dict(cycles=n, fraction=n / cycles)
                   for key, n in busy_cycles.most_common()},
        cycle_breakdown=dict(breakdown.most_common()),
    )


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_workload(args):
    with np.load(args.input, allow_pickle=False) as package:
        a, pruned = package["A"], package["W_pruned"]
        metadata = json.loads(str(package["metadata_json"].item()))
        if a.ndim != 2 or pruned.ndim != 2 or a.shape[1] != pruned.shape[0]:
            raise ValueError("expected A[M,K] and W_pruned[K,N]")
        m, k = a.shape
        n = pruned.shape[1]
        if metadata.get("dtype") != "bf16" or metadata.get("shape_MNK") != [m, n, k]:
            raise ValueError("expected BF16 metadata matching the input shape")
        rows, cols, depth = (m, n, k) if args.full else (
            min(8, m) if args.rows is None else args.rows,
            min(16, n) if args.cols is None else args.cols,
            min(64, k) if args.k is None else args.k)
        if not (0 < rows <= m and 0 < cols <= n and 0 < depth <= k):
            raise ValueError("requested dimensions are outside the input workload")
        a = a[:rows, :depth]
        weights = {}
        indices = None
        if args.mode in ("compare", "dense-pruned", "sparse"):
            weights["dense-pruned"] = pruned[:depth, :cols]
        if args.mode in ("compare", "sparse"):
            if depth % 4 or k % 4 or metadata.get("sparsity_axis") != "K":
                raise ValueError("sparse mode requires groups of four along K")
            values, indices = package["W_values"], package["W_indices"]
            if values.shape != (k // 2, n) or indices.shape != values.shape:
                raise ValueError("expected W_values and W_indices with shape [K/2,N]")
            values, indices = values[:depth // 2, :cols], indices[:depth // 2, :cols]
            if not np.array_equal(decompress_2of4(values, indices), weights["dense-pruned"]):
                raise ValueError("compressed sparse weights disagree with W_pruned")
            weights["sparse"] = values
        if args.mode == "dense-original":
            dense = package["W_dense"]
            if dense.shape != (k, n):
                raise ValueError("expected W_dense[K,N] matching W_pruned")
            weights["dense-original"] = dense[:depth, :cols]
    for operand in (a, *weights.values()):
        if operand.dtype != np.float32 or not np.isfinite(operand).all() or not np.array_equal(
                operand, bf16_round(operand)):
            raise ValueError("operands must be finite BF16 values stored as float32")
    return a, weights, indices, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("compare", "dense-pruned", "sparse", "dense-original"),
                        default="compare")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--full", action="store_true", help="use the complete workload dimensions")
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--dram-latency", type=int, default=6)
    parser.add_argument("--burst-bytes", type=int, default=8)
    parser.add_argument("--mux-latency", type=int, default=0)
    parser.add_argument("--cycle-limit", type=int, default=5_000_000, help="per output block")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="HTML and JSON for the first block per mode")
    parser.add_argument("--output", type=Path, help="save simulated outputs and BF16 references by mode")
    args = parser.parse_args()
    if not 1 <= args.block_rows <= MAX_ROWS or args.mux_latency < 0 or args.dram_latency < 1:
        parser.error("expected block rows 1–32, nonnegative mux latency and positive DRAM latency")
    if args.burst_bytes < 2 or args.burst_bytes % 2 or args.cycle_limit < 1:
        parser.error("burst bytes must be positive and even, and cycle limit must be positive")
    if args.full and any(x is not None for x in (args.rows, args.cols, args.k)):
        parser.error("use --full or explicit dimensions")
    modes = ("dense-pruned", "sparse") if args.mode == "compare" else (args.mode,)
    try:
        a, weights, indices, metadata = load_workload(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"{args.input}: {exc}")
    rows, depth = a.shape
    cols = weights["sparse" if "sparse" in modes else modes[0]].shape[1]
    report = dict(input=str(args.input), metadata=metadata, shape_MNK=[rows, cols, depth],
                  physical_array=[SIZE, SIZE], reduction="mixed", block_rows=args.block_rows,
                  dram_latency=args.dram_latency, burst_bytes=args.burst_bytes,
                  sparse_mux_latency=args.mux_latency, sparse_rtl_timing_validated=False,
                  timing_scope="sum_of_independent_full_atalla_output_block_programs", modes={})
    outputs = {}
    for mode in modes:
        outputs[mode], outputs[f"{mode}_reference"], report["modes"][mode] = run(
            a, weights[mode], indices if mode == "sparse" else None, args, mode)
    if len(modes) == 2:
        dense, sparse = (report["modes"][mode] for mode in modes)
        report["dense_over_sparse_cycle_ratio"] = dense["cycles"] / sparse["cycles"]
        report["dense_sparse_max_absolute_difference"] = float(
            np.max(np.abs(outputs["dense-pruned"] - outputs["sparse"])))
    if args.json:
        save_json(args.json, report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **outputs)
    print(json.dumps(report, indent=2))
    return 0 if all(r["correctness_passed"] for r in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
