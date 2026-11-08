# Atalla-Sim MEISSA 2:4 benchmarks

These synthetic GEMMs retain the padded shapes from the original benchmark generator. The runner assembles Atalla instructions and DRAM data, then executes them on the full Atalla-Sim scheduler, memory system, vector core, and MEISSA model. See the [upstream simulator README](README) for the other simulator flows.

## Reproduce a comparison

From the repository root, using Python 3.11 or newer:

```sh
python -m pip install numpy==2.3.5
python generate_benchmarks.py --workload resnet50_conv1
python tools/run_sparse_benchmark.py --input benchmarks/resnet50_conv1.npz --k 160 --json results/resnet.json --output results/resnet_outputs.npz --trace-dir results/resnet_trace
```

This uses eight activation rows, 16 output columns, and the full ResNet K dimension of 160. Comparison mode runs the same pruned weights through dense MEISSA and compressed 2:4 MEISSA. Each mode must match its own BF16 reference. Add `--mode dense-original` in a separate run for the unpruned baseline. Use `--rows`, `--cols`, and `--k` to select a leading submatrix, or `--full` to run all three dimensions. Full runs execute many independent output-block programs in series and can take substantially longer.

Omit `--workload` to generate all three packages. The default seed is 100. Each workload has its own RNG stream, so generating it alone produces the same arrays as generating it alongside the others. The package contains BF16 operands held in float32 and FP32 NumPy references for the original dense weights (`C_dense_fp32`) and the pruned weights (`C_pruned_fp32`). These full-size NumPy products provide mathematical references. The optional `--output` file contains simulated DRAM outputs and matching, schedule-specific BF16 reference arrays, such as `dense-pruned` and `dense-pruned_reference`. BF16 rounding at each K pass can make the two simulated modes differ even with identical pruned weights.

| Workload | Full M × N × K |
| --- | --- |
| `resnet50_conv1` | 12,544 × 64 × 160 |
| `llama3_q_proj` | 32 × 1,024 × 4,096 |
| `llama3_gate_proj` | 256 × 1,024 × 4,096 |

## Validated runs

These runs used seed 100, 6-cycle DRAM latency, 8-byte bursts, and zero extra mux cycles. Both modes matched their own BF16 reference bit for bit in each run.

| Workload | Simulated M × N × K | Dense-pruned cycles | 2:4 cycles |
| --- | --- | ---: | ---: |
| `resnet50_conv1` | 8 × 16 × 160 | 5,293 | 2,967 |
| `resnet50_conv1` | 8 × 64 × 160 | 21,172 | 11,868 |
| `resnet50_conv1`, partial tiles | 35 × 19 × 36 | 10,586 | 8,232 |
| `llama3_q_proj` | 8 × 16 × 4,096 | 129,823 | 70,403 |
| `llama3_gate_proj` | 8 × 16 × 4,096 | 129,823 | 70,403 |

The ResNet unpruned baseline also matched its reference at 8 × 16 × 160. The full workload inputs and FP32 references were generated for all three workloads. The table reports simulated submatrices.

## Sparse layout

- `A[M,K]` and `W_dense[K,N]` use BF16 values. For each output column, the generator keeps the two largest-magnitude weights per contiguous group of four along K. A tie selects the lower K position. `W_values[K/2,N]` and `W_indices[K/2,N]` store the retained values and their positions in ascending order within each group. There are always two stored slots, which can include zero. The runner checks that the requested compressed weights reconstruct `W_pruned[K,N]`.
- Both modes use a physical 16×16 MEISSA array with the mixed adder tree and 32-lane vector registers. Each dense pass covers 16 K positions. Each sparse pass covers 32, using each pair of indices to select two activations from its four-position group, independently in every output column.
- A sparse weight column occupies 36 bytes in DRAM and the scratchpad. Sixteen little-endian BF16 values come first, followed by a little-endian 32-bit metadata word. Index `i` occupies bits `[2i+1:2i]`. The existing VLSU and vector register carry the two metadata halfwords to the GSAU. The runner pads incomplete output-column and K passes with zero values and valid metadata.

## Read the performance report

- `cycles` includes instruction fetch, scheduler, DRAM, scratchpad, MEISSA, BF16 vector accumulation, and output writes to DRAM. The runner sums independent output-block programs, each spanning the requested K. Weight DMA can overlap compute within an output block using two scratchpad regions. Cross-block reuse and overlap are outside this schedule.
- `weight_payload_bytes` includes padded weights and packed metadata. `dram_bus_bytes` includes traffic for instructions, weights, activations and output, as well as burst granularity.
- `weight_wait_cycles` counts in-order dependency stalls on weight transfer instructions. The report separates DMA-to-scratchpad waits and VLSU-to-register waits. It also reports unit busy cycles and the number of simultaneously active VLSUs. `--trace-dir` writes an HTML activity view and per-cycle JSON for the first block of each mode.
- The 16×16 BF16 configuration, metadata packing, and default zero extra mux cycles are modeled conventions. Use `--mux-latency` to vary the modeled delay. The benchmark validates this Atalla-Sim path against its numerical references. Matching the team's sparse RTL packing, lane order, and cycle timing requires RTL co-validation.

Run the scheduler-path checks with:

```sh
python -m pip install pytest
python -m pytest -o pythonpath=src -q tests/atalla/test_scheduled_sparse_gemm.py tests/atalla/test_sysarr_meissa_end_to_end.py tests/scheduler/test_perf_monitor.py
```
