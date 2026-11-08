# Atalla-Sim MEISSA 2:4 benchmarks

These synthetic GEMMs retain the padded shapes from the original benchmark generator. The runner assembles Atalla instructions and DRAM data, then executes them on the full Atalla-Sim scheduler, memory system, vector core, and MEISSA model. See the [upstream simulator README](README) for the other simulator flows.

## Open RTL timing blocker

Confirm how many logical K positions the team's MEISSA RTL covers per sparse GEMM issue. The benchmark keeps a physical 16×16 grid and models 16 logical K positions per dense pass and 32 per sparse pass. Loading two weights and their indices per group of four establishes the compression format. The activation delivery and issue timing determine whether the same grid can cover twice as many logical K positions in one pass.

Until the K-per-issue mapping is confirmed, report these cycle counts and speedups as results of the modeled schedule. If sparse RTL keeps the same logical K per issue, update the pass schedule before using these numbers to estimate RTL performance.

## Reproduce a comparison

From the repository root, using Python 3.11 or newer:

```sh
python -m pip install numpy==2.3.5
python generate_benchmarks.py --workload resnet50_conv1 --seed 100
python tools/run_sparse_benchmark.py --input benchmarks/resnet50_conv1.npz --k 160 --json results/resnet.json --output results/resnet_outputs.npz --trace-dir results/resnet_trace
```

This uses eight activation rows, 16 output columns, and the full ResNet K dimension of 160. Comparison mode runs the same pruned weights through dense MEISSA and compressed 2:4 MEISSA. Each mode must match its own BF16 reference. Add `--mode dense-original` in a separate run for the unpruned baseline. Use `--rows`, `--cols`, and `--k` to select a leading submatrix, or `--full` to run all three dimensions. Full runs execute many independent output-block programs in series and can take substantially longer.

## Workload dimensions and seed

Run `python generate_benchmarks.py --seed 100` to regenerate all three packages in `benchmarks/`. Add `--workload NAME` for one workload or `--output-dir PATH` to write elsewhere. [generate_benchmarks.py](generate_benchmarks.py) calls the tracked [workload definitions](src/atalla/benchmarks/workloads.py).

The generator writes `A[M,K]`, `W_dense[K,N]`, `W_pruned[K,N]`, `W_values[K/2,N]`, `W_indices[K/2,N]`, and two FP32 reference matrices of shape `[M,N]`. These dimensions describe the complete synthetic GEMMs, using the padded dimensions supplied in the original benchmark definitions. The generator fills those dimensions directly with synthetic operands, including the padded extent. It does not construct an unpadded model tensor and zero-fill its tail.

| Workload | M | N | K | Seed stream index |
| --- | ---: | ---: | ---: | ---: |
| `resnet50_conv1` | 12,544 | 64 | 160 | 0 |
| `llama3_q_proj` | 32 | 1,024 | 4,096 | 1 |
| `llama3_gate_proj` | 256 | 1,024 | 4,096 | 2 |

The default seed is **100**. Each workload uses NumPy `default_rng(SeedSequence([seed, stream_index]))`, so selecting a workload by itself produces the same operands as generating all workloads together. The package records the seed, stream index, NumPy version, `shape_MNK`, operand format, and sparsity axis in `metadata_json`. Use the documented NumPy version for input reproduction and the same NumPy and BLAS environment when checking FP32 references bit for bit.

## Simulator padding and integration

- The runner handles up to 32 requested rows per output block. A short final row block uses its actual row count. It zero-pads a short N block to 16 output columns and discards padded columns from the reported output.
- A dense pass covers 16 K positions. The modeled sparse pass covers 32. Sparse inputs and sparse slices require K divisible by four. The runner zero-pads a short final K pass and supplies distinct indices for padded sparse groups. The three complete workload dimensions align to both K pass sizes and to 16 output columns.
- The runner consumes the generated NPZ directly and builds the instructions and DRAM image for Trial 2's Atalla-Sim platform. It checks that the compressed values and indices reconstruct the pruned weights. With `--output`, it also saves each mode's simulated results and matching schedule-specific BF16 references, such as `dense-pruned` and `dense-pruned_reference`.
- Matching the team's sparse RTL still requires confirming logical K per issue, the order of activation lanes, packed weight and metadata layout, valid/ready and mux timing, and the arithmetic formats and rounding. Confirm these against a sparse RTL tile trace before using the simulator's cycle counts as hardware estimates. The simulator results below cover selected submatrices.

## Numerically validated runs

These runs used seed 100, 6-cycle DRAM latency, 8-byte bursts, and zero extra mux cycles. Both modes matched their own BF16 reference bit for bit in each run.

| Workload | Simulated M × N × K | Dense-pruned cycles | 2:4 cycles |
| --- | --- | ---: | ---: |
| `resnet50_conv1` | 8 × 16 × 160 | 5,293 | 2,967 |
| `resnet50_conv1` | 8 × 64 × 160 | 21,172 | 11,868 |
| `resnet50_conv1`, partial tiles | 35 × 19 × 36 | 10,586 | 8,232 |
| `llama3_q_proj` | 8 × 16 × 4,096 | 129,823 | 70,403 |
| `llama3_gate_proj` | 8 × 16 × 4,096 | 129,823 | 70,403 |

The ResNet unpruned baseline also matched its reference at 8 × 16 × 160. The full workload inputs and FP32 references were generated for all three workloads. The table reports simulated submatrices.

## Numerical formats and RTL comparison

The same formats apply to ResNet50 and both Llama workloads. The workload selects matrix shapes and synthetic data. The benchmark implementation selects the numerical format.

| Data | Numerical format | Representation |
| --- | --- | --- |
| Trial 2 activations and weights | BF16 | NumPy float32 arrays containing exactly representable BF16 values. The simulator packs them as 16-bit BF16 words in DRAM. |
| Full-size NumPy references | FP32 | `C_dense_fp32` and `C_pruned_fp32` are FP32 matrix products computed from the BF16-rounded operands. |
| Simulated results and schedule-specific references | BF16 | The model uses FP32 products and tree results, rounds each array pass to BF16, and rounds each cross-pass vector addition to BF16. The optional output NPZ stores those values as NumPy float32 arrays. |
| Original PyTorch generator | FP16 | The original script generated FP16 activations and weights and saved both operands and reference outputs as FP16. |

For a generated Trial 2 package, inspect the metadata and array dtypes from the repository root:

```sh
python -c "import json, numpy as np; f=np.load('benchmarks/resnet50_conv1.npz'); print(json.loads(f['metadata_json'].item())['dtype']); print({k: str(f[k].dtype) for k in ('A', 'W_dense', 'C_dense_fp32', 'C_pruned_fp32')})"
```

The expected first line is `bf16`. The array dtypes on the next line are `float32`. The metadata identifies the modeled operand precision and the NumPy dtype identifies the storage container.

The upstream dense MEISSA model documents a BF16 RTL configuration with FP32 products and adder-tree sums, followed by BF16 reduction. The team's current sparse RTL configuration still needs confirmation. Check the activation and weight format, accumulator precision, and output format and rounding in its top-level configuration and testbench. A 16-bit width alone leaves the format ambiguous. IEEE FP16 uses 5 exponent bits and 10 fraction bits. BF16 uses 8 exponent bits and 7 fraction bits. A testbench that drives the value 1.0 as a raw word uses `16'h3c00` for FP16 or `16'h3f80` for BF16. If the team's sparse RTL uses FP16 or a different output or accumulation format, Trial 2's numerical reference needs to be matched to that configuration.

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
