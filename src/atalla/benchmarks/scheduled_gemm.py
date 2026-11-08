"""Generate and run scheduler-driven BF16 GEMM programs for dense and 2:4 MEISSA."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from atalla.atalla_platform import build_atalla_platform
from base.dtype import bf16_round
from scheduler.assembler import Instr, encode
from scheduler.isa import INST_W, PACKET_BYTE_W, PACKET_SIZE
from .reference import reference_product
from .sparse_codec import decode_bf16, encode_bf16

SIZE = 16
VECTOR_WIDTH = 32
MAX_ROWS = 32
DATA_BASE = 0x1000000


@dataclass
class GemmProgram:
    instructions: dict
    data: dict
    output_address: int
    weight_hazards: dict
    weight_bytes: int
    metadata_bytes: int
    passes: int


def make_program(a, w, indices):
    """One output block, with all K passes, timed accumulation and DRAM stores."""
    rows, k = a.shape
    sparse = indices is not None
    step = 2 * SIZE if sparse else SIZE
    passes = (k + step - 1) // step
    data = {}
    address = DATA_BASE

    def put(blob):
        nonlocal address
        start = address
        data.update((start + i, int.from_bytes(blob[i:i + 4], "little"))
                    for i in range(0, len(blob), 4))
        address += (len(blob) + 63) // 64 * 64
        return start

    act_addresses = []
    weight_addresses = []
    weight_lanes = SIZE + SIZE // 8 if sparse else SIZE
    for acts, weights, positions in gemm_passes(a, w, indices):
        act_addresses.append(put(encode_bf16(acts)))
        if sparse:
            # Each column is 16 BF16 weights followed by 32 bits of metadata.
            # An all-ones BF16 exponent would contain both indices of at least
            # one group set to 3, which valid metadata forbids. The VLSU's
            # BF16 bitcast therefore preserves both metadata words exactly.
            words = positions.T.astype(np.uint32) << (2 * np.arange(SIZE, dtype=np.uint32))
            packed = np.bitwise_or.reduce(words, axis=1).astype("<u4").tobytes()
            value_bytes = encode_bf16(weights.T)
            weight_addresses.append(put(b"".join(
                value_bytes[col * SIZE * 2:(col + 1) * SIZE * 2] + packed[col * 4:(col + 1) * 4]
                for col in range(SIZE))))
        else:
            weight_addresses.append(put(encode_bf16(weights.T)))

    instructions = {}
    weight_hazards = {}
    nop = encode(Instr("nop.s", [], 0))

    def emit(name, *operands, hazard=None):
        pc = len(instructions) * PACKET_BYTE_W
        word = encode(Instr(name, list(map(str, operands)), 0))
        packet = word
        for _ in range(PACKET_SIZE - 1):
            packet = (packet << INST_W) | nop
        instructions[pc] = packet
        if hazard is not None:
            weight_hazards[pc] = hazard

    def set_scalar(reg, value):
        if 0 <= value < 2048:
            emit("addi.s", reg, "x0", value)
        else:
            emit("lui.s", reg, value >> 7)
            emit("addi.s", reg, reg, value & 0x7F)

    def dma(kind, token, dram, pad, nr, nc):
        set_scalar("x5", dram)
        set_scalar("x6", (pad << 30) | ((nr - 1) << 25) | ((nc - 1) << 20))
        emit(f"scpad.{kind}", token, "x5", "x6")

    def vec_mem(kind, vreg, token, offset, width, pad, *, hazard=None):
        set_scalar("x7", offset)
        emit(f"vreg.{kind}", vreg, token, "x7", width - 1, pad, hazard=hazard)

    # The activation, weight and output traffic uses three original pads.
    # Weight DMA alternates between two regions in pad 1. The next transfer
    # is issued before this pass computes, while this pass uses its VRF copy.
    set_scalar("x3", SIZE * VECTOR_WIDTH * 2)
    dma("ld", "x2", weight_addresses[0], 1, SIZE, weight_lanes)
    for p in range(passes):
        token = ("x2", "x3")[p % 2]
        dma("ld", "x1", act_addresses[p], 0, rows, step)
        for row in range(rows):
            vec_mem("ld", 32 + row, "x1", row, step, 0)
        for col in reversed(range(SIZE)):
            vec_mem("ld", 16 + col, token, col, weight_lanes, 1,
                    hazard=("s", int(token[1:])))
        if p + 1 < passes:
            dma("ld", ("x3", "x2")[p % 2], weight_addresses[p + 1], 1, SIZE, weight_lanes)
        for col in reversed(range(SIZE)):
            emit("lw.vi", 0, 16 + col, 0, 0, hazard=("v", 16 + col))
        for row in range(rows):
            emit("gemm.vv", 64 + row, 32 + row, 0, 0)
        # This dependency also prevents the next pass from shifting weights
        # while any activation from the current pass is still in MEISSA.
        for row in range(rows):
            emit("add.vv", 96 + row, 64 + row, 96 + row if p else 0, 0)
    for row in range(rows):
        vec_mem("st", 96 + row, "x4", row, SIZE, 2)
    dma("st", "x4", address, 2, rows, SIZE)
    emit("halt.s")
    return GemmProgram(instructions, data, address, weight_hazards,
                       passes * SIZE * weight_lanes * 2,
                       passes * SIZE * (weight_lanes - SIZE) * 2, passes)


def gemm_passes(a, w, indices):
    sparse = indices is not None
    step = 2 * SIZE if sparse else SIZE
    for first in range(0, a.shape[1], step):
        acts = a[:, first:first + step]
        acts = np.pad(acts, ((0, 0), (0, step - acts.shape[1])))
        start = first // 2 if sparse else first
        weights = w[start:start + SIZE]
        padding = SIZE - len(weights)
        weights = np.pad(weights, ((0, padding), (0, 0)))
        positions = None
        if sparse:
            positions = indices[start:start + SIZE]
            default_positions = np.tile([[0], [1]], (padding // 2, SIZE))
            positions = np.concatenate((positions, default_positions), axis=0)
        yield acts, weights, positions


def run_block(a, w, *, indices=None, mux_latency=0, dram_latency=6, burst_bytes=8,
              timeline=False, activity=False, cycle_limit=5_000_000):
    rows, k = a.shape
    sparse = indices is not None
    if not 1 <= rows <= MAX_ROWS or k < 1 or w.shape != (k // 2 if sparse else k, SIZE):
        raise ValueError("expected an activation block of 1–32 rows and 16 weight columns")
    if sparse and (k % 4 or indices.shape != w.shape):
        raise ValueError("expected matching 2:4 values and indices, with K divisible by four")
    job = make_program(a, w, indices)
    platform = build_atalla_platform(
        job.instructions, job.data, systolic_array="meissa_2of4" if sparse else "meissa",
        meissa_size=SIZE, meissa_mixed=True, mux_latency=mux_latency, dram_latency=dram_latency,
        burst_bytes=burst_bytes, perf_timeline=True, activity=activity)
    weight_wait = Counter()
    wait_cycles = []
    while not platform.core.done:
        if platform.cycle >= cycle_limit:
            raise RuntimeError(f"no halt after {cycle_limit} cycles (PC {platform.core.pc:#x})")
        platform.tick()
        decode = platform.core.decode2
        hazard = job.weight_hazards.get(decode.in_pc)
        if (hazard is not None and decode.last_stall == "hazard" and decode.hazard_reg == hazard
                and not decode.in_flush and not decode.in_halt):
            stage = "dma_to_scratchpad" if hazard[0] == "s" else "vlsu_to_vrf"
            weight_wait[stage] += 1
            if platform.sa.algo_counts["act_rows"]:
                weight_wait["after_first_activation"] += 1
            if platform.sa._is_drained():
                weight_wait["array_idle"] += 1
            if timeline:
                wait_cycles.append(platform.cycle - 1)
    shape = (a.shape[0], SIZE)
    raw = platform.dram.read(job.output_address, a.shape[0] * SIZE * 2)
    result = decode_bf16(raw, shape)
    expected = np.zeros(shape, dtype=np.float32)
    for acts, weights, positions in gemm_passes(a, w, indices):
        expected = bf16_round(expected + reference_product(acts, weights, positions))
    report = platform.perf.report()
    busy = platform.perf.timeline_busy
    concurrency = Counter(sum(unit.startswith("vlsu") for unit in units) for units in busy)
    report.update(
        correctness_passed=bool(np.array_equal(result.view(np.uint32), expected.view(np.uint32))),
        max_absolute_error=float(np.max(np.abs(result - expected))),
        weight_wait_cycles=weight_wait["dma_to_scratchpad"] + weight_wait["vlsu_to_vrf"],
        weight_wait_by_stage={s: weight_wait[s] for s in ("dma_to_scratchpad", "vlsu_to_vrf")},
        weight_wait_array_idle_cycles=weight_wait["array_idle"],
        weight_wait_after_first_activation_cycles=weight_wait["after_first_activation"],
        vlsu_active_count_cycles=dict(sorted(concurrency.items())),
        weight_payload_bytes=job.weight_bytes, metadata_bytes=job.metadata_bytes,
        array_passes=job.passes,
    )
    if timeline:
        report["timeline"] = dict(bucket=platform.perf.timeline, busy=busy,
                                  weight_wait_cycles=wait_cycles)
    return result, expected, report, platform.activity
