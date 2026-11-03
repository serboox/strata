import argparse
import ctypes
import dataclasses
import glob
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_reader
from gguf_graft import COPY_CHUNK, GIB, SYNC_EVERY, Layout, align_up, copy_range, pack_string

Q8_0_TYPE_ID = 8
Q6_K_TYPE_ID = 14
Q8_0_BLOCK_WEIGHTS = 32
Q8_0_BLOCK_BYTES = 34
Q6_K_BLOCK_WEIGHTS = 256
Q6_K_BLOCK_BYTES = 210
SUBBLOCK_WEIGHTS = 16
SUBBLOCKS_PER_BLOCK = Q6_K_BLOCK_WEIGHTS // SUBBLOCK_WEIGHTS
HALF_WEIGHTS = Q6_K_BLOCK_WEIGHTS // 2
QUARTER_WEIGHTS = Q6_K_BLOCK_WEIGHTS // 8
Q6_K_LOW_BYTES = Q6_K_BLOCK_WEIGHTS // 2
Q6_K_HIGH_BYTES = Q6_K_BLOCK_WEIGHTS // 4
Q6_K_HIGH_START = Q6_K_LOW_BYTES
Q6_K_SCALES_START = Q6_K_HIGH_START + Q6_K_HIGH_BYTES
Q6_K_D_START = Q6_K_SCALES_START + SUBBLOCKS_PER_BLOCK
Q6_K_LEVEL_LIMIT = 32
Q6_K_SCALE_LIMIT = 127
Q6_K_SCALE_RANGE = 128
SCALE_SEARCH_STEPS = tuple(step for step in range(-9, 10) if step)
SCALE_SEARCH_STEP_SIZE = 0.1
GROUP_MAX_EPS = np.float32(1e-15)
NEAREST_INT_MAGIC = 12582912.0
Q8_0_LEVEL_LIMIT = 127
SLICE_BLOCKS = 256

DEFAULT_CHUNK_MIB = 64
DISK_HEADROOM = GIB
PROGRESS_EVERY = 50
SPOT_CHECK_MAX_ERROR = 0.05

SELFTEST_SEED = 20260503
SELFTEST_DEFAULT_BLOCKS = 20000
SELFTEST_SCALAR_BLOCKS = 200
SELFTEST_BENCHMARK_BLOCKS = 100000
SELFTEST_MAX_ROUND_TRIP_ERROR = 0.02
GGML_CHECKOUT = Path.home() / "clang/github.com/ggml-org/llama.cpp"
GGML_LIBRARY_ENV = "STRATA_GGML_BASE_LIB"


class Group(NamedTuple):
    pattern: str
    description: str


GROUPS = {
    "attn": Group(r"^blk\.\d+\.attn_(q|k|v|output)\.weight$", "full-attention projections"),
    "ssm": Group(r"^blk\.\d+\.(attn_qkv|attn_gate|ssm_out)\.weight$", "gated delta-net projections"),
    "shexp": Group(r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$", "shared-expert feed-forward"),
    "head": Group(r"^output\.weight$", "output head"),
    "embd": Group(r"^token_embd\.weight$", "token embedding"),
    "mtp": Group(r"^blk\.\d+\.nextn\.eh_proj\.weight$", "multi-token-prediction input projection"),
}
DEFAULT_GROUPS = ("attn", "ssm", "shexp")
ALL_GROUPS_NAME = "all"


def _nearest_int_scalar(value) -> int:
    shifted = np.float32(value) + np.float32(NEAREST_INT_MAGIC)
    bits = struct.unpack("<i", struct.pack("<f", shifted))[0]
    return (bits & 0x007FFFFF) - 0x00400000


def reference_make_qx_quants(weights: list) -> tuple:
    """Line-by-line scalar port of ggml make_qx_quants(16, 32, x, L, 1, NULL) in float32 arithmetic; slow, used for checks only."""
    peak_abs = np.float32(0)
    peak = np.float32(0)
    for weight in weights:
        magnitude = abs(weight)
        if magnitude > peak_abs:
            peak_abs = magnitude
            peak = weight
    if peak_abs < GROUP_MAX_EPS:
        return np.float32(0), [0] * len(weights)
    iscale = np.float32(-Q6_K_LEVEL_LIMIT) / peak
    sumlx = np.float32(0)
    suml2 = np.float32(0)
    levels = []
    for weight in weights:
        level = max(-Q6_K_LEVEL_LIMIT, min(Q6_K_LEVEL_LIMIT - 1, _nearest_int_scalar(iscale * weight)))
        levels.append(level + Q6_K_LEVEL_LIMIT)
        square = weight * weight
        sumlx += square * weight * np.float32(level)
        suml2 += square * np.float32(level) * np.float32(level)
    scale = sumlx / suml2 if suml2 else np.float32(0)
    best = scale * sumlx
    for step in SCALE_SEARCH_STEPS:
        iscale = -(np.float32(Q6_K_LEVEL_LIMIT) + np.float32(SCALE_SEARCH_STEP_SIZE) * np.float32(step)) / peak
        sumlx = np.float32(0)
        suml2 = np.float32(0)
        trial = []
        for weight in weights:
            level = max(-Q6_K_LEVEL_LIMIT, min(Q6_K_LEVEL_LIMIT - 1, _nearest_int_scalar(iscale * weight)))
            trial.append(level + Q6_K_LEVEL_LIMIT)
            square = weight * weight
            sumlx += square * weight * np.float32(level)
            suml2 += square * np.float32(level) * np.float32(level)
        if suml2 > 0 and sumlx * sumlx > best * suml2:
            levels = trial
            scale = sumlx / suml2
            best = scale * sumlx
    return scale, levels


def reference_quantize_q6_k_block(weights) -> bytes:
    """Line-by-line scalar port of one block of ggml quantize_row_q6_K_ref; slow, used for checks only."""
    weights = [np.float32(weight) for weight in weights]
    levels = [0] * Q6_K_BLOCK_WEIGHTS
    scales = []
    max_scale = np.float32(0)
    max_abs_scale = np.float32(0)
    for group in range(SUBBLOCKS_PER_BLOCK):
        first = SUBBLOCK_WEIGHTS * group
        scale, group_levels = reference_make_qx_quants(weights[first:first + SUBBLOCK_WEIGHTS])
        levels[first:first + SUBBLOCK_WEIGHTS] = group_levels
        scales.append(scale)
        if abs(scale) > max_abs_scale:
            max_abs_scale = abs(scale)
            max_scale = scale
    if max_abs_scale < GROUP_MAX_EPS:
        return bytes(Q6_K_BLOCK_BYTES)
    iscale = np.float32(-Q6_K_SCALE_RANGE) / max_scale
    block_scale = np.float16(np.float32(1) / iscale)
    scale_levels = [min(Q6_K_SCALE_LIMIT, _nearest_int_scalar(iscale * scale)) for scale in scales]
    for group in range(SUBBLOCKS_PER_BLOCK):
        step = np.float32(block_scale) * np.float32(scale_levels[group])
        if not step:
            continue
        for index in range(SUBBLOCK_WEIGHTS):
            level = _nearest_int_scalar(weights[SUBBLOCK_WEIGHTS * group + index] / step)
            levels[SUBBLOCK_WEIGHTS * group + index] = max(-Q6_K_LEVEL_LIMIT, min(Q6_K_LEVEL_LIMIT - 1, level)) + Q6_K_LEVEL_LIMIT
    low_bits = bytearray(Q6_K_LOW_BYTES)
    high_bits = bytearray(Q6_K_HIGH_BYTES)
    for half in range(0, Q6_K_BLOCK_WEIGHTS, HALF_WEIGHTS):
        for index in range(QUARTER_WEIGHTS):
            first, second, third, fourth = (levels[half + index + QUARTER_WEIGHTS * quarter] for quarter in range(4))
            low_bits[half // 2 + index] = (first & 0xF) | ((third & 0xF) << 4)
            low_bits[half // 2 + index + QUARTER_WEIGHTS] = (second & 0xF) | ((fourth & 0xF) << 4)
            high_bits[half // 4 + index] = (first >> 4) | ((second >> 4) << 2) | ((third >> 4) << 4) | ((fourth >> 4) << 6)
    return bytes(low_bits) + bytes(high_bits) + struct.pack("<16b", *scale_levels) + np.float16(block_scale).tobytes()


def _sequential_sum(rows: np.ndarray) -> np.ndarray:
    """Left-to-right float32 sum over the first axis, as the C loop does; numpy's own sum is pairwise and rounds differently."""
    total = rows[0].copy()
    for row in rows[1:]:
        total += row
    return total


def _make_qx_quants(rows: np.ndarray) -> tuple:
    """Vectorised ggml make_qx_quants(16, 32, ..., rmse_type 1) over many 16-weight groups.
    `rows` is float32 [16, groups]; returns the scale [groups] and the unsigned levels [16, groups] (0..63 as float32)."""
    columns = np.arange(rows.shape[1])
    magnitudes = np.abs(rows)
    peak_row = magnitudes.argmax(axis=0)
    is_zero = magnitudes[peak_row, columns] < GROUP_MAX_EPS
    peak = np.where(is_zero, np.float32(1), rows[peak_row, columns])
    squares = rows * rows
    cubes = squares * rows
    level_low, level_high = np.float32(-Q6_K_LEVEL_LIMIT), np.float32(Q6_K_LEVEL_LIMIT - 1)

    def trial(iscale):
        levels = np.rint(iscale * rows)
        np.clip(levels, level_low, level_high, out=levels)
        return levels, _sequential_sum(cubes * levels), _sequential_sum((squares * levels) * levels)

    with np.errstate(all="ignore"):
        best_levels, sumlx, suml2 = trial(np.float32(-Q6_K_LEVEL_LIMIT) / peak)
        scale = np.where(suml2 != 0, sumlx / suml2, np.float32(0))
        best = scale * sumlx
        for step in SCALE_SEARCH_STEPS:
            levels, sumlx, suml2 = trial(-(np.float32(Q6_K_LEVEL_LIMIT) + np.float32(SCALE_SEARCH_STEP_SIZE) * np.float32(step)) / peak)
            better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
            best_levels = np.where(better, levels, best_levels)
            scale = np.where(better, sumlx / suml2, scale)
            best = np.where(better, scale * sumlx, best)
    return np.where(is_zero, np.float32(0), scale), np.where(is_zero, np.float32(0), best_levels + np.float32(Q6_K_LEVEL_LIMIT))


def _quantize_slice(blocks: np.ndarray) -> np.ndarray:
    count = blocks.shape[0]
    rows = np.ascontiguousarray(blocks.reshape(count * SUBBLOCKS_PER_BLOCK, SUBBLOCK_WEIGHTS).T)
    scales, search_levels = _make_qx_quants(rows)
    scales = scales.reshape(count, SUBBLOCKS_PER_BLOCK)
    search_levels = search_levels.reshape(SUBBLOCK_WEIGHTS, count, SUBBLOCKS_PER_BLOCK).transpose(1, 2, 0)
    magnitudes = np.abs(scales)
    peak_group = magnitudes.argmax(axis=1)
    block_index = np.arange(count)
    is_zero = magnitudes[block_index, peak_group] < GROUP_MAX_EPS
    max_scale = np.where(is_zero, np.float32(1), scales[block_index, peak_group])
    with np.errstate(all="ignore"):
        iscale = np.float32(-Q6_K_SCALE_RANGE) / max_scale
        block_scale = (np.float32(1) / iscale).astype("<f2")
        scale_levels = np.minimum(np.rint(iscale[:, None] * scales), np.float32(Q6_K_SCALE_LIMIT)).astype(np.int8)
        group_step = block_scale.astype(np.float32)[:, None] * scale_levels.astype(np.float32)
        has_step = group_step != 0
        refined = np.rint(blocks.reshape(count, SUBBLOCKS_PER_BLOCK, SUBBLOCK_WEIGHTS) / np.where(has_step, group_step, np.float32(1))[:, :, None])
    np.clip(refined, -Q6_K_LEVEL_LIMIT, Q6_K_LEVEL_LIMIT - 1, out=refined)
    refined += np.float32(Q6_K_LEVEL_LIMIT)
    levels = np.where(has_step[:, :, None], refined, search_levels).astype(np.uint8).reshape(count, 2, 4, QUARTER_WEIGHTS)
    low, high = levels & 0x0F, levels >> 4
    result = np.empty((count, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    low_bits = result[:, :Q6_K_LOW_BYTES].reshape(count, 2, 2, QUARTER_WEIGHTS)
    low_bits[:, :, 0] = low[:, :, 0] | (low[:, :, 2] << 4)
    low_bits[:, :, 1] = low[:, :, 1] | (low[:, :, 3] << 4)
    result[:, Q6_K_HIGH_START:Q6_K_SCALES_START] = (high[:, :, 0] | (high[:, :, 1] << 2) | (high[:, :, 2] << 4) | (high[:, :, 3] << 6)).reshape(count, Q6_K_HIGH_BYTES)
    result[:, Q6_K_SCALES_START:Q6_K_D_START] = scale_levels.view(np.uint8)
    result[:, Q6_K_D_START:] = block_scale.view(np.uint8).reshape(count, 2)
    result[is_zero] = 0
    return result


def quantize_q6_k(values: np.ndarray) -> np.ndarray:
    """float32 [n_blocks, 256] -> uint8 [n_blocks, 210], bit-for-bit what ggml quantize_row_q6_K_ref produces."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != Q6_K_BLOCK_WEIGHTS:
        raise ValueError(f"expected an array of shape [n, {Q6_K_BLOCK_WEIGHTS}], got {values.shape}")
    result = np.empty((values.shape[0], Q6_K_BLOCK_BYTES), dtype=np.uint8)
    for start in range(0, values.shape[0], SLICE_BLOCKS):
        result[start:start + SLICE_BLOCKS] = _quantize_slice(values[start:start + SLICE_BLOCKS])
    return result


def dequantize_q6_k(blocks: np.ndarray) -> np.ndarray:
    """uint8 [n_blocks, 210] -> float32 [n_blocks, 256], as ggml dequantize_row_q6_K."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    count = blocks.shape[0]
    low_bits = blocks[:, :Q6_K_LOW_BYTES].reshape(count, 2, 2, QUARTER_WEIGHTS)
    high_bits = blocks[:, Q6_K_HIGH_START:Q6_K_SCALES_START].reshape(count, 2, QUARTER_WEIGHTS)
    levels = np.empty((count, 2, 4, QUARTER_WEIGHTS), dtype=np.uint8)
    levels[:, :, 0] = (low_bits[:, :, 0] & 0x0F) | ((high_bits & 3) << 4)
    levels[:, :, 1] = (low_bits[:, :, 1] & 0x0F) | (((high_bits >> 2) & 3) << 4)
    levels[:, :, 2] = (low_bits[:, :, 0] >> 4) | (((high_bits >> 4) & 3) << 4)
    levels[:, :, 3] = (low_bits[:, :, 1] >> 4) | (((high_bits >> 6) & 3) << 4)
    scale_levels = np.ascontiguousarray(blocks[:, Q6_K_SCALES_START:Q6_K_D_START]).view(np.int8).astype(np.float32)
    block_scale = np.ascontiguousarray(blocks[:, Q6_K_D_START:]).view("<f2").astype(np.float32)
    group_step = block_scale * scale_levels
    signed = levels.reshape(count, Q6_K_BLOCK_WEIGHTS).astype(np.int16) - Q6_K_LEVEL_LIMIT
    return np.repeat(group_step, SUBBLOCK_WEIGHTS, axis=1) * signed.astype(np.float32)


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """uint8 [n_blocks, 34] -> float32 [n_blocks, 32]."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    block_scale = np.ascontiguousarray(blocks[:, :2]).view("<f2").astype(np.float32)
    values = blocks[:, 2:].view(np.int8).astype(np.float32)
    values *= block_scale
    return values


def quantize_q8_0(values: np.ndarray) -> np.ndarray:
    """float32 [n_blocks, 32] -> uint8 [n_blocks, 34], as ggml quantize_row_q8_0_ref."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.abs(values).max(axis=1) / np.float32(Q8_0_LEVEL_LIMIT)
        inverse = np.where(step != 0, np.float32(1) / step, np.float32(0))
    scaled = (values * inverse[:, None]).astype(np.float64)
    levels = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(np.int8)
    result = np.empty((values.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    result[:, :2] = step.astype("<f2").view(np.uint8).reshape(-1, 2)
    result[:, 2:] = levels.view(np.uint8)
    return result


def structured_blocks() -> list:
    """Named edge-case blocks of 256 float32 weights."""
    rng = np.random.default_rng(SELFTEST_SEED)
    ramp = np.linspace(-1, 1, Q6_K_BLOCK_WEIGHTS, dtype=np.float32)
    outlier = rng.normal(0, 0.02, Q6_K_BLOCK_WEIGHTS).astype(np.float32)
    outlier[17] = 100.0
    lone_outlier = np.zeros(Q6_K_BLOCK_WEIGHTS, dtype=np.float32)
    lone_outlier[200] = -3.5
    mixed_scale = rng.normal(0, 1, Q6_K_BLOCK_WEIGHTS).astype(np.float32)
    mixed_scale[:64] *= 1e-4
    mixed_scale[64:128] *= 1e-8
    ties = (np.arange(Q6_K_BLOCK_WEIGHTS, dtype=np.float32) - 128) / np.float32(2)
    return [
        ("zeros", np.zeros(Q6_K_BLOCK_WEIGHTS, dtype=np.float32)),
        ("constant", np.full(Q6_K_BLOCK_WEIGHTS, 0.5, dtype=np.float32)),
        ("negative constant", np.full(Q6_K_BLOCK_WEIGHTS, -0.25, dtype=np.float32)),
        ("one outlier", outlier),
        ("single non-zero weight", lone_outlier),
        ("tiny 1e-6", (rng.normal(0, 1, Q6_K_BLOCK_WEIGHTS) * 1e-6).astype(np.float32)),
        ("large 1e4", (rng.normal(0, 1, Q6_K_BLOCK_WEIGHTS) * 1e4).astype(np.float32)),
        ("below the epsilon", np.full(Q6_K_BLOCK_WEIGHTS, 1e-20, dtype=np.float32)),
        ("just above the epsilon", np.full(Q6_K_BLOCK_WEIGHTS, 3e-15, dtype=np.float32)),
        ("ramp", ramp),
        ("alternating sign", np.where(np.arange(Q6_K_BLOCK_WEIGHTS) % 2, 1.0, -1.0).astype(np.float32)),
        ("mixed scales", mixed_scale),
        ("half-integer ties", ties),
    ]


class GgmlReference:
    """The real ggml quantisers, called through ctypes from the CPU-only libggml-base."""

    def __init__(self, path: str):
        self.path = path
        library = ctypes.CDLL(path)
        void_pointer = ctypes.c_void_p
        for name in ("quantize_row_q6_K_ref", "dequantize_row_q6_K", "quantize_row_q8_0_ref", "dequantize_row_q8_0"):
            function = getattr(library, name)
            function.argtypes = [void_pointer, void_pointer, ctypes.c_int64]
            function.restype = None
        self.library = library

    def _apply(self, function, source: np.ndarray, out_shape: tuple, out_dtype, weights_per_block: int) -> np.ndarray:
        source = np.ascontiguousarray(source)
        out = np.zeros(out_shape, dtype=out_dtype)
        function(source.ctypes.data, out.ctypes.data, source.shape[0] * weights_per_block)
        return out

    def quantize_q6_k(self, values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.float32)
        return self._apply(self.library.quantize_row_q6_K_ref, values, (values.shape[0], Q6_K_BLOCK_BYTES), np.uint8, Q6_K_BLOCK_WEIGHTS)

    def dequantize_q6_k(self, blocks: np.ndarray) -> np.ndarray:
        return self._apply(self.library.dequantize_row_q6_K, blocks, (blocks.shape[0], Q6_K_BLOCK_WEIGHTS), np.float32, Q6_K_BLOCK_WEIGHTS)

    def quantize_q8_0(self, values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.float32)
        return self._apply(self.library.quantize_row_q8_0_ref, values, (values.shape[0], Q8_0_BLOCK_BYTES), np.uint8, Q8_0_BLOCK_WEIGHTS)

    def dequantize_q8_0(self, blocks: np.ndarray) -> np.ndarray:
        return self._apply(self.library.dequantize_row_q8_0, blocks, (blocks.shape[0], Q8_0_BLOCK_WEIGHTS), np.float32, Q8_0_BLOCK_WEIGHTS)


def load_ggml_reference() -> tuple:
    """Returns (GgmlReference, "") or (None, the reason it is unavailable)."""
    override = os.environ.get(GGML_LIBRARY_ENV)
    candidates = [override] if override else sorted(glob.glob(str(GGML_CHECKOUT / "build*/bin/libggml-base.so")))
    if not candidates:
        return None, f"no libggml-base.so under {GGML_CHECKOUT}/build*/bin (set {GGML_LIBRARY_ENV} to a path)"
    failures = []
    for candidate in candidates:
        try:
            return GgmlReference(candidate), ""
        except (OSError, AttributeError) as error:
            failures.append(f"{candidate}: {error}")
    return None, "; ".join(failures)


def parse_groups(text: str) -> tuple:
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise ValueError("the group list is empty")
    if ALL_GROUPS_NAME in names:
        return tuple(GROUPS)
    unknown = [name for name in names if name not in GROUPS]
    if unknown:
        raise ValueError(f"unknown group {', '.join(repr(name) for name in unknown)}; valid groups: {', '.join(GROUPS)}, {ALL_GROUPS_NAME}")
    return tuple(dict.fromkeys(names))


@dataclasses.dataclass
class TensorPlan:
    tensor: gguf_reader.TensorInfo
    group: str | None
    convert: bool
    kept_reason: str
    old_size: int
    new_size: int
    new_offset: int

    @property
    def new_type_id(self) -> int:
        return Q6_K_TYPE_ID if self.convert else self.tensor.type_id


@dataclasses.dataclass
class Plan:
    source: Path
    layout: Layout
    groups: tuple
    entries: list
    header: bytes
    data_start: int
    total_size: int


def build_plan(source: Path, groups: tuple) -> Plan:
    layout = Layout(source)
    parsed = gguf_reader.GGUFFile(source)
    patterns = {name: re.compile(GROUPS[name].pattern) for name in groups}
    entries = []
    cursor = 0
    for tensor in parsed.tensors:
        old_size = tensor.expected_bytes()
        if old_size is None or layout.data_start + tensor.offset + old_size > layout.size:
            raise ValueError(f"cannot size the tensor {tensor.name} ({tensor.type_name}) of {source.name}")
        group = next((name for name in groups if patterns[name].match(tensor.name)), None)
        convert, kept_reason = False, ""
        if group is not None and tensor.type_id != Q8_0_TYPE_ID:
            kept_reason = f"kept (type {tensor.type_name})"
        elif group is not None and tensor.shape[0] % Q6_K_BLOCK_WEIGHTS:
            kept_reason = f"kept (row length {tensor.shape[0]} is not a multiple of {Q6_K_BLOCK_WEIGHTS})"
        elif group is not None:
            convert = True
        new_size = tensor.elements // Q6_K_BLOCK_WEIGHTS * Q6_K_BLOCK_BYTES if convert else old_size
        entries.append(TensorPlan(tensor, group, convert, kept_reason, old_size, new_size, cursor))
        cursor = align_up(cursor + new_size, layout.alignment)
    infos = b"".join(
        pack_string(entry.tensor.name) + struct.pack("<I", len(entry.tensor.shape))
        + b"".join(struct.pack("<Q", dim) for dim in entry.tensor.shape) + struct.pack("<IQ", entry.new_type_id, entry.new_offset)
        for entry in entries)
    header = layout.head[:layout.kv_end] + infos
    data_start = align_up(len(header), layout.alignment)
    return Plan(source, layout, groups, entries, header, data_start, data_start + cursor)


def group_statistics(plan: Plan) -> dict:
    statistics = {name: {"tensors": 0, "before": 0, "after": 0} for name in plan.groups}
    for entry in plan.entries:
        if entry.convert:
            statistics[entry.group]["tensors"] += 1
            statistics[entry.group]["before"] += entry.old_size
            statistics[entry.group]["after"] += entry.new_size
    return statistics


def nearest_existing_directory(path: Path) -> Path:
    directory = path.resolve().parent
    while not directory.exists():
        directory = directory.parent
    return directory


def report_plan(plan: Plan, dest: Path, out) -> bool:
    """Prints the dry-run report; returns whether the output would fit on the disk."""
    print(f"requant_dense dry run: {plan.source} -> {dest}", file=out)
    print(f"{'group':<8}{'tensors':>8}{'Q8_0 bytes':>16}{'Q6_K bytes':>16}{'saved bytes':>16}{'saved':>8}", file=out)
    statistics = group_statistics(plan)
    total_before = total_after = 0
    for name, stats in statistics.items():
        saved = stats["before"] - stats["after"]
        percent = 100.0 * saved / stats["before"] if stats["before"] else 0.0
        print(f"{name:<8}{stats['tensors']:>8}{stats['before']:>16}{stats['after']:>16}{saved:>16}{percent:>7.1f}%", file=out)
        total_before += stats["before"]
        total_after += stats["after"]
    saved = total_before - total_after
    print(f"{'total':<8}{sum(stats['tensors'] for stats in statistics.values()):>8}{total_before:>16}{total_after:>16}{saved:>16}"
          f"{100.0 * saved / total_before if total_before else 0.0:>7.1f}%", file=out)
    skipped = [entry for entry in plan.entries if entry.kept_reason]
    print(f"selected tensors left as they are: {len(skipped)}", file=out)
    for entry in skipped:
        print(f"  {entry.tensor.name}: {entry.kept_reason}", file=out)
    print(f"source file size: {plan.layout.size} bytes ({plan.layout.size / GIB:.3f} GiB)", file=out)
    print(f"new file size: {plan.total_size} bytes ({plan.total_size / GIB:.3f} GiB)", file=out)
    free = shutil.disk_usage(nearest_existing_directory(dest)).free
    needed = plan.total_size + DISK_HEADROOM
    fits = free >= needed
    print(f"free disk space at {nearest_existing_directory(dest)}: {free / GIB:.3f} GiB; needed {needed / GIB:.3f} GiB "
          f"(new file + 1 GiB): {'would fit' if fits else 'would NOT fit'}", file=out)
    return fits


class Sink:
    def __init__(self, path: Path):
        self.descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self.written = 0
        self.unsynced = 0

    def write(self, data) -> None:
        view = memoryview(data).cast("B")
        while view:
            count = os.write(self.descriptor, view)
            view = view[count:]
            self.written += count
            self.unsynced += count
        if self.unsynced >= SYNC_EVERY:
            self.flush()

    def flush(self) -> None:
        os.fdatasync(self.descriptor)
        os.posix_fadvise(self.descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        self.unsynced = 0

    def close(self) -> None:
        self.flush()
        self.abort()

    def abort(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def read_range(source, start: int, size: int) -> bytes:
    source.seek(start)
    data = source.read(size)
    if len(data) != size:
        raise OSError(f"{source.name} ended {size - len(data)} bytes early")
    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return data


def requantize_tensor(source, data_start: int, entry: TensorPlan, sink: Sink, chunk_bytes: int) -> None:
    tensor = entry.tensor
    row_weights = tensor.shape[0]
    rows = tensor.elements // row_weights
    source_row_bytes = row_weights // Q8_0_BLOCK_WEIGHTS * Q8_0_BLOCK_BYTES
    rows_per_chunk = max(1, chunk_bytes // (row_weights * 4))
    for first_row in range(0, rows, rows_per_chunk):
        count = min(rows_per_chunk, rows - first_row)
        raw = read_range(source, data_start + tensor.offset + first_row * source_row_bytes, count * source_row_bytes)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q8_0_BLOCK_BYTES)
        if not np.isfinite(np.ascontiguousarray(blocks[:, :2]).view("<f2")).all():
            raise ValueError(f"{tensor.name}: a Q8_0 block has a non-finite scale")
        values = dequantize_q8_0(blocks).reshape(-1, Q6_K_BLOCK_WEIGHTS)
        sink.write(quantize_q6_k(values))


def relative_rms_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    norm = float(np.sqrt(np.mean(reference ** 2)))
    difference = float(np.sqrt(np.mean((approximation.astype(np.float64) - reference) ** 2)))
    return difference / norm if norm else difference


def check_output(plan: Plan, output: Path) -> list:
    """Problems found in a finished output file: an empty list means none."""
    problems = []
    parsed = gguf_reader.GGUFFile(output)
    size = output.stat().st_size
    if size != plan.total_size:
        problems.append(f"the file is {size} bytes, the plan said {plan.total_size}")
    if parsed.data_start != plan.data_start or parsed.alignment != plan.layout.alignment:
        problems.append("the data start or the alignment differs from the plan")
    if [tensor.name for tensor in parsed.tensors] != [entry.tensor.name for entry in plan.entries]:
        problems.append("the tensor names or their order differ from the source")
        return problems
    with open(output, "rb") as handle:
        if handle.read(plan.layout.kv_end) != plan.layout.head[:plan.layout.kv_end]:
            problems.append("the metadata bytes differ from the source")
    previous_end = 0
    for entry, tensor in zip(plan.entries, parsed.tensors):
        expected_size = tensor.expected_bytes()
        if (tensor.type_id, tensor.shape) != (entry.new_type_id, entry.tensor.shape):
            problems.append(f"{tensor.name}: the type or shape is not the planned one")
        if expected_size is None or expected_size != entry.new_size:
            problems.append(f"{tensor.name}: the size is not the planned one")
            continue
        if tensor.offset % parsed.alignment or tensor.offset < previous_end or parsed.data_start + tensor.offset + expected_size > size:
            problems.append(f"{tensor.name}: bad offset")
        previous_end = tensor.offset + expected_size
    if problems:
        return problems
    with open(plan.source, "rb") as original, open(output, "rb") as converted:
        for entry in plan.entries:
            if entry.convert:
                problems.extend(spot_check_rows(plan, entry, original, converted, parsed.data_start))
    return problems


def spot_check_rows(plan: Plan, entry: TensorPlan, original, converted, converted_data_start: int) -> list:
    tensor = entry.tensor
    rows = tensor.elements // tensor.shape[0]
    source_row_bytes = tensor.shape[0] // Q8_0_BLOCK_WEIGHTS * Q8_0_BLOCK_BYTES
    new_row_bytes = tensor.shape[0] // Q6_K_BLOCK_WEIGHTS * Q6_K_BLOCK_BYTES
    problems = []
    for row in sorted({0, rows - 1}):
        before = dequantize_q8_0(np.frombuffer(
            read_range(original, plan.layout.data_start + tensor.offset + row * source_row_bytes, source_row_bytes), dtype=np.uint8).reshape(-1, Q8_0_BLOCK_BYTES))
        after = dequantize_q6_k(np.frombuffer(
            read_range(converted, converted_data_start + entry.new_offset + row * new_row_bytes, new_row_bytes), dtype=np.uint8).reshape(-1, Q6_K_BLOCK_BYTES))
        if relative_rms_error(before, after.reshape(before.shape)) > SPOT_CHECK_MAX_ERROR:
            problems.append(f"{tensor.name}: row {row} of the output does not match the source row")
    return problems


def convert(source: Path, dest: Path, groups: tuple, chunk_bytes: int = DEFAULT_CHUNK_MIB << 20, progress=None) -> Plan:
    """Writes `dest`: `source` with the selected Q8_0 tensors re-quantised to Q6_K; everything else byte for byte."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; remove it first")
    plan = build_plan(source, groups)
    if not any(entry.convert for entry in plan.entries):
        raise ValueError(f"no tensor of {source.name} would be converted by the groups {','.join(groups)}; a plain copy is not worth the disk space")
    free = shutil.disk_usage(nearest_existing_directory(dest)).free
    if free < plan.total_size + DISK_HEADROOM:
        raise OSError(f"not enough disk space at {dest.parent}: {free} bytes free, {plan.total_size + DISK_HEADROOM} needed")
    part = dest.with_name(dest.name + ".part")
    sink = Sink(part)
    try:
        sink.write(plan.header + bytes(plan.data_start - len(plan.header)))
        with open(source, "rb") as handle:
            for position, entry in enumerate(plan.entries, start=1):
                if entry.convert:
                    requantize_tensor(handle, plan.layout.data_start, entry, sink, chunk_bytes)
                else:
                    copy_range(handle, plan.layout.data_start + entry.tensor.offset, entry.old_size, sink.write, COPY_CHUNK)
                sink.write(bytes(align_up(entry.new_offset + entry.new_size, plan.layout.alignment) - entry.new_offset - entry.new_size))
                if progress is not None and (position % PROGRESS_EVERY == 0 or position == len(plan.entries)):
                    progress(position, len(plan.entries), sink.written, plan.total_size)
        sink.close()
        problems = check_output(plan, part)
        if problems:
            raise ValueError("the output failed its check: " + "; ".join(problems[:5]))
        if dest.exists():
            raise FileExistsError(f"{dest} appeared while the conversion ran")
        os.rename(part, dest)
    except BaseException:
        sink.abort()
        if part.exists():
            part.unlink()
        raise
    return plan


def selftest(blocks: int, out=sys.stdout, benchmark_blocks: int = SELFTEST_BENCHMARK_BLOCKS) -> bool:
    """Prints how the quantiser agrees with the references and how fast it is; returns False if any check failed."""
    rng = np.random.default_rng(SELFTEST_SEED)
    ok = True
    print("requant_dense selftest", file=out)

    scalar_count = min(blocks, SELFTEST_SCALAR_BLOCKS)
    sample = (rng.normal(0, 1, (scalar_count, Q6_K_BLOCK_WEIGHTS)) * 10 ** rng.uniform(-3, 1, (scalar_count, 1))).astype(np.float32)
    named = structured_blocks()
    sample = np.concatenate([sample, np.stack([values for _, values in named])])
    fast = quantize_q6_k(sample)
    slow = np.stack([np.frombuffer(reference_quantize_q6_k_block(row), dtype=np.uint8) for row in sample])
    identical = int((fast == slow).all(axis=1).sum())
    print(f"scalar port vs numpy quantiser: {sample.shape[0]} blocks ({scalar_count} random, {len(named)} structured): "
          f"{identical} identical ({100.0 * identical / sample.shape[0]:.2f} %), {int((fast != slow).sum())} differing bytes", file=out)
    ok &= identical == sample.shape[0]

    ggml, reason = load_ggml_reference()
    random_blocks = (rng.normal(0, 1, (blocks, Q6_K_BLOCK_WEIGHTS)) * 10 ** rng.uniform(-3, 1, (blocks, 1))).astype(np.float32)
    if ggml is None:
        print(f"ggml cross-check: skipped ({reason})", file=out)
    else:
        cases = [("random", random_blocks)] + [(label, values.reshape(1, -1)) for label, values in named]
        total = identical_total = 0
        for label, values in cases:
            same = int((quantize_q6_k(values) == ggml.quantize_q6_k(values)).all(axis=1).sum())
            total += values.shape[0]
            identical_total += same
            if same != values.shape[0]:
                print(f"  ggml differs on {label}: {values.shape[0] - same} of {values.shape[0]} blocks", file=out)
        packed = ggml.quantize_q6_k(random_blocks)
        dequant_difference = float(np.abs(dequantize_q6_k(packed) - ggml.dequantize_q6_k(packed)).max())
        q8_same = int((quantize_q8_0(random_blocks.reshape(-1, Q8_0_BLOCK_WEIGHTS)) == ggml.quantize_q8_0(random_blocks.reshape(-1, Q8_0_BLOCK_WEIGHTS))).all(axis=1).sum())
        q8_total = random_blocks.size // Q8_0_BLOCK_WEIGHTS
        print(f"ggml cross-check ({ggml.path}): Q6_K quantise {identical_total} of {total} blocks byte-identical ({100.0 * identical_total / total:.4f} %); "
              f"Q6_K dequantise max abs difference {dequant_difference:g}; Q8_0 quantise {q8_same} of {q8_total} blocks identical", file=out)
        ok &= identical_total == total and dequant_difference == 0.0

    for label, source, is_checked in (("gaussian", rng.normal(0, 0.02, (blocks, Q6_K_BLOCK_WEIGHTS)), True),
                                      ("laplace", rng.laplace(0, 0.02, (blocks, Q6_K_BLOCK_WEIGHTS)), False)):
        original = source.astype(np.float32)
        q8_values = dequantize_q8_0(quantize_q8_0(original.reshape(-1, Q8_0_BLOCK_WEIGHTS))).reshape(-1, Q6_K_BLOCK_WEIGHTS)
        q6_values = dequantize_q6_k(quantize_q6_k(q8_values))
        added = relative_rms_error(q8_values, q6_values)
        print(f"round trip, {label} weights ({blocks} blocks): Q8_0 vs original {100 * relative_rms_error(original, q8_values):.3f} % rms; "
              f"Q8_0 -> Q6_K vs Q8_0 source {100 * added:.3f} % rms; Q6_K vs original {100 * relative_rms_error(original, q6_values):.3f} % rms"
              f"{'' if is_checked else ' (informational)'}", file=out)
        ok &= added < SELFTEST_MAX_ROUND_TRIP_ERROR or not is_checked

    bench = rng.normal(0, 0.02, (benchmark_blocks, Q6_K_BLOCK_WEIGHTS)).astype(np.float32)
    started = time.perf_counter()
    packed = quantize_q6_k(bench)
    quantise_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dequantize_q6_k(packed)
    dequantise_seconds = time.perf_counter() - started
    print(f"speed on {benchmark_blocks} random blocks, one core: quantise {benchmark_blocks / quantise_seconds:,.0f} blocks/s "
          f"({quantise_seconds:.2f} s), dequantise {benchmark_blocks / dequantise_seconds:,.0f} blocks/s", file=out)
    print("selftest: " + ("ok" if ok else "FAILED"), file=out)
    return ok


def build_parser() -> argparse.ArgumentParser:
    epilog = "groups (a tensor is converted only when it matches a selected group, is Q8_0 and its row length is a multiple of 256):\n" + "".join(
        f"  {name:<6}{group.description}: {group.pattern}\n" for name, group in GROUPS.items()) + (
        f"  {ALL_GROUPS_NAME:<6}every group above\ndefault groups: {','.join(DEFAULT_GROUPS)}\n"
        "routed experts (ffn_*_exps), norms, the router and the state-space parameters are never touched.")
    parser = argparse.ArgumentParser(
        description="Copy a GGUF file with the selected dense Q8_0 tensors re-quantised to Q6_K (about 23 % fewer bytes for those tensors).",
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", metavar="SRC", nargs="?", type=Path)
    parser.add_argument("dest", metavar="DST", nargs="?", type=Path, help="must not exist; written as DST.part and renamed when checked")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="comma-separated group names or 'all' (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, the new size and the disk check; write nothing")
    parser.add_argument("--chunk-mib", type=float, default=DEFAULT_CHUNK_MIB, help="float32 working-set size per chunk of rows (default: %(default)s)")
    parser.add_argument("--selftest", action="store_true", help="check the quantiser against the scalar port, ggml and a round trip, and time it")
    parser.add_argument("--blocks", type=int, default=SELFTEST_DEFAULT_BLOCKS, help="random blocks used by --selftest (default: %(default)s)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return 0 if selftest(args.blocks) else 1
    if args.source is None or args.dest is None:
        parser.error("SRC and DST are required")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    try:
        groups = parse_groups(args.groups)
    except ValueError as error:
        parser.error(str(error))

    def progress(done: int, total: int, written: int, expected: int) -> None:
        print(f"requant_dense: {done}/{total} tensors, {written / GIB:.2f} of {expected / GIB:.2f} GiB", file=sys.stderr, flush=True)

    try:
        if args.dry_run:
            if args.dest.exists():
                raise FileExistsError(f"{args.dest} already exists; remove it first")
            return 0 if report_plan(build_plan(args.source, groups), args.dest, sys.stdout) else 1
        started = time.monotonic()
        plan = convert(args.source, args.dest, groups, int(args.chunk_mib * (1 << 20)), progress)
    except (OSError, ValueError) as error:
        print(f"requant_dense: error: {error}", file=sys.stderr)
        return 1
    saved = plan.layout.size - plan.total_size
    converted = sum(1 for entry in plan.entries if entry.convert)
    print(f"requant_dense: wrote {args.dest}: {plan.total_size} bytes ({plan.total_size / GIB:.3f} GiB), "
          f"{converted} tensors re-quantised, saved {saved} bytes ({saved / GIB:.3f} GiB, {100.0 * saved / plan.layout.size:.1f} % of the source), "
          f"{time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
