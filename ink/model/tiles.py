"""A stack's tiles and the arithmetic both lanes' readers share: the statistics a stack is normalized by, its empty
columns filled, 512-pixel tiles every 384 pixels, and their maps blended by a Hann window."""

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from runner.compiled_kernels import load

TILE, STEP = 512, 384
DEPTH, STEM_CHANNELS = 17, 16
FILLING_TRIES = 400
GRID = 32
MEASURED_SIDE, MEASURED_TILES = 128, 8
CANDIDATES = 64
EVERY_NTH_VALUE = 97


def measured_tiles(stack):
    """The windows of a stack most covered by sheet out of a fixed random draw."""
    _, height, width = stack.shape
    side_y, side_x = min(MEASURED_SIDE, height), min(MEASURED_SIDE, width)
    middle = stack[stack.shape[0] // 2]
    if middle.dtype != np.uint8:
        middle = middle > 0
    rng = np.random.default_rng(height * 31 + width)
    spots = list(zip(rng.integers(0, height - side_y + 1, CANDIDATES), rng.integers(0, width - side_x + 1, CANDIDATES)))
    spots.sort(key=lambda s: -np.count_nonzero(middle[s[0] : s[0] + side_y, s[1] : s[1] + side_x]))
    return [np.ascontiguousarray(stack[:, y : y + side_y, x : x + side_x]) for y, x in spots[:MEASURED_TILES]]


def raw_scale(tiles):
    """Clip limits, centre and spread of raw values, the published robust normalization taken once for a stack."""
    values = np.concatenate([t[t > 0][::EVERY_NTH_VALUE] for t in tiles]).astype(np.float32)
    if not values.size:
        return 0.0, 255.0, 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    values = np.clip(values, low, high)
    centre = np.median(values)
    return float(low), float(high), float(centre), max(float(1.4826 * np.median(np.abs(values - centre))), 1.0)


def measured(stack):
    """What a stack's statistics are taken from."""
    tiles = measured_tiles(stack)
    return np.stack(tiles), raw_scale(tiles)


def scaled(raw, scale):
    low, high, centre, spread = scale
    return (raw.half().clamp_(low, high) - centre) / spread


@cache
def fill_calls():
    calls = load("tile-filling", [Path(__file__).with_name("tile_filling.c")])
    ptr, i32, i64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
    calls.hx_fill_pending.argtypes = [ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i64]
    calls.hx_fill_pending.restype = i64
    calls.hx_fill_valid.argtypes = [ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i64]
    calls.hx_fill_valid.restype = i64
    calls.hx_gather_filled.argtypes = [ptr, ptr, i64, i64, ptr]
    calls.hx_gather_filled.restype = None
    return calls


def filled(stack, keep=None):
    """A stack's empty columns taken from shifted, flipped copies of itself."""
    keep = np.ascontiguousarray(stack.any(axis=0) if keep is None else keep)
    valid = np.count_nonzero(keep)
    remaining = keep.size - valid
    if not valid or not remaining:
        return stack
    height, width = keep.shape
    rng = np.random.default_rng(height * 31 + width)
    source = np.arange(height * width, dtype=np.int32)
    sparse = valid * 4 < keep.size
    columns = np.flatnonzero(keep if sparse else ~keep).astype(np.int32)
    rows, cols = columns // width, columns % width
    if sparse:
        source[~keep.ravel()] = -1
    else:
        source[columns] = 0
    row_ptr, col_ptr = rows.ctypes.data, cols.ctypes.data
    source_ptr, keep_ptr = source.ctypes.data, keep.ctypes.data
    calls = fill_calls()
    for _ in range(FILLING_TRIES):
        dy, dx = int(rng.integers(0, height)), int(rng.integers(0, width))
        fy, fx = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        if sparse:
            remaining -= calls.hx_fill_valid(source_ptr, row_ptr, col_ptr, height, width, dy, dx, fy, fx, valid)
        else:
            remaining = calls.hx_fill_pending(
                source_ptr, keep_ptr, row_ptr, col_ptr, height, width, dy, dx, fy, fx, remaining
            )
        if not remaining:
            break
    if sparse:
        np.maximum(source, 0, out=source)
    if stack.dtype == np.uint8 and keep.dtype == np.bool_ and keep.shape == stack.shape[1:] and source.size < 2**31:
        raw = np.ascontiguousarray(stack)
        result = np.empty_like(raw)
        calls.hx_gather_filled(raw.ctypes.data, source_ptr, raw.shape[0], source.size, result.ctypes.data)
        return result
    return np.take(stack.reshape(stack.shape[0], -1), source, axis=1).reshape(stack.shape)


class PreparedStack(NamedTuple):
    values: np.ndarray
    coverage: np.ndarray

    @property
    def shape(self):
        return self.values.shape


def prepare_stack(stack):
    """Fill model context while retaining the coverage of the original stack."""
    coverage = stack.any(axis=0)
    return PreparedStack(filled(stack, coverage), coverage)


def tile_starts(length, tile):
    last = max(length - tile, 0)
    return sorted({*range(0, last + 1, STEP), last})


class Sums:
    """A map adding up its tiles."""

    def __init__(self, height, width):
        self.tile_y, self.tile_x = min(TILE, -(-height // GRID) * GRID), min(TILE, -(-width // GRID) * GRID)
        self.blend = np.outer(np.hanning(self.tile_y + 2)[1:-1], np.hanning(self.tile_x + 2)[1:-1]).astype(np.float32)
        self.total, self.weight = np.zeros((2, height, width), dtype=np.float32)
        self.failed = None

    def add(self, y, x, h, w, read):
        self.total[y : y + h, x : x + w] += read[:h, :w] * self.blend[:h, :w]
        self.weight[y : y + h, x : x + w] += self.blend[:h, :w]

    def result(self):
        return self.failed or self.total / np.maximum(self.weight, 1e-6)


def fixed_norms(torch, stem):
    """The stem's instance norms swapped for ones holding statistics measured on the last pass marked measuring."""

    class Fixed(torch.nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.norm, self.measuring = norm, False

        def scale_and_shift(self):
            scale = self.norm.weight.float() * torch.rsqrt(self.variance + self.norm.eps)
            return scale, self.norm.bias.float() - self.mean * scale

        def measure(self, x):
            values = x.float()
            self.mean, self.variance = values.mean([0, 2, 3, 4]), values.var([0, 2, 3, 4], unbiased=False)

        def forward(self, x):
            if self.measuring:
                self.measure(x)
            scale, shift = self.scale_and_shift()
            return x * scale.reshape(1, -1, 1, 1, 1).to(x.dtype) + shift.reshape(1, -1, 1, 1, 1).to(x.dtype)

    found = []
    for at in (1, 4):
        found.append(Fixed(stem.features[at]))
        stem.features[at] = found[-1]
    return found
