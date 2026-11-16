"""The taught ink network on 512-pixel tiles: Metal depth stem, Neural Engine body."""

import collections
import ctypes
import queue
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache, cached_property
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ink.model import stock_model
from ink.model.neural_engine import SIDE, STEM_CHANNELS, EngineBody
from runner.compiled_kernels import load

TILE, STEP = 512, 384
FILLING_TRIES = 400
GRID = 32
MEASURED_SIDE, MEASURED_TILES = 128, 8
CANDIDATES = 64
EVERY_NTH_VALUE = 97
TILES_AHEAD = 8
ENGINE_CALLS = 2
PAIR_GAP = 280


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


class DepthStem:
    """The network's depth stem as one Metal kernel, its norms fixed to the statistics of a stack."""

    def __init__(self, module, device="mps"):
        import torch

        self.torch, self.device, self.module = torch, device, module.to(device).half()
        features = module.features
        if isinstance(features[0], torch.nn.Conv3d):
            features[0], features[3] = stock_model.folded(features[0]), stock_model.folded(features[3])
            module.to(device).half()
        self.norms = fixed_norms(torch, module)
        self.source = (Path(__file__).parent / "depth_stem.metal").read_text()
        self.kernel = torch.mps.compile_shader(self.source)
        self.attention_bias = float(module.attention_logits.bias.float())
        self.scale, self.table = None, None

    @cached_property
    def raw_kernel(self):
        return self.torch.mps.compile_shader("#define RAW_INPUT\n" + self.source)

    @cached_property
    def edge_kernel(self):
        return self.torch.mps.compile_shader("#define RAW_INPUT\n#define TILE_EDGE\n" + self.source)

    def statistics(self, stack):
        return self.statistics_of(*measured(stack))

    def statistics_of(self, tiles, scale):
        """Raw scale and folded stem weights of a stack, as a patch's own norms would take them but the same everywhere on it."""
        torch = self.torch
        with torch.inference_mode():
            features = self.module.features
            image = scaled(torch.from_numpy(tiles).to(self.device), scale)[:, None]
            first = features[0](image)
            self.norms[0].measure(first)
            second = features[3](features[2](self.norms[0](first)))
            self.norms[1].measure(second)
        with torch.no_grad():
            return scale, stem_of(self.module)

    def __call__(self, image, statistics):
        return self.features(self.kernel, image, statistics)

    def features(self, kernel, image, statistics, *extra):
        batch, _, height, width = image.shape
        out = self.torch.empty(batch, 32, height, width, device=self.device, dtype=self.torch.float16)
        sizes = self.torch.tensor(
            np.frombuffer(struct.pack("IIIf", batch, height, width, self.attention_bias), dtype=np.int32).copy(),
            device=self.device,
        )
        kernel.depth_stem(
            image,
            *statistics[1],
            out,
            sizes,
            *extra,
            threads=[(width + 15) // 16 * 128, (height + 7) // 8, batch],
            group_size=[128, 1, 1],
        )
        return out

    def tiles(self, block, statistics, gap):
        """Stem features for one tile or two sharing their convolution interior."""
        torch = self.torch
        if self.scale != statistics[0]:
            self.scale = statistics[0]
            self.table = scaled(torch.arange(256, dtype=torch.int32, device=self.device), self.scale)
        image = torch.from_numpy(np.ascontiguousarray(block, dtype=np.uint8)).to(self.device)[None]
        features = self.features(self.raw_kernel, image, statistics, self.table)
        if not gap:
            yield features
            return
        for offset in (0, gap):
            tile = features[:, :, :, offset : offset + TILE].contiguous()
            dims = struct.pack("IIIfIII", 1, TILE, TILE, self.attention_bias, TILE + gap, offset, int(offset == 0))
            sizes = torch.tensor(np.frombuffer(dims, dtype=np.int32).copy(), device=self.device)
            self.edge_kernel.depth_stem(
                image, *statistics[1], tile, sizes, self.table, threads=[128, TILE // 8, 1], group_size=[128, 1, 1]
            )
            yield tile


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


def tile_groups(starts, pair):
    """Keep blend order while pairing close tiles, starting with the last (often closest) pair."""
    if not pair:
        return [(x,) for x in starts]
    starts = starts.copy()
    groups = []
    while starts:
        x = starts.pop()
        groups.append((starts.pop(), x) if starts and x - starts[-1] <= PAIR_GAP else (x,))
    return groups[::-1]


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


class TileReader:
    def __init__(self, weights, device="mps"):
        import torch

        self.torch, self.device, self.weights = torch, device, weights
        self.stem = DepthStem(stock_model.taught_stem(weights, STEM_CHANNELS), device)
        self.engine = EngineBody(weights, load_body=lambda: self.body)

    @cached_property
    def body(self):
        """The Torch body for uncompiled checkpoints and tiles smaller than the engine model."""
        return (
            stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[self.weights], device=self.device)
            .models[0]
            .model.model.requires_grad_(False)
            .network
        )

    def read(self, stack, fixed=False):
        """Probability of ink of a (depth, height, width) uint8 stack, tiles blended by a Hann window."""
        (read,) = self.maps([(stack, None if fixed else measured(stack))])
        if isinstance(read, Exception):
            raise read
        return read

    def maps(self, stacks):
        """Read each raw or prepared stack with its measurements, or None to reuse the last statistics."""
        tiles, stop, pending = queue.Queue(TILES_AHEAD), threading.Event(), collections.deque()
        with ThreadPoolExecutor(1) as stem_thread, ThreadPoolExecutor(ENGINE_CALLS) as engine:
            stems = stem_thread.submit(self.stems, stacks, tiles, stop)
            try:
                for sums, place, features, on_engine in iter(tiles.get, None):
                    pending.append((sums, place, engine.submit(self.ink_of, features) if on_engine else features))
                    while pending and (len(pending) > ENGINE_CALLS or pending[0][1] is None):
                        yield from self.settled(pending)
                while pending:
                    yield from self.settled(pending)
            finally:
                stop.set()
            stems.result()

    def ink_of(self, features):
        return 1 / (1 + np.exp(-self.engine(features)))

    def settled(self, pending):
        """The oldest tile added to its map, always in tile order, or the map once its stack has no tile left."""
        sums, place, read = pending.popleft()
        if place is None:
            yield sums.result()
        elif sums.failed is None:
            try:
                sums.add(*place, read.result() if isinstance(read, Future) else read)
            except Exception as error:
                sums.failed = error

    def stems(self, stacks, tiles, stop):
        """Every tile of every stack through the stem, in order, for the Neural Engine."""
        torch = self.torch

        def send(item):
            while not stop.is_set():
                try:
                    return tiles.put(item, timeout=1)
                except queue.Full:
                    continue
            return None

        try:
            with torch.inference_mode():
                for stack, measure in stacks:
                    _, height, width = stack.shape
                    sums = Sums(height, width)
                    try:
                        if measure is not None:
                            self.fixed = self.stem.statistics_of(*measure)
                        ty, tx = sums.tile_y, sums.tile_x
                        whole, coverage = stack if isinstance(stack, PreparedStack) else prepare_stack(stack)
                        on_engine = (ty, tx) == (SIDE, SIDE)
                        pair = on_engine and height >= ty
                        xs = tile_starts(width, tx)
                        for y in tile_starts(height, ty):
                            active = [x for x in xs if coverage[y : y + ty, x : x + tx].any()]
                            for group in tile_groups(active, pair):
                                x, last = group[0], group[-1]
                                block = whole[:, y : y + ty, x : last + tx]
                                h, w = block.shape[1:]
                                if h < ty or w < tx:
                                    block = np.pad(block, ((0, 0), (0, ty - h), (0, tx - w)))
                                for at, features in zip(group, self.stem.tiles(block, self.fixed, last - x)):
                                    place = (y, at, h, min(w, tx))
                                    if on_engine:
                                        send((sums, place, features.cpu().numpy(), True))
                                    else:
                                        read = torch.sigmoid(self.body(features)["ink"].float())[0, 0].cpu().numpy()
                                        send((sums, place, read, False))
                    except Exception as error:
                        sums.failed = error
                    send((sums, None, None, None))
        finally:
            send(None)

    def fix_to(self, stack):
        self.fixed = self.stem.statistics(stack)


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


def stem_of(stem):
    """The stem's weights with its norms folded in, as the Metal kernel reads them."""
    first, second = stem.features[1].scale_and_shift(), stem.features[4].scale_and_shift()
    w1 = stem.features[0].weight.float().reshape(16, 3, 1, 3, 3)[:, :, 0] * first[0].reshape(-1, 1, 1, 1)
    w2 = stem.features[3].weight.float().reshape(16, 3, 16, 3, 3) * second[0].reshape(-1, 1, 1, 1, 1)
    return (
        w1.permute(1, 2, 3, 0).reshape(27, 16).contiguous().half(),
        first[1].half(),
        w2.permute(1, 3, 4, 2, 0).reshape(27, 16, 16).contiguous().half(),
        second[1].half(),
        stem.attention_logits.weight.float().reshape(16).half(),
    )
