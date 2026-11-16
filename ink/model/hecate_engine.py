"""Hecate 9.6 µm (scrollprize/hecate, MIT) on the Neural Engine: the released network as 2D convolutions, half precision.

On MPS the reference hecate.py takes about 400 ms a 16 x 64 x 64 patch and our MLX port (hecate_mlx) 265 ms; the
network is about 211 G multiply-adds a patch. Here it runs as one Core ML program on the Neural Engine at about 87 ms
a patch, every operation on the engine, called from C (engine_calls.m) so several patches are in flight at once.

The network is rewritten, not changed. Depth is folded into the batch: a patch is 16 images of 64 x 64, and each
3 x 3 x 3 convolution is one 2D convolution giving its three depth taps as three blocks of channels, the blocks
shifted one image up and down within the patch and summed. Strided convolutions keep every other image; the decoder's
trilinear upsampling is a linear blend of neighbouring images (edges clamped) and a bilinear one in plane. Batch norms
are folded into the convolutions before them. Two rewrites keep half precision safe and Core ML correct:
  - each group norm's input is divided by a power of two near its typical size (GN_SCALES, the RMS measured on real
    CT), so the activations and their squares stay well inside half precision (peaks are 10 to 30 times the RMS);
    up to 16 of it is folded into the bias-free convolution before, keeping the weights (median 0.1) clear of half
    precision's subnormals, the rest is an exact multiply; the norm is unchanged, its epsilon divided by the square;
  - each group norm takes spatial means first, then reduces the small (depth, group, channel) table one axis at a
    time: Core ML's CPU and Neural Engine get a single reduction over two apart axes wrong (the GPU does not).
In float32 the rewrite matches hecate.py to 4e-5 on logits of range 12; in half precision on the engine, probabilities
correlate 0.9997 with float32, mean difference 0.003.

../../runtime/python/bin/python -m ink.model.hecate_engine CHECKPOINT      builds, then checks against hecate.py on CPU
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ink.model.hecate_mlx import DEC, EPS, LAYERS, PATCH, Weights
from ink.model.neural_engine import CPU_AND_ENGINE, engine_calls
from ink.model.stock_model import checkpoint_hash
from storage.data_roots import NEURAL_ENGINE_BUILDS

DEPTH, SIDE = PATCH[0], PATCH[1]
# 2 ** round(log2(RMS)) of each group norm's input, median over 8 patches of PHerc0211 (peaks 9 to 27 x RMS)
GN_SCALES = {
    "canonical.decoder.channel_reduce.0.1": 8.0,
    "canonical.decoder.channel_reduce.1.1": 16.0,
    "canonical.decoder.channel_reduce.2.1": 512.0,
    "canonical.decoder.channel_reduce.3.1": 256.0,
    "canonical.decoder.decoder_blocks.0.gn1": 32.0,
    "canonical.decoder.decoder_blocks.0.gn2": 128.0,
    "canonical.decoder.decoder_blocks.1.gn1": 32.0,
    "canonical.decoder.decoder_blocks.1.gn2": 16.0,
    "canonical.decoder.decoder_blocks.2.gn1": 32.0,
    "canonical.decoder.decoder_blocks.2.gn2": 16.0,
}
IN_WEIGHTS = 16.0  # the most of a scale folded into a convolution's weights
VERSION = "v2"  # of the rewrite: a new one builds a new program


def flat_network(checkpoint, scales=GN_SCALES):
    """The network as a torch module of 2D operations: (1, 16, 64, 64) raw CT values 0..255 to (1, 64, 64) logits."""
    import torch
    from torch.nn import functional

    w = Weights(checkpoint)

    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    def folded(prefix, bn=None, bias=None, norm=None):
        k = w.s[f"{prefix}.weight"]
        k = k[:, :, None] if k.ndim == 4 else k
        b = w.s[bias] if bias else np.zeros(k.shape[0], np.float32)
        if bn:
            scale, shift = w.bn(bn)
            k, b = k * scale[:, None, None, None, None], b * scale + shift
        if norm:  # bias-free before a group norm: divided by up to 16 of the norm's power of two
            assert not b.any()
            k = k / min(scales.get(norm, 1.0), IN_WEIGHTS)
        return k, b

    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            scale, shift = w.bn("canonical.normalization")
            self.input = (float(scale[0]) / 255.0, float(shift[0]))
            self.weights = torch.nn.ParameterList()
            self.spec, self.norms, self.blocks = {}, {}, []
            p, d = "canonical", "canonical.decoder"
            self.keep("stem", *folded(f"{p}.backbone.conv1", f"{p}.backbone.bn1"))
            for li, n in enumerate(LAYERS, start=1):
                for bi in range(n):
                    q = f"{p}.backbone.layer{li}.{bi}"
                    for c in ("conv1", "conv2", "conv3"):
                        self.keep(f"{q}.{c}", *folded(f"{q}.{c}", f"{q}.bn{c[-1]}"))
                    down = f"{q}.downsample.0.weight" in w.s
                    if down:
                        self.keep(f"{q}.down", *folded(f"{q}.downsample.0", f"{q}.downsample.1"))
                    self.blocks.append((li, q, 2 if (bi == 0 and li > 1) else 1, down))
            for i in range(4):
                norm = f"{d}.channel_reduce.{i}.1"
                self.keep(f"reduce{i}", *folded(f"{d}.channel_reduce.{i}.0", norm=norm))
                self.keep_norm(norm)
            for i in range(3):
                q = f"{d}.decoder_blocks.{i}"
                self.keep(f"{q}.conv1", *folded(f"{q}.conv1", norm=f"{q}.gn1"))
                self.keep(f"{q}.conv2", *folded(f"{q}.conv2", norm=f"{q}.gn2"))
                if f"{q}.shortcut.weight" in w.s:
                    self.keep(f"{q}.shortcut", *folded(f"{q}.shortcut"))
                self.keep_norm(f"{q}.gn1")
                self.keep_norm(f"{q}.gn2")
            self.keep("logit", *folded(f"{d}.logit", bias=f"{d}.logit.bias"))
            self.keep("attn", *folded(f"{d}.depth_collapse.attn_conv", bias=f"{d}.depth_collapse.attn_conv.bias"))
            for depth in (16, 8, 4, 2):
                z = np.arange(depth)
                self.register_buffer(f"prev{depth}", tensor(z != 0).view(-1, 1, 1, 1))
                self.register_buffer(f"next{depth}", tensor(z != depth - 1).view(-1, 1, 1, 1))
            z = (np.arange(DEPTH) - DEPTH // 2) / (DEPTH // 2)
            self.register_buffer("depth_bias", tensor(w.depth_scale * z).view(1, DEPTH, 1, 1))

        def keep(self, name, k, b):
            taps = k.shape[2]
            k2 = np.concatenate([k[:, :, i] for i in range(3)], 0) if taps == 3 else k[:, :, 0]
            self.spec[name] = (len(self.weights), taps, k.shape[0])
            self.weights.append(torch.nn.Parameter(tensor(k2), requires_grad=False))
            self.weights.append(torch.nn.Parameter(tensor(b).view(1, -1, 1, 1), requires_grad=False))

        def keep_norm(self, name):
            scale = scales.get(name, 1.0)
            rest = 1.0 / (scale / min(scale, IN_WEIGHTS))
            self.norms[name] = (len(self.weights), int(w.s[f"{name}.weight"].shape[0]), EPS / scale**2, rest)
            for part in ("weight", "bias"):
                self.weights.append(
                    torch.nn.Parameter(tensor(w.s[f"{name}.{part}"]).view(1, -1, 1, 1), requires_grad=False)
                )

        def shifted(self, y, depth):
            """y[z - 1] and y[z + 1], zero past the patch's ends."""
            zero = torch.zeros_like(y[:1])
            before = torch.cat([zero, y[:-1]], 0) * getattr(self, f"prev{depth}")
            after = torch.cat([y[1:], zero], 0) * getattr(self, f"next{depth}")
            return before, after

        def every_other(self, y, depth, channels):
            """The even images of y, its plane already halved to 2 * depth by a strided conv."""
            side = 2 * depth
            return y.reshape(depth // 2, 2, channels, side * side)[:, 0].reshape(depth // 2, channels, side, side)

        def conv(self, name, x, depth, stride=1):
            i, taps, out = self.spec[name]
            weight, bias = self.weights[i], self.weights[i + 1]
            if taps == 1:
                y = functional.conv2d(x, weight, stride=stride)
            else:
                y = functional.conv2d(x, weight, stride=stride, padding=1)
                before, _ = self.shifted(y[:, :out], depth)
                _, after = self.shifted(y[:, 2 * out :], depth)
                y = before + y[:, out : 2 * out] + after
            return (self.every_other(y, depth, out) if stride == 2 else y) + bias

        def group_norm(self, name, x, depth, groups):
            i, channels, eps, rest = self.norms[name]
            per = channels // groups
            x = x * rest if rest != 1.0 else x

            def over_group(v):  # (depth, channels, 1, 1) spatial means to the patch's group means, per channel
                v = v.reshape(1, depth, groups, per).mean(3, keepdim=True).mean(1, keepdim=True)
                return v.expand(1, depth, groups, per).reshape(depth, channels, 1, 1)

            centred = x - over_group(x.mean((2, 3), keepdim=True))
            var = over_group((centred * centred).mean((2, 3), keepdim=True))
            return centred * (torch.rsqrt(var + eps) * self.weights[i]) + self.weights[i + 1]

        def up(self, x, depth, channels):
            """x2 trilinear, align_corners False: depth by hand (edges clamped), then bilinear in plane."""
            before, after = self.shifted(x, depth)
            before = before + (1 - getattr(self, f"prev{depth}")) * x
            after = after + (1 - getattr(self, f"next{depth}")) * x
            even, odd = 0.75 * x + 0.25 * before, 0.75 * x + 0.25 * after
            side = 4 * depth
            pair = [v.reshape(depth, 1, channels, side * side) for v in (even, odd)]
            x = torch.cat(pair, 1).reshape(2 * depth, channels, side, side)
            return functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, ct):
            x = ct.reshape(DEPTH, 1, SIDE, SIDE) * self.input[0] + self.input[1]
            depth = DEPTH
            x = functional.relu(self.conv("stem", x, depth))
            feats, current = [], 1
            for li, q, stride, down in self.blocks:
                if li != current:
                    feats.append(x)
                    current = li
                residual = self.conv(f"{q}.down", x, depth, stride) if down else x
                out = functional.relu(self.conv(f"{q}.conv1", x, depth))
                out = functional.relu(self.conv(f"{q}.conv2", out, depth, stride))
                depth //= stride
                x = functional.relu(self.conv(f"{q}.conv3", out, depth) + residual)
            feats.append(x)
            d = "canonical.decoder"
            reduced = []
            for i, f in enumerate(feats):
                level = DEPTH >> i
                norm = f"{d}.channel_reduce.{i}.1"
                reduced.append(
                    functional.relu(self.group_norm(norm, self.conv(f"reduce{i}", f, level), level, min(32, DEC[i])))
                )
            x, depth = reduced[3], DEPTH >> 3
            for k, skip in enumerate((reduced[2], reduced[1], reduced[0])):
                q, channels = f"{d}.decoder_blocks.{k}", DEC[2 - k]
                x = torch.cat([self.up(x, depth, DEC[3 - k]), skip], 1)
                depth *= 2
                identity = self.conv(f"{q}.shortcut", x, depth) if f"{q}.shortcut" in self.spec else x
                out = functional.relu(
                    self.group_norm(f"{q}.gn1", self.conv(f"{q}.conv1", x, depth), depth, min(32, channels))
                )
                out = self.group_norm(f"{q}.gn2", self.conv(f"{q}.conv2", out, depth), depth, min(32, channels))
                x = functional.relu(out + identity)
            logits = self.conv("logit", x, depth).reshape(1, DEPTH, SIDE, SIDE)
            scores = self.conv("attn", x, depth).reshape(1, DEPTH, SIDE, SIDE) + self.depth_bias
            return (torch.softmax(scores, 1) * logits).sum(1)

    return Flat().eval()


def package_of(checkpoint):
    return (
        NEURAL_ENGINE_BUILDS
        / f"hecate-{checkpoint_hash(checkpoint).hexdigest()[:16]}-{DEPTH}x{SIDE}-{VERSION}.mlpackage"
    )


def compile_network(checkpoint, path):
    """The flat network as a half-precision Core ML program for the Neural Engine, written whole so two may race."""
    import coremltools as ct
    import torch

    example = torch.zeros(1, DEPTH, SIDE, SIDE)
    with torch.inference_mode():
        traced = torch.jit.trace(flat_network(checkpoint), example, strict=False)
    program = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="ink", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}-{os.getpid()}.part.mlpackage")
    program.save(str(part))
    if path.exists():
        shutil.rmtree(part)
    else:
        part.rename(path)


class HecateEngine:
    """Hecate's logits of single patches on the Neural Engine, and hecate.predict's tiled map."""

    def __init__(self, checkpoint, threads=3):
        path = package_of(checkpoint)
        if not path.exists():
            compile_network(checkpoint, path)
        self.calls = engine_calls()
        self.engine = self.calls.hx_engine_open(
            str(path).encode(), str(path.with_suffix(".mlmodelc")).encode(), CPU_AND_ENGINE, True
        )
        if not self.engine:
            raise RuntimeError(f"no engine from {path}")
        self.threads = threads

    def logits(self, patch):
        """Attention-weighted logits (64, 64) float32 of one (16, 64, 64) uint8 patch."""
        image = np.ascontiguousarray(patch, dtype=np.float16)
        out = np.empty((SIDE, SIDE), np.float16)
        failed = self.calls.hx_engine_read(self.engine, image.ctypes.data, DEPTH, SIDE, out.ctypes.data)
        if failed:
            raise RuntimeError(f"engine call failed: {failed}")
        out = out.astype(np.float32)
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite Hecate logits")
        return out

    def predict(self, volume, stride=32):
        """hecate.predict's 2D map of a (z, y, x) uint8 render at 9.6 µm, central 16 planes, as float probabilities."""
        depth, h, w = volume.shape
        planes = volume[depth // 2 - DEPTH // 2 : depth // 2 - DEPTH // 2 + DEPTH]

        def starts(n):
            return [*range(0, max(0, n - SIDE), stride), max(0, n - SIDE)]

        win = np.hanning(SIDE)  # torch.hann_window(p, periodic=False)
        win = np.outer(win, win)
        win = np.maximum(win / win.max(), 0.001).astype(np.float32)
        numerator, denominator = np.zeros((2, h, w), np.float32)
        active = []
        for y in starts(h):
            for x in starts(w):
                ph, pw = min(SIDE, h - y), min(SIDE, w - x)
                denominator[y : y + ph, x : x + pw] += win[:ph, :pw]
                if planes[:, y : y + ph, x : x + pw].any():
                    active.append((y, x, ph, pw))

        def read(origin):
            y, x, ph, pw = origin
            patch = np.zeros((DEPTH, SIDE, SIDE), np.uint8)
            patch[:, :ph, :pw] = planes[:, y : y + ph, x : x + pw]
            support = patch.any(axis=0)
            return 1 / (1 + np.exp(-self.logits(patch))) * support

        with ThreadPoolExecutor(self.threads) as pool:
            for (y, x, ph, pw), prob in zip(active, pool.map(read, active)):
                numerator[y : y + ph, x : x + pw] += (prob * win)[:ph, :pw]
        return np.clip(numerator / denominator, 0, 1)


if __name__ == "__main__":
    import importlib.util
    import sys
    import time

    import torch

    checkpoint = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("hecate", str(checkpoint.with_name("hecate.py")))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ref = reference.load_model(str(checkpoint), "cpu")
    rng = np.random.default_rng(0)
    raw = rng.integers(60, 200, size=(2, DEPTH, SIDE, SIDE)).astype(np.uint8)
    with torch.inference_mode():
        expect = ref(
            torch.from_numpy(raw.astype(np.float32) / 255.0)[:, None],
            torch.ones(2, 1, DEPTH, SIDE, SIDE, dtype=torch.bool),
        )
        expect = torch.sigmoid(expect).numpy()[:, 0]
        flat = flat_network(checkpoint)
        rewritten = np.stack(
            [torch.sigmoid(flat(torch.from_numpy(r.astype(np.float32))[None])).numpy()[0] for r in raw]
        )
    print("rewrite in float32 vs hecate.py: largest probability difference", float(np.abs(rewritten - expect).max()))
    began = time.time()
    engine = HecateEngine(checkpoint)
    print("engine ready", round(time.time() - began, 1), "s")
    got = np.stack([1 / (1 + np.exp(-engine.logits(r))) for r in raw])
    print(
        "engine vs hecate.py: largest probability difference",
        float(np.abs(got - expect).max()),
        "mean",
        float(np.abs(got - expect).mean()),
    )
