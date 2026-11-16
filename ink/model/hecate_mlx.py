"""Hecate 9.6 µm (scrollprize/hecate, MIT) on MLX: the released network's forward and its tiled predict, made fast.

hecate.py (sha256 c232c18a..., read before use) runs a 3D ResNet-152-style encoder with no stem downsampling, a 3D
decoder with group norms and trilinear skips, and a learned attention that collapses depth into one map. On MPS it
takes about 400 ms a 16 x 64 x 64 patch. Here the same network runs on MLX in float32 with channels last: every batch
norm (the input's and the encoder's) folded into the convolution before it, nothing else changed. predict() repeats
hecate.predict: patches every 32 px of the 16 central planes, the empty ones skipped but still counted in the Hann
weights, the probability as the sigmoid of the attention-weighted logits, the sum normalised by the weights.

../../runtime/python/bin/python -m ink.model.hecate_mlx CHECKPOINT      checks the forward against hecate.py on CPU
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

EPS = 1e-5
LAYERS = (3, 8, 36, 3)
PLANES = (64, 128, 256, 512)
DEC = (64, 128, 256, 512)
PATCH = (16, 64, 64)


def _np(t):
    return t.detach().float().cpu().numpy()


class Weights:
    """The checkpoint's EMA weights as MLX arrays, convolutions channels last with their batch norms folded in."""

    def __init__(self, checkpoint):
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("architecture_family") != "hecate" or payload.get("sampling_um") != 9.6:
            raise ValueError("expects the released Hecate 9.6 µm checkpoint")
        self.s = {k: _np(v) for k, v in payload["ema_model"].items()}
        self.depth_scale = float(self.s["depth_coordinate_scale"])

    def bn(self, prefix):
        s = self.s
        scale = s[f"{prefix}.weight"] / np.sqrt(s[f"{prefix}.running_var"] + EPS)
        return scale, s[f"{prefix}.bias"] - s[f"{prefix}.running_mean"] * scale

    def conv(self, prefix, bn=None, bias=None):
        """(weight O,kD,kH,kW,I, bias O) of a conv, with a batch norm after it folded in."""
        w = self.s[f"{prefix}.weight"]
        if w.ndim == 4:  # a 2D 1x1 head used as a 3D 1x1x1 conv
            w = w[:, :, None]
        b = self.s[bias] if bias else np.zeros(w.shape[0], np.float32)
        if bn:
            scale, shift = self.bn(bn)
            w = w * scale[:, None, None, None, None]
            b = b * scale + shift
        return mx.array(np.ascontiguousarray(w.transpose(0, 2, 3, 4, 1))), mx.array(b.astype(np.float32))


def conv(x, wb, stride=1):
    w, b = wb
    k = w.shape[1]
    return mx.conv3d(x, w, stride=stride, padding=k // 2) + b


def group_norm(x, prefix, w, groups):
    s = w.s
    c = x.shape[-1]
    n, d, h, wd, _ = x.shape
    g = x.reshape(n, d * h * wd, groups, c // groups)
    mean = g.mean(axis=(1, 3), keepdims=True)
    var = ((g - mean) ** 2).mean(axis=(1, 3), keepdims=True)
    g = (g - mean) * mx.rsqrt(var + EPS)
    return g.reshape(x.shape) * mx.array(s[f"{prefix}.weight"]) + mx.array(s[f"{prefix}.bias"])


class Hecate:
    def __init__(self, checkpoint):
        w = self.w = Weights(checkpoint)
        p = "canonical"
        scale, shift = w.bn(f"{p}.normalization")
        self.input = (float(scale[0]), float(shift[0]))
        self.stem = w.conv(f"{p}.backbone.conv1", f"{p}.backbone.bn1")
        self.blocks = []
        for li, n in enumerate(LAYERS, start=1):
            for bi in range(n):
                q = f"{p}.backbone.layer{li}.{bi}"
                stride = 2 if (bi == 0 and li > 1) else 1
                down = w.conv(f"{q}.downsample.0", f"{q}.downsample.1") if f"{q}.downsample.0.weight" in w.s else None
                self.blocks.append(
                    (li, stride, w.conv(f"{q}.conv1", f"{q}.bn1"), w.conv(f"{q}.conv2", f"{q}.bn2"), w.conv(f"{q}.conv3", f"{q}.bn3"), down)
                )  # fmt: skip
        d = f"{p}.decoder"
        self.reduce = [w.conv(f"{d}.channel_reduce.{i}.0") for i in range(4)]
        self.dec = []
        for i in range(3):
            q = f"{d}.decoder_blocks.{i}"
            short = w.conv(f"{q}.shortcut") if f"{q}.shortcut.weight" in w.s else None
            self.dec.append((q, w.conv(f"{q}.conv1"), w.conv(f"{q}.conv2"), short))
        self.logit = w.conv(f"{d}.logit", bias=f"{d}.logit.bias")
        self.attn = w.conv(f"{d}.depth_collapse.attn_conv", bias=f"{d}.depth_collapse.attn_conv.bias")
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode="linear", align_corners=False)

    def features(self, x):
        """Decoder features before the depth collapse; x is (n, 16, 64, 64, 1) already divided by 255."""
        x = x * self.input[0] + self.input[1]
        x = nn.relu(conv(x, self.stem))
        feats = []
        current = 1
        for li, stride, c1, c2, c3, down in self.blocks:
            if li != current:
                feats.append(x)
                current = li
            residual = conv(x, down, stride) if down is not None else x
            out = nn.relu(conv(x, c1))
            out = nn.relu(conv(out, c2, stride))
            x = nn.relu(conv(out, c3) + residual)
        feats.append(x)
        w = self.w
        reduced = []
        for i, f in enumerate(feats):
            c = DEC[i]
            r = conv(f, self.reduce[i])
            r = nn.relu(group_norm(r, f"canonical.decoder.channel_reduce.{i}.1", w, min(32, c)))
            reduced.append(r)
        x = reduced[3]
        for k, skip in enumerate((reduced[2], reduced[1], reduced[0])):
            q, c1, c2, short = self.dec[k]
            x = mx.concatenate([self.up(x), skip], axis=-1)
            identity = conv(x, short) if short is not None else x
            out = conv(x, c1)
            out = nn.relu(group_norm(out, f"{q}.gn1", w, min(32, out.shape[-1])))
            out = group_norm(conv(out, c2), f"{q}.gn2", w, min(32, out.shape[-1]))
            x = nn.relu(out + identity)
        return x

    def __call__(self, x, support):
        """Ink probability (n, 64, 64) of normalised patches x (n, 16, 64, 64, 1) and their columns holding CT (n, 64, 64)."""
        f = self.features(x)
        logits = conv(f, self.logit)[..., 0]  # (n, 16, 64, 64)
        scores = conv(f, self.attn)[..., 0]
        z = (mx.arange(PATCH[0], dtype=mx.float32) - PATCH[0] // 2) / (PATCH[0] // 2)
        scores = scores + self.w.depth_scale * z[None, :, None, None]
        mask = support[:, None]
        scores = mx.where(mask, scores, -1e4)
        weights = mx.softmax(scores, axis=1) * mask
        weights = weights / mx.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        return mx.sigmoid((weights * logits).sum(axis=1))


def predict(model, volume, stride=32, batch=8):
    """hecate.predict's 2D map of a (z, y, x) uint8 render at 9.6 µm, central 16 planes, as float probabilities."""
    d, p = PATCH[0], PATCH[1]
    depth, h, w = volume.shape
    z0 = depth // 2 - d // 2
    planes = volume[z0 : z0 + d]

    def starts(n):
        return [*range(0, max(0, n - p), stride), max(0, n - p)]

    win = np.hanning(p)  # torch.hann_window(p, periodic=False)
    win = np.outer(win, win)
    win = np.maximum(win / win.max(), 0.001).astype(np.float32)
    numerator = np.zeros((h, w), np.float32)
    denominator = np.zeros((h, w), np.float32)
    origins = [(y, x) for y in starts(h) for x in starts(w)]
    for k in range(0, len(origins), batch):
        active, patches = [], []
        for y, x in origins[k : k + batch]:
            ph, pw = min(p, h - y), min(p, w - x)
            denominator[y : y + ph, x : x + pw] += win[:ph, :pw]
            raw = planes[:, y : y + ph, x : x + pw]
            if not raw.any():
                continue
            patch = np.zeros((d, p, p), np.uint8)
            patch[:, :ph, :pw] = raw
            patches.append(patch)
            active.append((y, x, ph, pw))
        if not patches:
            continue
        raw = np.stack(patches)
        support = mx.array(raw.any(axis=1))
        image = mx.array(raw.astype(np.float32)[..., None] / 255.0)
        prob = np.array(model(image, support) * support)
        for out, (y, x, ph, pw) in zip(prob * win, active):
            numerator[y : y + ph, x : x + pw] += out[:ph, :pw]
    return np.clip(numerator / denominator, 0, 1)


if __name__ == "__main__":
    import importlib.util
    import sys
    import time

    import torch

    checkpoint = sys.argv[1]
    spec = importlib.util.spec_from_file_location(
        "hecate", str(__import__("pathlib").Path(checkpoint).with_name("hecate.py"))
    )
    ref_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_module)
    ref = ref_module.load_model(checkpoint, "cpu")
    ours = Hecate(checkpoint)
    rng = np.random.default_rng(0)
    raw = rng.integers(60, 200, size=(4, 16, 64, 64)).astype(np.uint8)
    raw[1, :, :, :20] = 0
    image = torch.from_numpy(raw.astype(np.float32) / 255.0)[:, None]
    support = torch.from_numpy(raw).any(1)
    with torch.inference_mode():
        expect = torch.sigmoid(ref(image, support[:, None, None].expand(-1, 1, 16, -1, -1))).numpy()[:, 0]
    t = time.time()
    got = np.array(ours(mx.array(raw.astype(np.float32)[..., None] / 255.0), mx.array(raw.any(axis=1))))
    print(
        "forward: largest difference", float(np.abs(got - expect).max()), "time", round(time.time() - t, 2), "s for 4"
    )
