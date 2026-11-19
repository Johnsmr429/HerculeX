"""The Vesuvius Challenge open ink models: mean of two seeds, ink probability per voxel."""

import hashlib
import importlib
import os
import subprocess
import sys
import urllib.request
import zipfile
from argparse import Namespace

import numpy as np

from storage.data_roots import EXTERNAL_TOOLS, INK_WEIGHTS

SOURCE = EXTERNAL_TOOLS / "villa"
COMMIT = "960d76ac7f32557c2ecf323323288d8f23bb71f1"
REVISION = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
CHECKPOINTS = {
    "hybrid_3d2d-seed42/step-075000.pth": "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab",
    "hybrid_3d2d-seed43/step-075000.pth": "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f",
}
NO_INK = 0.25
SURE = 0.65
BATCH = 4
_last_checkpoint = None


def _checkpoint_stamp(stream):
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def checkpoint_hash(path):
    """Reuse the full hash of the last unchanged checkpoint within the process."""
    global _last_checkpoint
    with open(path, "rb") as stream:
        stamp = _checkpoint_stamp(stream)
        last = _last_checkpoint
        if last is not None and last[0] == stamp:
            return last[1].copy()
        model = hashlib.file_digest(stream, "sha256")
        if _checkpoint_stamp(stream) != stamp:
            raise RuntimeError(f"checkpoint changed while reading {path}")
    _last_checkpoint = stamp, model
    return model.copy()


def source():
    """The inference code at its pinned revision."""
    if not SOURCE.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", "https://github.com/ScrollPrize/villa.git", str(SOURCE)],
            check=True,
        )
        subprocess.run(["git", "-C", str(SOURCE), "sparse-checkout", "set", "vesuvius"], check=True)
        subprocess.run(["git", "-C", str(SOURCE), "checkout", "--detach", COMMIT], check=True)
    if subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError(f"Ink code is not at {COMMIT}: {SOURCE}")
    return SOURCE / "vesuvius" / "src"


def checkpoint(name):
    """A published checkpoint, fetched once and checked against its digest."""
    path = INK_WEIGHTS / "ink-9um" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with (
            urllib.request.urlopen(
                f"https://huggingface.co/scrollprize/ink_9um/resolve/{REVISION}/{name}", timeout=120
            ) as response,
            part.open("wb") as out,
        ):
            while block := response.read(2**20):
                out.write(block)
        part.replace(path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINTS[name]:
            raise ValueError(f"Checkpoint differs from its published digest: {path}")
    return path


def folded(conv):
    """A 3x3x3 convolution as a 2-D one over three depth-shifted copies."""
    import torch
    from torch.nn import functional

    class Folded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            out, channels = conv.weight.shape[:2]
            self.register_buffer(
                "weight", conv.weight.detach().permute(0, 2, 1, 3, 4).reshape(out, 3 * channels, 3, 3).contiguous()
            )

        def forward(self, x):
            batch, channels, depth, height, width = x.shape
            padded = functional.pad(x, (0, 0, 0, 0, 1, 1))
            stacked = torch.cat([padded[:, :, k : k + depth] for k in range(3)], dim=1).permute(0, 2, 1, 3, 4)
            del padded
            stacked = stacked.reshape(batch * depth, 3 * channels, height, width)
            flat = functional.conv2d(stacked, self.weight, padding=1)
            return flat.reshape(batch, depth, -1, height, width).permute(0, 2, 1, 3, 4)

    return Folded()


def taught_stem(weights, channels):
    """Load the trained depth stem without materializing the rest of the checkpoint."""
    import torch

    sys.path.insert(0, str(source()))
    hybrid = importlib.import_module("vesuvius.ink_detection.models.hybrid_3d2d")
    stem = hybrid.LocalDepthFusionStem(channels=channels).eval().requires_grad_(False)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(weights))
    prefix = "model.depth_fusion."
    stem.load_state_dict({key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)})
    features = stem.features
    features[0], features[3] = folded(features[0]), folded(features[3])
    return stem


class InkModel:
    def __init__(self, names=tuple(CHECKPOINTS), device="mps", weights=()):
        """The published seeds, or the same networks with weights taught on a scroll, one file for each."""
        import torch

        sys.path.insert(0, str(source()))
        self.torch, self.device = torch, torch.device(device)
        self.api = importlib.import_module("vesuvius.ink_detection.inference.infer")
        self.models = [
            self.api.configure_model(Namespace(checkpoint=checkpoint(n), amp_dtype="default")) for n in names
        ]
        for configured, path in zip(self.models, weights):
            configured.model.load_state_dict(torch.load(path, weights_only=True))
        self.models = self.models[: len(weights) or len(self.models)]
        for configured in self.models:
            features = configured.model.model.depth_fusion.features
            features[0], features[3] = folded(features[0]), folded(features[3])
            configured.model.to(self.device).half()
        self.patch, self.depth = self.models[0].patch_size, self.models[0].input_depth
        self.weight = self.api.compute_importance_map_2d(patch_size=(self.patch, self.patch), mode="hann").numpy()

    def read(self, stack, every=1, patch=None, step=None):
        """Probability of ink of a (depth, height, width) uint8 stack, with the upstream normalization and Hann blend, a patch every step pixels."""
        _, height, width = stack.shape
        patch = patch or self.patch
        step = step or patch // 2
        blend = (
            self.weight
            if patch == self.patch
            else self.api.compute_importance_map_2d(patch_size=(patch, patch), mode="hann").numpy()
        )
        total, weight = np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)
        ys = sorted({*range(0, max(height - patch, 0) + 1, step), max(height - patch, 0)})
        xs = sorted({*range(0, max(width - patch, 0) + 1, step), max(width - patch, 0)})
        spots = [(y, x) for y in ys for x in xs if stack[:, y : y + patch, x : x + patch].any()][::every]
        batch = max(1, BATCH * self.patch**2 // patch**2)
        with self.torch.inference_mode():
            for k in range(0, len(spots), batch):
                part = spots[k : k + batch]
                patches = np.stack([self.patch_at(stack, y, x, patch) for y, x in part])
                images = self.torch.from_numpy(patches)[:, None].to(self.device).half()
                mean = sum(
                    self.api.logits_to_probabilities(m.model(images), image_hw=(patch, patch)) for m in self.models
                ) / len(self.models)
                for (y, x), tile in zip(part, mean.cpu().numpy()[:, 0]):
                    h, w = tile[: height - y, : width - x].shape
                    total[y : y + h, x : x + w] += tile[:h, :w] * blend[:h, :w]
                    weight[y : y + h, x : x + w] += blend[:h, :w]
        self.torch.mps.empty_cache()
        return total / np.maximum(weight, 1e-6)

    def patch_at(self, stack, y, x, patch):
        block = np.zeros((stack.shape[0], patch, patch), dtype=np.float32)
        part = stack[:, y : y + patch, x : x + patch]
        block[:, : part.shape[1], : part.shape[2]] = part
        return self.api.normalize_flat_patch(block, self.models[0].preprocessing)
