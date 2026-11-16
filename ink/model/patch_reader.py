"""The organisers' per-patch ink reading, the same arithmetic as stock_model.InkModel.read, made fast.

InkModel.read normalises every 128-px patch on the CPU one at a time (normalize_robust: clip to the patch's 1st and
99th percentiles, centre on its median, divide by 1.4826 x its MAD), runs the network four patches at a time and adds
each tile into the map with NumPy. Here the patches of a batch are normalised together (the same percentiles, median
and MAD along each patch's own values, the same fallbacks for a flat patch), the next batch is prepared on a thread
while the network runs on the current one, and the Hann-weighted sums are kept on the GPU. The network, its weights,
its precision and the patch grid are InkModel's own. For checkpoints of the organisers' hybrid 3D-2D network
(ink_9um and fine-tunes such as TRACE), whose patches are normalised per patch.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ink.model.stock_model import InkModel

PERCENTILES = (1.0, 99.0)


def normalised(patches):
    """normalize_robust of each (z, y, x) patch of a (n, z, y, x) float32 batch, on its own values."""
    n = len(patches)
    flat = patches.reshape(n, -1)
    std = flat.std(axis=1)
    lo, hi = np.percentile(flat, PERCENTILES, axis=1)
    lowest, highest = flat.min(axis=1), flat.max(axis=1)
    np.clip(flat, lo[:, None], hi[:, None], out=flat)
    median = np.median(flat, axis=1)
    mad = 1.4826 * np.median(np.abs(flat - median[:, None]), axis=1)
    bad = ~np.isfinite(mad) | (mad < 1e-6)
    for k in np.flatnonzero(bad):
        candidates = [abs(std[k])] if np.isfinite(std[k]) else []
        span = hi[k] - lo[k] if np.isfinite(hi[k] - lo[k]) else highest[k] - lowest[k]
        if np.isfinite(span):
            candidates.append(abs(span) / 2.0)
        mad[k] = next((c for c in candidates if c >= 1e-6), 1.0)
    flat -= median[:, None]
    flat /= mad[:, None]
    np.nan_to_num(flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.reshape(patches.shape)


class PatchReader:
    """A fast InkModel.read for checkpoints of the organisers' hybrid network."""

    def __init__(self, weights=(), names=None, device="mps", batch=4):
        from ink.model import stock_model

        self.model = InkModel(names=names or tuple(stock_model.CHECKPOINTS)[:1], weights=list(weights), device=device)
        if self.model.models[0].preprocessing != "tifxyz_robust":
            raise ValueError(f"PatchReader reproduces tifxyz_robust only, not {self.model.models[0].preprocessing}")
        self.torch, self.batch = self.model.torch, batch

    def read(self, stack):
        torch, m = self.torch, self.model
        depth, height, width = stack.shape
        patch, step = m.patch, m.patch // 2
        ys = sorted({*range(0, max(height - patch, 0) + 1, step), max(height - patch, 0)})
        xs = sorted({*range(0, max(width - patch, 0) + 1, step), max(width - patch, 0)})
        spots = [(y, x) for y in ys for x in xs if stack[:, y : y + patch, x : x + patch].any()]
        blend = torch.from_numpy(m.weight).to(m.device)
        total = torch.zeros((height, width), dtype=torch.float32, device=m.device)
        weight = torch.zeros((height, width), dtype=torch.float32, device=m.device)

        def prepared(part):
            block = np.zeros((len(part), depth, patch, patch), dtype=np.float32)
            for k, (y, x) in enumerate(part):
                piece = stack[:, y : y + patch, x : x + patch]
                block[k, :, : piece.shape[1], : piece.shape[2]] = piece
            return part, normalised(block)

        batches = [spots[k : k + self.batch] for k in range(0, len(spots), self.batch)]
        with torch.inference_mode(), ThreadPoolExecutor(1) as ahead:
            pending = ahead.submit(prepared, batches[0]) if batches else None
            for k in range(len(batches)):
                part, block = pending.result()
                if k + 1 < len(batches):
                    pending = ahead.submit(prepared, batches[k + 1])
                images = torch.from_numpy(block)[:, None].to(m.device).half()
                mean = sum(
                    m.api.logits_to_probabilities(c.model(images), image_hw=(patch, patch)) for c in m.models
                ) / len(m.models)
                for (y, x), tile in zip(part, mean[:, 0]):
                    h, w = min(patch, height - y), min(patch, width - x)
                    total[y : y + h, x : x + w] += tile[:h, :w] * blend[:h, :w]
                    weight[y : y + h, x : x + w] += blend[:h, :w]
        out = (total / weight.clamp_min(1e-6)).cpu().numpy()
        torch.mps.empty_cache()
        return out
