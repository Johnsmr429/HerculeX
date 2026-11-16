"""A Reader v3 network read wherever PyTorch runs (CPU or CUDA): TileReader's arithmetic without Metal or the Neural Engine.

TileReader runs the depth stem as a Metal kernel and the 2-D body on the Neural Engine, in half precision, on Apple
silicon only. Here the same arithmetic runs in PyTorch, in float32:
- the stack's raw values clipped and scaled once (tile_reader.measured, raw_scale), and the stem's instance norms
  measured once on the same tiles (tile_reader.fixed_norms), as TileReader fixes them;
- empty columns filled from shifted, flipped copies of the stack (tile_reader.filled), coverage kept;
- the whole network (neural_engine.whole_network: the stem expressed in 2-D over the 17 layers, its attention pool and
  peak, then the body) on 512-pixel tiles every 384, one at a time on a CPU (BATCH);
- tiles blended by a Hann window (tile_reader.Sums).
TurnedPortableReader reads the eight turns and flips of a stack, each turned back, as TurnedReader does.
"""

import numpy as np

from ink.model import stock_model, tile_reader
from ink.model.neural_engine import STEM_CHANNELS, whole_network

BATCH = {"cpu": 1, "cuda": 4}  # tiles a forward pass: one keeps a CPU run within 8 GB


class PortableReader:
    def __init__(self, weights, device="cpu"):
        import torch

        self.torch, self.device = torch, torch.device(device)
        self.batch = BATCH.get(self.device.type, 1)
        net = (
            stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[weights], device="cpu")
            .models[0]
            .model.model.requires_grad_(False)
        )
        self.stem, body = net.depth_fusion.float().eval(), net.network.float().eval()
        features = self.stem.features
        if isinstance(
            features[0], torch.nn.Conv3d
        ):  # the 3-D convolutions as 2-D ones over depth taps, as TileReader's
            features[0], features[3] = stock_model.folded(features[0]), stock_model.folded(features[3])
        self.whole = whole_network(self.stem, body).float().to(self.device).eval()
        self.norms = tile_reader.fixed_norms(torch, self.stem)

    def fix_to(self, stack):
        """The raw scale of a stack and its stem norms, measured on the tiles TileReader measures."""
        torch = self.torch
        tiles = tile_reader.measured_tiles(stack)
        scale = tile_reader.raw_scale(tiles)
        low, high, centre, spread = scale
        image = (np.clip(np.stack(tiles).astype(np.float32), low, high) - centre) / spread
        for norm in self.norms:
            norm.measuring = True
        with torch.inference_mode():
            self.stem(torch.from_numpy(image)[:, None])
        for norm in self.norms:
            norm.measuring = False
        with torch.no_grad():
            parts = [
                p.reshape(1, STEM_CHANNELS, 1, 1).float().to(self.device)
                for norm in self.norms
                for p in norm.scale_and_shift()
            ]
        return scale, parts

    def read(self, stack):
        """Probability of ink of a (17, height, width) uint8 stack."""
        torch = self.torch
        scale, norms = self.fix_to(stack)
        low, high, centre, spread = scale
        grey = ((np.clip(np.arange(256, dtype=np.float32), low, high) - centre) / spread).astype(np.float32)
        depth, height, width = stack.shape
        sums = tile_reader.Sums(height, width)
        ty, tx = sums.tile_y, sums.tile_x
        whole, coverage = tile_reader.prepare_stack(np.ascontiguousarray(stack))
        places = [
            (y, x)
            for y in tile_reader.tile_starts(height, ty)
            for x in tile_reader.tile_starts(width, tx)
            if coverage[y : y + ty, x : x + tx].any()
        ]
        with torch.inference_mode():
            for k in range(0, len(places), self.batch):
                part = places[k : k + self.batch]
                block = np.zeros((len(part), depth, ty, tx), dtype=np.uint8)
                for i, (y, x) in enumerate(part):
                    piece = whole[:, y : y + ty, x : x + tx]
                    block[i, :, : piece.shape[1], : piece.shape[2]] = piece
                image = torch.from_numpy(grey[block]).to(self.device)
                logits = self.whole(image, *norms)
                reads = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
                for (y, x), read in zip(part, reads):
                    sums.add(y, x, min(ty, height - y), min(tx, width - x), read)
        return sums.result()


class TurnedPortableReader:
    """A portable reader asked eight times, each read turned back before the mean (as turned_reader.TurnedReader)."""

    def __init__(self, weights, device="cpu"):
        self.reader = PortableReader(weights, device)

    def read(self, stack):
        total = np.zeros(stack.shape[1:], dtype=np.float32)
        for turns in range(4):
            for flipped in (False, True):
                view = np.rot90(stack[:, :, ::-1] if flipped else stack, turns, (1, 2))
                read = np.rot90(self.reader.read(np.ascontiguousarray(view)), -turns)
                total += read[:, ::-1] if flipped else read
        return total / 8
