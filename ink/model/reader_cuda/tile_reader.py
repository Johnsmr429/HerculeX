"""The CUDA lane's ink reader: the Apple lane's arithmetic in PyTorch, in half precision on CUDA and in float32 on a CPU,
the norms measured in float32 on both.

- The stack's raw values are clipped and scaled once, and the stem's instance norms measured once on the same tiles
  (tiles.measured, tiles.fixed_norms), as the Apple lane fixes them.
- Empty columns are filled from shifted, flipped copies of the stack (tiles.filled), coverage kept.
- The whole network (native_network: the stem's two 3-D convolutions, its attention pool and peak as reductions over
  depth, then the body) reads 512-pixel tiles every 384 pixels, BATCH at a time, cut from the stack on the device.
- Tiles are blended by a Hann window (tiles.Sums' window and order) on the device, and the map is copied back once.
"""

import copy
import sys

import numpy as np

from ink.model import stock_model
from ink.model.tiles import STEM_CHANNELS, TILE, PreparedStack, Sums, fixed_norms, measured, prepare_stack, tile_starts

BATCH = {"cpu": 1, "cuda": 4}  # tiles a forward pass: one keeps a CPU run within 8 GB


def native_network(stem, body):
    """Depth stem and body as one torch module taking a tile's 17 layers and the stack's norms."""
    import torch
    from torch.nn import functional

    def unfolded(layer):
        out, taps = layer.weight.shape[:2]
        return layer.weight.detach().reshape(out, 3, taps // 3, 3, 3).permute(0, 2, 1, 3, 4).contiguous()

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = body
            self.register_buffer("first", unfolded(stem.features[0]))
            self.register_buffer("second", unfolded(stem.features[3]))
            self.register_buffer(
                "attention", stem.attention_logits.weight.detach().reshape(1, -1, 1, 1, 1).contiguous()
            )
            self.register_buffer("attention_bias", stem.attention_logits.bias.detach().reshape(1).contiguous())

        def forward(self, image, scale_1, shift_1, scale_2, shift_2):
            first, second = self.first * scale_1.reshape(-1, 1, 1, 1, 1), self.second * scale_2.reshape(-1, 1, 1, 1, 1)
            a = functional.leaky_relu(functional.conv3d(image[:, None], first, shift_1.reshape(-1), padding=1), 0.01)
            b = functional.leaky_relu(functional.conv3d(a, second, shift_2.reshape(-1), padding=1), 0.01)
            share = torch.softmax(functional.conv3d(b, self.attention, self.attention_bias), dim=2).to(b.dtype)
            return self.body(torch.cat([(b * share).sum(dim=2), b.amax(dim=2)], 1))["ink"]

    return Native().eval()


class TileReader:
    def __init__(self, weights, device=None):
        import torch

        self.torch, self.device = torch, torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device is None and self.device.type == "cpu":
            print(
                "the ink reader runs on the CPU, many times slower, as PyTorch sees no CUDA GPU",
                file=sys.stderr,
                flush=True,
            )
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
        self.whole = native_network(self.stem, body).float().to(self.device).eval()
        self.half = self.device.type == "cuda"
        self.compiled = torch.compile(self.whole, dynamic=False) if self.half else None
        if self.half:
            self.stem = copy.deepcopy(self.stem).to(self.device)
        self.norms = fixed_norms(torch, self.stem)

    def fix_to(self, stack):
        """Fix the statistics to a stack, measured on the tiles TileReader measures."""
        self.fixed = self.statistics_of(*measured(stack))

    def statistics_of(self, tiles, scale):
        """The raw scale of measured tiles and the stem norms they give."""
        torch = self.torch
        low, high, centre, spread = scale
        image = torch.from_numpy((np.clip(tiles.astype(np.float32), low, high) - centre) / spread)[:, None]
        for norm in self.norms:
            norm.measuring = True
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            self.stem(image.to(self.device))
        for norm in self.norms:
            norm.measuring = False
        with torch.no_grad():
            parts = [
                p.reshape(1, STEM_CHANNELS, 1, 1).float().to(self.device)
                for norm in self.norms
                for p in norm.scale_and_shift()
            ]
        return scale, parts

    def read(self, stack, fixed=False):
        """Probability of ink of a (17, height, width) uint8 stack, with its own statistics unless they stay fixed."""
        (read,) = self.maps([(stack, None if fixed else measured(stack))])
        if isinstance(read, Exception):
            raise read
        return read

    def maps(self, stacks):
        """Read each raw or prepared stack with its measurements, or None to reuse the last statistics, as TileReader.maps does."""
        for stack, measure in stacks:
            try:
                if measure is not None:
                    self.fixed = self.statistics_of(*measure)
                yield self.read_fixed(stack)
            except Exception as error:
                yield error

    def network(self, tile):
        """The compiled network for whole tiles, which make up most of a read; the eager one for smaller tiles."""
        torch = self.torch
        whole = tile == (TILE, TILE)

        def run(image, *norms):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                if whole and self.compiled is not None:
                    try:
                        return self.compiled(image, *norms)
                    except Exception as error:
                        print(f"the compiled reader failed ({error!r:.200}); reading uncompiled", file=sys.stderr)
                        self.compiled = None
                return self.whole(image, *norms)

        return run

    def read_fixed(self, stack):
        """One stack read with the fixed statistics."""
        torch = self.torch
        scale, norms = self.fixed
        low, high, centre, spread = scale
        grey = ((np.clip(np.arange(256, dtype=np.float32), low, high) - centre) / spread).astype(np.float32)
        grey = torch.from_numpy(grey).to(self.device)
        depth, height, width = stack.shape
        sums = Sums(height, width)
        ty, tx = sums.tile_y, sums.tile_x
        prepared = isinstance(stack, PreparedStack)
        whole, coverage = stack if prepared else prepare_stack(np.ascontiguousarray(stack))
        places = [
            (y, x)
            for y in tile_starts(height, ty)
            for x in tile_starts(width, tx)
            if coverage[y : y + ty, x : x + tx].any()
        ]
        source = torch.from_numpy(np.ascontiguousarray(whole)).to(self.device)
        blend = torch.from_numpy(sums.blend).to(self.device)
        total = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        weight = torch.zeros_like(total)
        with torch.inference_mode():
            for k in range(0, len(places), self.batch):
                part = places[k : k + self.batch]
                block = torch.zeros((len(part), depth, ty, tx), dtype=torch.uint8, device=self.device)
                for i, (y, x) in enumerate(part):
                    piece = source[:, y : y + ty, x : x + tx]
                    block[i, :, : piece.shape[1], : piece.shape[2]] = piece
                logits = self.network((ty, tx))(grey[block.int()], *norms)
                reads = torch.sigmoid(logits.float())[:, 0]
                for (y, x), read in zip(part, reads):
                    h, w = min(ty, height - y), min(tx, width - x)
                    total[y : y + h, x : x + w] += read[:h, :w] * blend[:h, :w]
                    weight[y : y + h, x : x + w] += blend[:h, :w]
        return (total / torch.clamp(weight, min=1e-6)).cpu().numpy()
