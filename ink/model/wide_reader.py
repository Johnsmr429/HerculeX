"""The tile reader with a depth stem that takes more layers than the 17 it was taught on, the production reader left untouched."""

from ink.model import stock_model, tile_reader
from ink.model.input_stack import DEPTH
from ink.model.neural_engine import STEM_CHANNELS

WIDE = "wide"


class WideStem(tile_reader.DepthStem):
    """The Metal depth stem compiled for another number of layers."""

    def __init__(self, module, device, depth):
        super().__init__(module, device)
        self.depth = depth
        self.source = self.source.replace(f"Z = {DEPTH};", f"Z = {depth};")
        self.kernel = self.torch.mps.compile_shader(self.source)


class WideReader(tile_reader.TileReader):
    """A tile reader of stacks of `depth` layers."""

    def __init__(self, weights, depth, device="mps"):
        super().__init__(weights, device)
        self.depth = depth
        self.stem = WideStem(stock_model.taught_stem(weights, STEM_CHANNELS), device, depth)


def wide_of(path):
    """(depth, weights) of a path written as wideNN:WEIGHTS, or None."""
    head, _, rest = path.partition(":")
    return (int(head.removeprefix(WIDE)), rest) if head.startswith(WIDE) and rest else None
