"""The ink reader this machine runs, kept out of tile_reader.py, whose bytes are part of the Mac read digest."""

from ink.model.portable_reader import PortableReader
from ink.model.tile_reader import TileReader
from runner.compiled_kernels import APPLE


def reader(weights):
    """Metal and the Neural Engine on a Mac, else the same arithmetic in PyTorch."""
    return TileReader(weights) if APPLE else PortableReader(weights)
