"""A reader's mean over the eight turns and flips of what it reads, and the mean of several readers."""

from pathlib import Path

import numpy as np

from ink.model.tile_reader import TileReader
from ink.model.wide_reader import WideReader, wide_of

TURNS = 4
TURNED = "turned:"
JOINED = "+"


class TurnedReader:
    """A tile reader asked eight times, each read turned back before the mean."""

    def __init__(self, weights):
        self.reader = TileReader(weights)

    def read(self, stack):
        total = np.zeros(stack.shape[1:], dtype=np.float32)
        for turns in range(TURNS):
            for flipped in (False, True):
                view = np.rot90(stack[:, :, ::-1] if flipped else stack, turns, (1, 2))
                read = np.rot90(self.reader.read(np.ascontiguousarray(view)), -turns)
                total += read[:, ::-1] if flipped else read
        return total / (2 * TURNS)


class JoinedReader:
    """The mean of several readers' maps of the same stack."""

    def __init__(self, readers):
        self.readers = readers

    def read(self, stack):
        return np.mean([reader.read(stack) for reader in self.readers], axis=0)


def named_reader(path):
    """A reader of weights and its name: over eight turns after TURNED, of more layers after wideNN:, the mean of readers joined by JOINED."""
    if JOINED in path:
        named = [named_reader(part) for part in path.split(JOINED)]
        return " + ".join(name for name, _ in named), JoinedReader([reader for _, reader in named])
    wide = wide_of(path)
    if wide:
        return f"{Path(wide[1]).stem} {wide[0]} layers", WideReader(Path(wide[1]), wide[0])
    turned = path.startswith(TURNED)
    path = Path(path.removeprefix(TURNED))
    return f"{path.stem}{' turned' if turned else ''}", TurnedReader(path) if turned else TileReader(path)
