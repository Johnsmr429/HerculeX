"""A reader's own reads of our pieces of a scroll, kept beside them as targets no hand drew."""

import sys
from pathlib import Path

import numpy as np

from ink.model.input_stack import PAST_FACE, face_layer_of
from ink.model.turned_reader import TurnedReader
from storage.data_roots import OWN_MESH_LABELS

READS = "reads"
ROUTES = ("level_route", "grown_route")
LAYERS = 17


def read_path(path):
    return path.parent / READS / path.name


def build(sample, weights):
    """The mean read over eight turns of every window of a scroll on our own pieces, as bytes."""
    reader = TurnedReader(Path(weights))
    for route in ROUTES:
        for path in sorted((OWN_MESH_LABELS / route / sample).glob("*.npy")):
            out = read_path(path)
            if out.exists():
                continue
            stack = np.load(path, mmap_mode="r")
            first = min(max(face_layer_of(stack) + PAST_FACE, 0), stack.shape[0] - LAYERS)
            read = reader.read(np.ascontiguousarray(stack[first : first + LAYERS]))
            out.parent.mkdir(exist_ok=True)
            np.save(out, np.round(read * 255).astype(np.uint8))
            print(route, path.stem, read.shape, np.percentile(read, [10, 50, 90, 99]).round(3), flush=True)


if __name__ == "__main__":
    build(*sys.argv[1:3])
