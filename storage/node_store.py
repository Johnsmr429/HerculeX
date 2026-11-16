"""A chunk's sheet nodes and cells packed without loss to about 0.43 of their size."""

from pathlib import Path

import numpy as np
import zstandard

from unwrapping.detection.kernel_bindings import NODE

LEVEL = 6


def squeezed(array):
    return zstandard.ZstdCompressor(level=LEVEL, threads=2).compress(array)


def nodes_from(packed, span=None):
    columns = np.frombuffer(zstandard.ZstdDecompressor().decompress(packed), dtype=np.uint8).reshape(NODE.itemsize, -1)
    if span is not None:
        columns = columns[:, slice(*span)]
    return np.ascontiguousarray(columns.T).view(NODE).ravel()


def cells_from(packed):
    steps = np.frombuffer(zstandard.ZstdDecompressor().decompress(packed), dtype=np.int32)
    return np.cumsum(steps, dtype=np.int32)


def read_nodes(folder, span=None):
    """A chunk's nodes, packed or plain, optionally within [start, stop)."""
    folder = Path(folder)
    packed = folder / "nodes.zst"
    if packed.exists():
        return nodes_from(packed.read_bytes(), span)
    path = folder / "nodes.bin"
    if span is None:
        return np.fromfile(path, dtype=NODE)
    start, stop = span
    return np.fromfile(path, dtype=NODE, count=stop - start, offset=start * NODE.itemsize)


def read_cells(folder):
    """A chunk's cells, packed or plain."""
    packed = Path(folder) / "cells.zst"
    return (
        cells_from(packed.read_bytes()) if packed.exists() else np.fromfile(Path(folder) / "cells.bin", dtype=np.int32)
    )


def pack_chunk(folder):
    """The plain files go only once the packed ones decode exactly."""
    folder = Path(folder)
    nodes = np.fromfile(folder / "nodes.bin", dtype=NODE)
    cells = np.fromfile(folder / "cells.bin", dtype=np.int32)
    packed_nodes = squeezed(np.ascontiguousarray(nodes.view(np.uint8).reshape(-1, NODE.itemsize).T))
    packed_cells = squeezed(np.diff(cells, prepend=np.int32(0)))
    nodes_match = np.array_equal(nodes_from(packed_nodes).view(np.uint8), nodes.view(np.uint8))
    if not nodes_match or not np.array_equal(cells_from(packed_cells), cells):
        raise ValueError(f"packing would change {folder}")
    for name, packed in (("nodes.zst", packed_nodes), ("cells.zst", packed_cells)):
        part = folder / f"{name}.part"
        part.write_bytes(packed)
        part.rename(folder / name)
    (folder / "nodes.bin").unlink()
    (folder / "cells.bin").unlink()
