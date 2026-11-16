"""The sheet graph of a field: cells, sheet edges and rays between them."""

import itertools
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from runner.stage_queue import record_of
from storage.data_roots import SCRATCH as RUNTIME_SCRATCH
from storage.node_store import read_cells, read_nodes
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level.piece_kernel_bindings import EDGE, PAIR, library

SIGMA = 2.0
MAX_DEPTH = 4.0
MIN_COS = float(np.cos(np.radians(30)))
MAX_LATERAL = 1.9 * CELL
MIN_PROMINENCE = 3.0
ATOM_PLANES = 24
QUALITY_SPAN = 0.65
RADIAL_REACH = 40.0
RADIAL_TUBE = 6.0
SAME_SHEET = 3.0
RADIAL_RANKS = 1
SCRATCH = Path(os.environ.get("HERCULEX_PIECE_SCRATCH", RUNTIME_SCRATCH / "herculex-pieces"))


def scratch_file():
    """A file with no name, on the internal disc."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryFile(dir=SCRATCH)


def file_backed(array):
    """A finished array moved into a scratch file."""
    if not array.size:
        return array
    with scratch_file() as stream:
        array.tofile(stream)
        stream.flush()
        return np.memmap(stream, dtype=array.dtype, mode="r", shape=array.shape)


def merged_edges(runs):
    """Runs of edges, each strongest first, as one file-backed array, the order of a stable sort of them all."""
    total = sum(len(run) for run in runs)
    if not total:
        return np.zeros(0, dtype=EDGE)
    with scratch_file() as stream:
        stream.truncate(total * EDGE.itemsize)
        out = np.memmap(stream, dtype=EDGE, mode="r+", shape=(total,))
    starts = np.array([run.ctypes.data for run in runs], dtype=np.int64)
    lengths = np.array([len(run) for run in runs], dtype=np.int64)
    library().hx_merge_edges(starts.ctypes.data, lengths.ctypes.data, len(runs), out.ctypes.data)
    return out


def load_field(chunks, planes=None):
    """Nodes of consecutive chunks within a range of cell planes, each with its cell in one grid and its id."""
    chunks = [Path(c) for c in chunks]
    records = [record_of(c) for c in chunks]
    for a, b in itertools.pairwise(records):
        if a["cell_planes"][1] != b["cell_planes"][0]:
            raise ValueError(f"chunks do not tile: {a['cell_planes']} then {b['cell_planes']}")
    low, high = planes or (records[0]["cell_planes"][0], records[-1]["cell_planes"][1])
    _, gy, gx = records[0]["grid"]
    per_plane = gy * gx
    nodes, cells, ids = [], [], []
    for chunk, r in zip(chunks, records):
        cell = read_cells(chunk)
        first, last = r["cell_planes"]
        # Cells are stored in ascending order, so a plane range selects one node span.
        start = 0 if low <= first else int(cell.searchsorted((low - first) * per_plane))
        stop = len(cell) if high >= last else int(cell.searchsorted((high - first) * per_plane))
        nodes.append(read_nodes(chunk, (start, stop)))
        cells.append(cell[start:stop] + np.int32((first - low) * per_plane))
        ids.append((np.int64(r.get("rows", [0])[0]) << 32) + np.arange(start, stop, dtype=np.int64))
        del cell
    record = records[0] | {"cell_planes": [low, high], "chunks": [c.name for c in chunks]}
    one = len(nodes) == 1
    return (
        nodes[0] if one else np.concatenate(nodes),
        cells[0] if one else np.concatenate(cells),
        record,
        (high - low, gy, gx),
        ids[0] if one else np.concatenate(ids),
    )


def filtered(nodes, cells, grid, ids=None):
    """Drop weak peaks and rebuild the cell table."""
    keep = nodes["prominence"] / 2 >= MIN_PROMINENCE
    nodes, cell_of = nodes[keep], np.ascontiguousarray(cells[keep], dtype=np.int32)
    ids = None if ids is None else ids[keep]
    first = np.empty(int(np.prod(grid)) + 1, dtype=np.int32)
    library().hx_cell_table(cell_of.ctypes.data, len(cell_of), first.ctypes.data, len(first) - 1)
    return (
        (np.ascontiguousarray(nodes), first, cell_of)
        if ids is None
        else (np.ascontiguousarray(nodes), first, cell_of, ids)
    )


def used_planes(first, grid):
    """The cell planes that hold nodes, from the table's first entry of each plane."""
    return np.nonzero(np.diff(first[:: grid[1] * grid[2]]))[0]


def build_edges(nodes, first, grid, theta, quality, planes, workers=6, max_depth=MAX_DEPTH):
    """Edges from the given cell planes to their forward neighbours, strongest first."""
    lib = library()
    GZ, GY, GX = grid
    capacity = max(1 << 22, 40 * len(nodes) // max(GZ, 1))

    def plane(z):
        out = np.zeros(capacity, dtype=EDGE)
        n = lib.hx_sheet_edges(
            nodes.ctypes.data,
            first.ctypes.data,
            GZ,
            GY,
            GX,
            z,
            theta.ctypes.data,
            quality.ctypes.data,
            SIGMA,
            max_depth,
            MIN_COS,
            MAX_LATERAL,
            out.ctypes.data,
            capacity,
        )
        if n < 0:
            raise MemoryError("edge buffer too small")
        return out[:n].copy()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        edges = np.concatenate(list(pool.map(plane, planes)) or [np.zeros(0, dtype=EDGE)])
    scratch = np.empty_like(edges)
    lib.hx_sort_edges(edges.ctypes.data, len(edges), scratch.ctypes.data)
    return edges


class Forest:
    """Union-find over nodes that keeps each node's winding offset to its root."""

    def __init__(self, count):
        self.count = count
        self.arrays = [np.zeros(count, dtype=np.int32) for _ in range(5)]
        library().hx_forest_init(count, *[a.ctypes.data for a in self.arrays])

    def pointers(self):
        return [a.ctypes.data for a in self.arrays]

    def labels(self):
        root, n = np.zeros(self.count, dtype=np.int32), np.zeros(self.count, dtype=np.int32)
        library().hx_labels(self.count, *self.pointers(), root.ctypes.data, n.ctypes.data)
        return root, n


def radial_pairs(nodes, first, grid, plane0, member, workers=6):
    """Sheets met along each member's outward normal, as (node, node, rank), by casting rays through the cells."""
    lib = library()
    GZ, GY, GX = grid
    flags = np.ascontiguousarray(member.astype(np.uint8))
    capacity = max(1 << 20, 8 * RADIAL_RANKS * int(member.sum()) // max(GZ, 1))

    def plane(z):
        out = np.zeros(capacity, dtype=PAIR)
        n = lib.hx_radial_pairs(
            nodes.ctypes.data,
            first.ctypes.data,
            GZ,
            GY,
            GX,
            z,
            CELL,
            plane0,
            flags.ctypes.data,
            RADIAL_REACH,
            RADIAL_TUBE,
            SAME_SHEET,
            RADIAL_RANKS,
            out.ctypes.data,
            capacity,
        )
        if n < 0:
            raise MemoryError("pair buffer too small")
        return out[:n].copy()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.concatenate(list(pool.map(plane, used_planes(first, grid).tolist())))
