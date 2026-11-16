"""Sheet nodes of a brick store on the CPU: the reference the GPU path is tested against."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from unwrapping.detection.kernel_bindings import (
    AIR_LEVEL,
    CELL,
    MARGIN,
    MAXN,
    NODE,
    STATS,
    cpu_library,
)

CELLS_PER_SLAB = 8
WORKERS = 6


def slab_nodes(store, cz0, cz1, grid, first_plane, workers=WORKERS, cell=CELL, margin=MARGIN):
    """Nodes of cell planes cz0..cz1 in cell order, each with its cell index among the used planes."""
    lib = cpu_library(cell, margin)
    Z, Y, X = store.shape
    _, cy, cx = grid
    z0, z1 = max(cz0 * cell - margin, 0), min(cz1 * cell + margin, Z)
    slab = np.ascontiguousarray(store.read((z0, z1), (0, Y), (0, X)))
    rows = (cz1 - cz0) * cy
    nodes = np.zeros((rows, cx, MAXN), dtype=NODE)
    counts = np.zeros((rows, cx), dtype=np.uint8)
    stats = np.zeros((rows, cx), dtype=STATS)

    def run(row):
        k, y = divmod(row, cy)
        lib.hx_cell_row(
            slab.ctypes.data,
            z1 - z0,
            Y,
            X,
            (cz0 + k) * cell - z0,
            y * cell,
            cx,
            AIR_LEVEL,
            z0,
            nodes[row].ctypes.data,
            counts[row].ctypes.data,
            stats[row].ctypes.data,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(rows)))
    keep = np.arange(MAXN)[None, None, :] < counts[:, :, None]
    cells = (cz0 - first_plane) * cy * cx + np.repeat(np.arange(rows * cx, dtype=np.int64), counts.ravel())
    return nodes[keep], cells.astype(np.int32)


def detect(store, cell_planes, sink_nodes, sink_cells, workers=WORKERS, cell=CELL, margin=MARGIN):
    """Write the nodes of the given cell planes to two open files."""
    Z, Y, X = store.shape
    grid = (-(-Z // cell), -(-Y // cell), -(-X // cell))
    first, last = cell_planes
    total = 0
    for cz0 in range(first, last, CELLS_PER_SLAB):
        nodes, cells = slab_nodes(store, cz0, min(cz0 + CELLS_PER_SLAB, last), grid, first, workers, cell, margin)
        nodes.tofile(sink_nodes)
        cells.tofile(sink_cells)
        total += len(nodes)
    return total
