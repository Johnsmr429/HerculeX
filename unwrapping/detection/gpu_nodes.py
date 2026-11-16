"""Sheet nodes of a brick store on the GPU: one Metal threadgroup per 8-voxel cell, the next slab read while this one runs."""

import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from unwrapping.detection.kernel_bindings import (
    AIR_LEVEL,
    CELL,
    MARGIN,
    NODE,
    gpu_library,
)

PLANES_PER_SLAB = 64


def read_slab(store, plane0, plane1):
    Z, Y, X = store.shape
    z0, z1 = max(plane0 * CELL - MARGIN, 0), min(plane1 * CELL + MARGIN, Z)
    return z0, z1, np.ascontiguousarray(store.read((z0, z1), (0, Y), (0, X)))


def slab_nodes(store, read, plane0, plane1, grid, first_plane):
    """Nodes of cell planes plane0..plane1 in cell order, each cell's nodes in depth order."""
    lib = gpu_library()
    _, Y, X = store.shape
    _, cy, cx = grid
    z0, z1, slab = read
    capacity = min((plane1 - plane0) * cy * cx * 3, 2**31 // 20)
    nodes = np.zeros(capacity, dtype=NODE)
    ids = np.zeros(capacity, dtype=np.int32)
    count, seconds = ctypes.c_uint32(0), ctypes.c_double(0)
    status = lib.hx_gpu_cells(
        slab.ctypes.data,
        z1 - z0,
        Y,
        X,
        plane1 - plane0,
        cy,
        cx,
        plane0 * CELL - z0,
        AIR_LEVEL,
        z0,
        nodes.ctypes.data,
        ids.ctypes.data,
        capacity,
        ctypes.byref(count),
        ctypes.byref(seconds),
    )
    if status:
        raise RuntimeError(f"GPU cells failed with status {status}")
    n = count.value
    order = np.argsort(ids[:n], kind="stable")
    return nodes[:n][order], (ids[:n][order] + (plane0 - first_plane) * cy * cx).astype(np.int32), seconds.value


def detect(store, cell_planes, sink_nodes, sink_cells):
    """Write the nodes of the given cell planes to two open files."""
    Z, Y, X = store.shape
    grid = (-(-Z // CELL), -(-Y // CELL), -(-X // CELL))
    first, last = cell_planes
    per_slab = PLANES_PER_SLAB // CELL
    starts = list(range(first, last, per_slab))
    if not starts:
        raise ValueError(f"no cell planes between {first} and {last}")
    total, gpu = 0, 0.0
    with ThreadPoolExecutor(max_workers=1) as pool:
        ahead = pool.submit(read_slab, store, starts[0], min(starts[0] + per_slab, last))
        for index, plane0 in enumerate(starts):
            read = ahead.result()
            if index + 1 < len(starts):
                ahead = pool.submit(read_slab, store, starts[index + 1], min(starts[index + 1] + per_slab, last))
            nodes, cells, seconds = slab_nodes(store, read, plane0, min(plane0 + per_slab, last), grid, first)
            nodes.tofile(sink_nodes)
            cells.tofile(sink_cells)
            total, gpu = total + len(nodes), gpu + seconds
    return total, gpu
