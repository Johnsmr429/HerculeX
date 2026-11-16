"""Normals turned to face out of the body by one minimum cut."""

import numpy as np
from scipy import ndimage

from unwrapping.level.kernel_bindings import (
    block_pairs,
    blocks,
    domains,
    two_labels,
)

CELL_VOXELS = 16.0
BLOCK_VOXELS = 128.0
PRIOR = 0.01
UNITS = 4096
JOINED_VOXELS = 16.0


def body_distance(x, y, close=5):
    """Distance from outside air on a coarse grid of the section, in cells, with its origin."""
    x0, y0 = x.min() - 8 * CELL_VOXELS, y.min() - 8 * CELL_VOXELS
    i, j = ((x - x0) / CELL_VOXELS).astype(np.int32), ((y - y0) / CELL_VOXELS).astype(np.int32)
    grid = np.zeros((j.max() + 10, i.max() + 10), dtype=bool)
    grid[j, i] = True
    body = ndimage.binary_fill_holes(
        ndimage.binary_closing(grid, iterations=close, structure=ndimage.generate_binary_structure(2, 2))
    )
    return ndimage.distance_transform_edt(body), x0, y0


def air_facing(nodes):
    """Each node's facing along the way out of the body, section by section."""
    n = np.stack([nodes["nx"], nodes["ny"]], axis=1).astype(np.float32)
    n /= np.maximum(np.sqrt((n * n).sum(axis=1) + nodes["nz"].astype(np.float32) ** 2)[:, None], 1e-6)
    facing = np.zeros(len(nodes), dtype=np.float32)
    part = (nodes["z"] // BLOCK_VOXELS).astype(np.int64)
    for b in np.unique(part):
        mine = np.flatnonzero(part == b)
        x, y = nodes["x"][mine].astype(np.float64), nodes["y"][mine].astype(np.float64)
        d, x0, y0 = body_distance(x, y)
        gy, gx = np.gradient(ndimage.gaussian_filter(d, 2.0))
        i, j = ((x - x0) / CELL_VOXELS).astype(np.int32), ((y - y0) / CELL_VOXELS).astype(np.int32)
        ox, oy = -gx[j, i], -gy[j, i]
        size = np.hypot(ox, oy)
        facing[mine] = (n[mine, 0] * ox + n[mine, 1] * oy) / np.maximum(size, 1e-6) * np.minimum(size / 0.5, 1)
    return facing


def turned(nodes, runs):
    """Nodes to turn over, and a report."""
    facing = air_facing(nodes)
    block, count = blocks(nodes, runs, JOINED_VOXELS)
    unary = np.rint(np.bincount(block, weights=facing, minlength=count) * (-PRIOR * UNITS)).astype(np.int64)
    del facing
    flip = two_labels(count, *block_pairs(block, count, nodes, runs, UNITS), unary)[block]
    del block, unary
    keep, kept, opposed = domains(nodes, runs, flip.astype(np.uint8))
    return keep, {
        "opposed_before": opposed,
        "turned_share": float(keep.mean()),
        "domains": int(kept),
        "cut_share": float(flip.mean()),
        "blocks": int(count),
    }
