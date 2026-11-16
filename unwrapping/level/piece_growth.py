"""What the level solve starts from: a field's sheet edges, normals turned to agree along them, and patches merged
into quick pieces over the sheet graph."""

import numpy as np

from unwrapping.level.piece_kernel_bindings import library
from unwrapping.level.sheet_graph import (
    ATOM_PLANES,
    QUALITY_SPAN,
    Forest,
    build_edges,
    file_backed,
    used_planes,
)

PIECE_DEPTH = 2.0


def sheet_edges(nodes, first, grid, theta, quality):
    """Sheet edges of a field, one strongest-first array per run of cell planes."""
    used = used_planes(first, grid)
    return [
        file_backed(
            build_edges(
                nodes,
                first,
                grid,
                theta,
                quality,
                used[(used >= s) & (used < s + ATOM_PLANES)].tolist(),
                max_depth=PIECE_DEPTH,
            )
        )
        for s in range(0, grid[0], ATOM_PLANES)
    ]


def turn_to_sheets(nodes, loose):
    """Normals made to agree along sheet edges, each connected sheet keeping the side most of its nodes face."""
    forest = Forest(len(nodes))
    none = np.zeros(0, dtype=loose[0].dtype)
    stats = np.zeros(4, dtype=np.int64)
    for edges in loose:
        chain = np.empty_like(edges)
        library().hx_turned_steps(edges.ctypes.data, len(edges), nodes.ctypes.data, chain.ctypes.data)
        library().hx_mutex_grow(
            chain.ctypes.data, None, len(chain), none.ctypes.data, 0, len(nodes), *forest.pointers(), stats.ctypes.data
        )
    root, parity = forest.labels()
    odd = parity % 2 == 1
    _ids, sheet = np.unique(root, return_inverse=True)
    majority = np.bincount(sheet, weights=odd) * 2 > np.bincount(sheet)
    turn = odd != majority[sheet]
    for axis in ("nx", "ny", "nz"):
        nodes[axis] = np.where(turn, -nodes[axis], nodes[axis])
    return float(turn.mean())


RAY_SURE = 0.75
RAY_RIM = 0.45


def ray_weights(nodes, pairs, quality, rim=RAY_RIM):
    """Confidence that each ray meets a different sheet."""
    length = np.sqrt(sum((nodes[c][pairs["b"]].astype(np.float64) - nodes[c][pairs["a"]]) ** 2 for c in "xyz"))
    gap = float(np.median(length[length > 0.6 * np.median(length)]))
    sure = np.clip((length / gap - rim) / (RAY_SURE - rim), 0, 1)
    return (sure * np.minimum(quality[pairs["a"]], quality[pairs["b"]])).astype(np.float32)


PATCH_VOXELS = 24
RAY_SCALE = 1.0
NECK = 0.1
NECK_CAP = 1000.0
FIELD_RAY_RIM = 0.6


def same_label(edges, label):
    """A mask of the edges whose two ends share a label."""
    mask = np.empty(len(edges), dtype=bool)
    test = library().hx_same_label_wide if label.dtype == np.int64 else library().hx_same_label
    test(edges.ctypes.data, len(edges), label.ctypes.data, mask.ctypes.data)
    return mask


def potential_agrees(edges, potential):
    """A mask of the edges whose step the potentials hold exactly."""
    mask = np.empty(len(edges), dtype=bool)
    library().hx_potential_agrees(edges.ctypes.data, len(edges), potential.ctypes.data, mask.ctypes.data)
    return mask


def patch_labels(root):
    """Each node's rank among the distinct roots, and their number."""
    present = np.zeros(len(root), dtype=bool)
    present[root] = True
    rank = np.cumsum(present, dtype=np.int32)
    return rank[root] - 1, int(rank[-1])


def patches_of(nodes, edges, use, repel):
    """Each node's patch and winding inside it."""
    block = (
        ((nodes["z"] // PATCH_VOXELS).astype(np.int64) * (1 << 20) + (nodes["y"] // PATCH_VOXELS).astype(np.int64))
        * (1 << 20)
    ) + (nodes["x"] // PATCH_VOXELS).astype(np.int64)
    inside = use & same_label(edges, block)
    del block
    forest = Forest(len(nodes))
    stats = np.zeros(4, dtype=np.int64)
    library().hx_mutex_grow(
        edges.ctypes.data,
        inside.ctypes.data,
        len(edges),
        repel.ctypes.data,
        len(repel),
        len(nodes),
        *forest.pointers(),
        stats.ctypes.data,
    )
    root, inner = forest.labels()
    patch, count = patch_labels(root)
    return patch, count, inner


def settled(nodes, edges, use, label):
    """Forest of the edges inside each label."""
    inside = use & same_label(edges, label)
    forest = Forest(len(nodes))
    none, stats = np.zeros(0, dtype=edges.dtype), np.zeros(4, dtype=np.int64)
    library().hx_mutex_grow(
        edges.ctypes.data,
        inside.ctypes.data,
        len(edges),
        none.ctypes.data,
        0,
        len(nodes),
        *forest.pointers(),
        stats.ctypes.data,
    )
    return forest


def agglomerated(nodes, patch, count, edges, use, repel):
    """Patches merged on the evidence summed between them, as a forest of pieces, and the merge's stats."""
    label, stats = np.zeros(count, dtype=np.int32), np.zeros(4, dtype=np.int64)
    library().hx_agglomerate(
        count,
        patch.ctypes.data,
        len(patch),
        edges.ctypes.data,
        use.ctypes.data,
        len(edges),
        repel.ctypes.data,
        len(repel),
        RAY_SCALE,
        0.0,
        NECK,
        NECK_CAP,
        label.ctypes.data,
        stats.ctypes.data,
    )
    return settled(nodes, edges, use, label[patch]), stats


def node_quality(nodes):
    """How surely each node is a sheet."""
    quality = (nodes["prominence"].astype(np.float32) / (QUALITY_SPAN * float(np.median(nodes["prominence"])))).clip(
        0, 1
    )
    return np.ascontiguousarray(quality * (nodes["coherence"].astype(np.float32) / 255 + 0.25).clip(0, 1))
