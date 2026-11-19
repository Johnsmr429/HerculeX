"""What the level solve starts from: a field's sheet edges, normals turned to agree along them, and patches merged
into quick pieces over the sheet graph."""

import os

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
RAY_MODEL = 0
RAY_MODEL_SCALE = 1.0
RAY_MODEL_FILE = os.environ.get("HERCULEX_RAY_MODEL", "")


def ray_weights(nodes, pairs, quality, rim=RAY_RIM):
    """Confidence that each ray meets a different sheet."""
    length = np.sqrt(sum((nodes[c][pairs["b"]].astype(np.float64) - nodes[c][pairs["a"]]) ** 2 for c in "xyz"))
    gap = float(np.median(length[length > 0.6 * np.median(length)]))
    sure = np.clip((length / gap - rim) / (RAY_SURE - rim), 0, 1)
    return (sure * np.minimum(quality[pairs["a"]], quality[pairs["b"]])).astype(np.float32)


def ray_features(nodes, pairs, degree):
    """Each ray's features for the ray model, from its two nodes and their sheet edges."""
    a, b = pairs["a"], pairs["b"]
    xyz = [nodes[c].astype(np.float64) for c in "xyz"]
    d = np.stack([xyz[k][b] - xyz[k][a] for k in range(3)], axis=1)
    n = np.stack([nodes[c].astype(np.float64) for c in ("nx", "ny", "nz")], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    length = np.linalg.norm(d, axis=1)
    gap = float(np.median(length[length > 0.6 * np.median(length)]))
    along = (d * n[a]).sum(axis=1)
    return {
        "log_length_over_gap": np.log(np.maximum(length / gap, 1e-3)),
        "degree_min": np.minimum(degree[a], degree[b]).astype(np.float64),
        "normal_dot": np.abs((n[a] * n[b]).sum(axis=1)),
        "coherence_min": np.minimum(nodes["coherence"][a], nodes["coherence"][b]) / 255.0,
        "prominence_a": nodes["prominence"][a] / 2.0,
        "prominence_b": nodes["prominence"][b] / 2.0,
        "lateral": np.sqrt(np.maximum(length**2 - along**2, 0)),
        "width_a": nodes["width"][a] / 4.0,
        "width_b": nodes["width"][b] / 4.0,
    }


def logistic(features, model, prefix):
    """A fitted logistic model's probability for every ray."""
    names = [str(n) for n in model[f"{prefix}_names"]]
    z = np.full(len(next(iter(features.values()))), float(model[f"{prefix}_intercept"]))
    for name, mean, scale, weight in zip(
        names, model[f"{prefix}_mean"], model[f"{prefix}_scale"], model[f"{prefix}_weights"], strict=True
    ):
        z += weight * (features[name] - mean) / scale
    return 1.0 / (1.0 + np.exp(-z))


def ray_costs(nodes, pairs, quality, edges):
    """Each ray's cost of meeting its own sheet again (its weight) and of skipping sheets (its up cost): the ray model's
    odds against each, times the ray's quality."""
    model = np.load(RAY_MODEL_FILE)
    degree = np.bincount(edges["a"], minlength=len(nodes)) + np.bincount(edges["b"], minlength=len(nodes))
    features = ray_features(nodes, pairs, degree)
    same, skip = logistic(features, model, "same"), logistic(features, model, "skip")
    q = RAY_MODEL_SCALE * np.minimum(quality[pairs["a"]], quality[pairs["b"]])
    return ((1 - same) * q).astype(np.float32), ((1 - skip) * q).astype(np.float32)


def field_rays(nodes, pairs, quality, edges, rim):
    """Rays' weights and skip costs (None: hidden times the weight) for a solve."""
    if RAY_MODEL:
        return ray_costs(nodes, pairs, quality, edges)
    return ray_weights(nodes, pairs, quality, rim), None


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
