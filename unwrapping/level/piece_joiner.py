"""Pieces of neighbouring chunks joined across their seam by the sheet edges that cross it, unless certain rays say they lie stacked."""

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from unwrapping.level import scroll_axis
from unwrapping.level.piece_growth import PIECE_DEPTH, node_quality, ray_weights
from unwrapping.level.sheet_graph import (
    build_edges,
    filtered,
    load_field,
    radial_pairs,
)

SEAM_PLANES = 4
MIN_EDGES = 3
STACK_SHARE = 0.05
SPREAD_STEPS = 3


def boundary_labels(unit):
    """A unit's node ids in its shared plane ranges, sorted, with each node's piece."""
    data = np.load(unit / "boundary.npz")
    order = np.argsort(data["ids"])
    return data["ids"][order], data["piece"][order]


def seam_field(chunks, seam, labels):
    """The nodes within SEAM_PLANES of a seam."""
    nodes, cells, _record, grid, ids = load_field(chunks, (seam - SEAM_PLANES, seam + SEAM_PLANES))
    nodes, first, cell_of, ids = filtered(nodes, cells, grid, ids)
    knots = scroll_axis.axis(nodes, float(nodes["z"].min()), float(nodes["z"].max()) + 1)
    centre = scroll_axis.centre_at(knots, nodes["z"])
    theta = np.arctan2(nodes["y"] - centre[1], nodes["x"] - centre[0]).astype(np.float32)
    quality = node_quality(nodes)
    used = np.nonzero(np.diff(first).reshape(grid).any(axis=(1, 2)))[0].tolist()
    edges = build_edges(nodes, first, grid, theta, quality, used, max_depth=PIECE_DEPTH)
    pairs = radial_pairs(nodes, first, grid, seam - SEAM_PLANES, np.ones(len(nodes), dtype=bool))
    certainty = ray_weights(nodes, pairs, quality)
    side = (cell_of // (grid[1] * grid[2]) >= SEAM_PLANES).astype(np.int64)
    piece = np.full(len(nodes), -1, dtype=np.int64)
    for s, (known_ids, known_piece) in enumerate(labels):
        here = side == s
        index = np.searchsorted(known_ids, ids[here])
        index = np.minimum(index, len(known_ids) - 1)
        found = known_ids[index] == ids[here]
        piece[np.flatnonzero(here)[found]] = known_piece[index[found]]
    a, b = edges["a"], edges["b"]
    same_side = side[a] == side[b]
    for _ in range(SPREAD_STEPS):
        for u, v in ((a, b), (b, a)):
            take = same_side & (piece[u] < 0) & (piece[v] >= 0)
            piece[u[take]] = piece[v[take]]
    return piece, side, edges, pairs, certainty, ids, nodes


def joins_of(piece, side, edges, pairs, certainty):
    """(lower piece, upper piece, edges) of every pair of pieces with enough crossing sheet edges and too little certain stacking between them."""
    a, b = edges["a"], edges["b"]
    across = (side[a] != side[b]) & (piece[a] >= 0) & (piece[b] >= 0)
    if not across.any():
        return []
    lower = np.where(side[a] == 0, piece[a], piece[b])[across]
    upper = np.where(side[a] == 0, piece[b], piece[a])[across]
    ra, rb = piece[pairs["a"]], piece[pairs["b"]]
    cross = (side[pairs["a"]] != side[pairs["b"]]) & (ra >= 0) & (rb >= 0)
    ray_lower = np.where(side[pairs["a"]] == 0, ra, rb)[cross]
    ray_upper = np.where(side[pairs["a"]] == 0, rb, ra)[cross]
    span = int(max(upper.max(), ray_upper.max(initial=0))) + 1
    keys, counts = np.unique(lower * span + upper, return_counts=True)
    ray_keys, inverse = np.unique(ray_lower * span + ray_upper, return_inverse=True)
    stacking = np.bincount(inverse, weights=certainty[cross], minlength=len(ray_keys))
    weight = np.zeros(len(keys))
    if len(ray_keys):
        found = np.minimum(np.searchsorted(ray_keys, keys), len(ray_keys) - 1)
        hit = ray_keys[found] == keys
        weight[hit] = stacking[found[hit]]
    keep = (counts >= MIN_EDGES) & (weight <= STACK_SHARE * counts)
    order = np.argsort(-counts[keep], kind="stable")
    return [(int(k // span), int(k % span), int(n)) for k, n in zip(keys[keep][order], counts[keep][order])]


def seam_joins(chunks, seam, labels):
    """(lower piece, upper piece, edges) of every join across one seam."""
    return joins_of(*seam_field(chunks, seam, labels)[:5])


def join(counts, seams):
    """Global sheet of every (unit, piece)."""
    offsets = np.cumsum([0, *counts])
    a, b = [], []
    for k, joins in enumerate(seams):
        for x, y, _ in joins:
            a.append(offsets[k] + x)
            b.append(offsets[k + 1] + y)
    graph = sparse.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(int(offsets[-1]),) * 2)
    return csgraph.connected_components(graph, directed=False)[1]
