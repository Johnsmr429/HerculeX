"""A mesh grid given a low-distortion, near-isometric flattening: laid flat by its 3D edge lengths, resampled regularly.

A piece's tifxyz grid is its flattening: node (r, c) sits at (c, r) x step in the image. Our unwraps' grids are not
isometric (neighbouring nodes 0.6 to 1.8 steps apart in 3D), so the image stretches and squeezes the papyrus. Here the
valid nodes are laid out in 2D so that every edge (rows, columns and both diagonals of each cell) is as long as in 3D:
stress majorization (each node moved to the mean of where its neighbours would put it at their true distance),
started from the grid itself so the layout keeps its orientation and does not fold. The 3D surface is then
resampled on a regular grid in that layout, at the same step, by linear interpolation inside the old grid's triangles,
so a new node is a point on the old mesh and the image of the new grid is isometric to within the mesh's own
non-developability.
"""

import numpy as np

ITERATIONS = 3000
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def edges_of(good):
    """Index pairs of neighbouring valid nodes (rows, columns, both diagonals) in a grid's valid-node numbering."""
    h, w = good.shape
    index = np.full((h, w), -1)
    index[good] = np.arange(good.sum())
    pairs = []
    for dr, dc in OFFSETS:
        r0, r1 = 0, h - dr
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = index[r0:r1, c0:c1]
        b = index[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        keep = (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[keep], b[keep]], axis=1))
    return index, np.concatenate(pairs)


def laid_flat(points, edges, start, iterations=ITERATIONS):
    """2D positions of points whose edges keep their 3D lengths (stress majorization from start), and the history of
    the median relative length error."""
    target = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    q = start.astype(np.float64).copy()
    degree = np.bincount(edges.ravel(), minlength=len(q)).astype(np.float64)
    history = []
    for k in range(iterations):
        d = q[edges[:, 0]] - q[edges[:, 1]]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        unit = d / length[:, None]
        pull = np.zeros_like(q)
        np.add.at(pull, edges[:, 0], q[edges[:, 1]] + target[:, None] * unit)
        np.add.at(pull, edges[:, 1], q[edges[:, 0]] - target[:, None] * unit)
        q = pull / degree[:, None]
        if k % 500 == 0 or k == iterations - 1:
            history.append(round(float(np.median(np.abs(length / target - 1))), 4))
    return q, history


def stretch(xyz, step):
    """Each neighbouring pair's 3D distance over the grid step, along rows and columns, for valid nodes."""
    good = xyz[0] > 0
    out = []
    for axis in (1, 2):
        n = xyz.shape[axis]
        a, b = np.take(xyz, range(n - 1), axis=axis), np.take(xyz, range(1, n), axis=axis)
        ga, gb = np.take(good, range(n - 1), axis=axis - 1), np.take(good, range(1, n), axis=axis - 1)
        out.append(np.linalg.norm(b - a, axis=0)[ga & gb] / step)
    return np.concatenate(out)


def isometric(xyz, step, iterations=ITERATIONS):
    """(new (3, H', W') grid at the same step, flat position (2, H, W) of each old node in new-grid steps, report)."""
    good = xyz[0] > 0
    _, edges = edges_of(good)
    linked = np.zeros(int(good.sum()), bool)
    linked[edges.ravel()] = True
    isolated = int((~linked).sum())  # nodes with no valid neighbour: nothing holds them in the layout, so left out
    if isolated:
        keep = np.zeros_like(good)
        keep[good] = linked
        good = keep
        _, edges = edges_of(good)
    rows, cols = np.nonzero(good)
    points = xyz[:, good].T.astype(np.float64)
    start = np.stack([cols, rows], axis=1).astype(np.float64) * step
    flat, history = laid_flat(points, edges, start, iterations)
    # keep the start's orientation: the best rotation of the layout back onto the grid (Kabsch, no reflection)
    a, b = flat - flat.mean(0), start - start.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    flat = a @ rot + start.mean(0)
    low = flat.min(0)
    flat = (flat - low) / step  # in new-grid steps
    w, h = int(np.floor(flat[:, 0].max())) + 1, int(np.floor(flat[:, 1].max())) + 1
    new = resampled(xyz, good, flat, (h, w))
    position = np.full((2, *good.shape), np.nan, np.float32)
    position[:, good] = flat.T
    before, after = stretch(xyz, step), stretch(new, step)
    report = {
        "iterations": iterations,
        "isolated_nodes_left_out": isolated,
        "median_edge_error_by_500": history,
        "grid_before": list(good.shape),
        "grid_after": [h, w],
        "stretch_before_p5_p50_p95": np.percentile(before, [5, 50, 95]).round(3).tolist(),
        "stretch_after_p5_p50_p95": np.percentile(after, [5, 50, 95]).round(3).tolist(),
        "within_10pct_before": round(float(np.mean(np.abs(before - 1) <= 0.1)), 3),
        "within_10pct_after": round(float(np.mean(np.abs(after - 1) <= 0.1)), 3),
        "nodes_after": int((new[0] > 0).sum()),
    }
    return new, position, report


def resampled(xyz, good, flat, shape):
    """The old mesh sampled at every integer point of the layout: each point inside one of the old grid's triangles
    (two per cell, laid flat, folded ones skipped) takes the barycentric mix of that triangle's three nodes."""
    h, w = shape
    index = np.full(good.shape, -1)
    index[good] = np.arange(good.sum())
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    tris = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    tris = tris[(tris >= 0).all(1)]
    p = flat[tris]  # (t, 3, 2)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = det > 1e-9  # same winding as the grid: not folded
    tris, p, e1, e2, det = tris[keep], p[keep], e1[keep], e2[keep], det[keep]
    lo = np.floor(p.min(1)).astype(int)
    span = int(np.ceil((p.max(1) - p.min(1)).max())) + 2
    points = xyz[:, good].T.astype(np.float64)
    new = np.full((h, w, 3), -1.0)
    for dy in range(span):
        for dx in range(span):
            q = (lo + np.array([dx, dy])).astype(np.float64)
            r = q - p[:, 0]
            l1 = (r[:, 0] * e2[:, 1] - r[:, 1] * e2[:, 0]) / det
            l2 = (e1[:, 0] * r[:, 1] - e1[:, 1] * r[:, 0]) / det
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
            inside &= (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < w) & (q[:, 1] < h)
            t = np.flatnonzero(inside)
            mix = ((1 - l1[t] - l2[t])[:, None] * points[tris[t, 0]] + l1[t, None] * points[tris[t, 1]]
                   + l2[t, None] * points[tris[t, 2]])  # fmt: skip
            new[q[t, 1].astype(int), q[t, 0].astype(int)] = mix
    return new.transpose(2, 0, 1).astype(np.float32)
