"""Grid surfaces shared by references and predictions."""

from collections import deque

import numpy as np


def grid_normals(x, y, z, valid):
    """Unit normals from central differences of valid neighbours, one-sided at edges."""
    p = np.stack([x, y, z], axis=-1).astype(np.float64)
    tangents = []
    for axis in (1, 0):
        forward = np.zeros_like(p)
        ok_f = np.zeros(valid.shape, dtype=bool)
        backward = np.zeros_like(p)
        ok_b = np.zeros(valid.shape, dtype=bool)
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        forward[lo] = p[hi] - p[lo]
        ok_f[lo] = valid[hi] & valid[lo]
        backward[hi] = p[hi] - p[lo]
        ok_b[hi] = valid[hi] & valid[lo]
        t = np.where((ok_f & ok_b)[..., None], (forward + backward) / 2, np.where(ok_f[..., None], forward, backward))
        tangents.append((t, ok_f | ok_b))
    normal = np.cross(tangents[0][0], tangents[1][0])
    length = np.linalg.norm(normal, axis=-1)
    good = valid & tangents[0][1] & tangents[1][1] & (length > 1e-9)
    normal = np.where(good[..., None], normal / np.maximum(length, 1e-12)[..., None], 0.0)
    return normal.astype(np.float32), good


class Axis:
    """Centre of the scroll per z slab."""

    def __init__(self, knots):
        self.knots = np.asarray(knots, dtype=np.float64)

    def centre(self, z):
        return np.interp(z, self.knots[:, 0], self.knots[:, 1]), np.interp(z, self.knots[:, 0], self.knots[:, 2])

    def theta(self, x, y, z):
        cx, cy = self.centre(z)
        return np.arctan2(y - cy, x - cx)

    def radius(self, x, y, z):
        cx, cy = self.centre(z)
        return np.hypot(x - cx, y - cy)


def row_runs(valid):
    """(row, first column, end column) of every run of valid cells along each row."""
    runs = []
    for r in range(valid.shape[0]):
        edges = np.diff(np.concatenate([[0], valid[r].astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            runs.append((r, int(s), int(e)))
    return runs


def unwrap_grid(theta, valid):
    """Unwrapped angle over each connected part of a grid surface, and the part of every cell (-1 where invalid)."""
    phi = np.full(theta.shape, np.nan, dtype=np.float64)
    part = np.full(theta.shape, -1, dtype=np.int64)
    runs = row_runs(valid)
    if not runs:
        return phi, part
    by_row = {}
    for i, (r, s, e) in enumerate(runs):
        phi[r, s:e] = np.unwrap(theta[r, s:e].astype(np.float64))
        by_row.setdefault(r, []).append(i)
    neighbours = [[] for _ in runs]
    for i, (r, s, e) in enumerate(runs):
        for j in by_row.get(r + 1, []):
            _, s2, e2 = runs[j]
            lo, hi = max(s, s2), min(e, e2)
            if hi > lo:
                turns = int(np.round(np.median(phi[r, lo:hi] - phi[r + 1, lo:hi]) / (2 * np.pi)))
                neighbours[i].append((j, turns))
                neighbours[j].append((i, -turns))
    seen = np.zeros(len(runs), dtype=bool)
    shift = np.zeros(len(runs), dtype=np.int64)
    order = sorted(range(len(runs)), key=lambda i: runs[i][1] - runs[i][2])
    label = 0
    for start in order:
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            i = queue.popleft()
            r, s, e = runs[i]
            part[r, s:e] = label
            for j, turns in neighbours[i]:
                if not seen[j]:
                    seen[j] = True
                    shift[j] = shift[i] + turns
                    queue.append(j)
        label += 1
    for i, (r, s, e) in enumerate(runs):
        phi[r, s:e] += 2 * np.pi * shift[i]
    return phi, part
