"""Nearest points on the same sheet: within a depth along a normal and a reach across it, streamed in z slabs."""

import numpy as np
from scipy.spatial import cKDTree

CHUNK = 200_000
WORKERS = 4


def on_sheet(source, query, normals, depth, lateral, keep=1, normal_of="query", k=16):
    """Up to `keep` source points per query point lying within `depth` along the normal and `lateral` across it,
    shallowest first: index (n, keep) int64 (-1 where none) and depth (n, keep) float32 (inf where none).
    `normals` belong to the query points (normal_of='query') or to the source points (normal_of='source');
    non-finite points or normals never match."""
    n = len(query)
    index = np.full((n, keep), -1, dtype=np.int64)
    found_depth = np.full((n, keep), np.inf, dtype=np.float32)
    if not len(source) or not n:
        return index, found_depth
    reach = float(np.hypot(depth, lateral))
    finite = np.all(np.isfinite(source), axis=1)
    source_order = np.argsort(np.where(finite, source[:, 2], np.inf), kind="stable")[: int(finite.sum())]
    del finite
    source_z = source[source_order, 2]
    usable = np.all(np.isfinite(query), axis=1)
    usable &= np.all(np.isfinite(normals), axis=1) if normal_of == "query" else True
    query_order = np.argsort(np.where(usable, query[:, 2], np.inf), kind="stable")[: int(usable.sum())]
    for start in range(0, len(query_order), CHUNK):
        rows = query_order[start : start + CHUNK]
        q = query[rows].astype(np.float64)
        low, high = np.searchsorted(source_z, [q[:, 2].min() - reach, q[:, 2].max() + reach])
        window = source_order[low:high]
        flat = source[window, :2]
        window = window[np.all((flat >= q[:, :2].min(axis=0) - reach) & (flat <= q[:, :2].max(axis=0) + reach), axis=1)]
        if not len(window):
            continue
        count = min(k, len(window))
        distance, found = cKDTree(source[window].astype(np.float64)).query(
            q, k=count, distance_upper_bound=reach, workers=WORKERS
        )
        distance, found = distance.reshape(len(q), count), found.reshape(len(q), count)
        hit = np.isfinite(distance)
        candidate = window[np.where(hit, found, 0)]
        offset = q[:, None, :] - source[candidate]
        direction = normals[rows][:, None, :] if normal_of == "query" else normals[candidate]
        along = np.abs((offset * direction).sum(axis=2))
        across = np.maximum((offset**2).sum(axis=2) - along**2, 0.0)
        ok = hit & np.isfinite(along) & (along <= depth) & (across <= lateral**2)
        along = np.where(ok, along, np.inf)
        best = np.argsort(along, axis=1, kind="stable")[:, :keep]
        picked = np.take_along_axis(along, best, axis=1)
        width = picked.shape[1]
        index[rows, :width] = np.where(np.isfinite(picked), np.take_along_axis(candidate, best, axis=1), -1)
        found_depth[rows, :width] = picked
    return index, found_depth
