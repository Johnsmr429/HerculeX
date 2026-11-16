"""Winding offsets of pieces by L1 synchronisation over their joins."""

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from unwrapping.level.piece_kernel_bindings import library

LENIENCY = 0.5
MIN_TERM = 1e-6
WINDOW_ZONES = 4
REPAIR_PASSES = 8
SCALE = 1024
PIVOT_LIMIT = 2_000_000_000


def aggregate(a, b, k, weight, bound, count, chunk=4_000_000):
    """Votes with the same atoms, offset and kind are one term of summed weight, keyed as one integer, a chunk at a time."""
    low = int(k.min())
    span = int(k.max()) - low + 1
    keys, totals = [], []
    for start in range(0, len(a), chunk):
        part = slice(start, start + chunk)
        key = ((a[part].astype(np.int64) * count + b[part]) * span + (k[part].astype(np.int64) - low)) * 3 + (
            bound[part].astype(np.int64) + 1
        )
        unique, inverse = np.unique(key, return_inverse=True)
        keys.append(unique)
        totals.append(np.bincount(inverse, weights=weight[part], minlength=len(unique)))
    unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    total = np.bincount(inverse, weights=np.concatenate(totals), minlength=len(unique))
    keep = total > MIN_TERM
    unique, total = unique[keep], total[keep]
    kind = unique % 3 - 1
    rest = unique // 3
    return np.stack([rest // span // count, rest // span % count, rest % span + low, kind], axis=1), total


def solve_offsets(count, a, b, k, w, bound):
    """The exact optimum."""
    above = np.where(bound < 0, LENIENCY, 1.0) * w
    below = np.where(bound > 0, LENIENCY, 1.0) * w
    tail = np.ascontiguousarray(np.concatenate([a, b]), dtype=np.int32)
    head = np.ascontiguousarray(np.concatenate([b, a]), dtype=np.int32)
    cost = np.ascontiguousarray(np.concatenate([k, -k]), dtype=np.int32)
    capacity = np.ascontiguousarray(np.round(np.concatenate([above, below]) * SCALE), dtype=np.int64)
    used = capacity > 0
    tail, head, cost, capacity = (np.ascontiguousarray(v[used]) for v in (tail, head, cost, capacity))
    potentials = np.zeros(count + 1, dtype=np.int64)
    pivots = library().hx_network_simplex(
        count,
        len(tail),
        tail.ctypes.data,
        head.ctypes.data,
        cost.ctypes.data,
        capacity.ctypes.data,
        potentials.ctypes.data,
        PIVOT_LIMIT,
    )
    if pivots < 0:
        raise RuntimeError("network simplex hit its pivot limit")
    return potentials[:count].astype(np.float64)


def cost_of(f, a, b, k, w, bound):
    """The objective at offsets f."""
    r = f[b] - f[a] - k
    return float(
        (w * np.where(r > 0, np.where(bound < 0, LENIENCY, 1.0) * r, -np.where(bound > 0, LENIENCY, 1.0) * r)).sum()
    )


def solve_subset(free, ta, tb, tk, w, tbound, f):
    """Re-solve the atoms in `free` exactly, holding all others at f through one ground node."""
    count = int(free.sum())
    local = np.full(len(free), -1, dtype=np.int64)
    local[free] = np.arange(count)
    both = free[ta] & free[tb]
    out_a = free[ta] & ~free[tb]
    out_b = ~free[ta] & free[tb]
    ground = count
    a = np.concatenate([local[ta[both]], local[ta[out_a]], np.full(int(out_b.sum()), ground)])
    b = np.concatenate([local[tb[both]], np.full(int(out_a.sum()), ground), local[tb[out_b]]])
    k = np.concatenate([tk[both], tk[out_a] - f[tb[out_a]], tk[out_b] + f[ta[out_b]]])
    weight = np.concatenate([w[both], w[out_a], w[out_b]])
    bound = np.concatenate([tbound[both], tbound[out_a], tbound[out_b]])
    keep = a != b
    terms, total = aggregate(a[keep], b[keep], k[keep], weight[keep], bound[keep], count + 1)
    solved = solve_offsets(
        count + 1, terms[:, 0], terms[:, 1], terms[:, 2].astype(np.float64), total, terms[:, 3].astype(np.float64)
    )
    return solved[:count] - solved[ground]


def windows_of(zone, shift):
    return (zone + shift) // WINDOW_ZONES


def synchronize(a, b, k, weight, count, bound, zone):
    """Offsets f minimising sum weight * |f[b] - f[a] - k|, a bound vote costing LENIENCY on its permitted side."""
    terms, w = aggregate(a, b, k, weight, bound, count)
    ta, tb, tk, tbound = terms[:, 0], terms[:, 1], terms[:, 2].astype(np.float64), terms[:, 3].astype(np.float64)
    graph = sparse.coo_matrix((np.ones(len(w), dtype=np.int8), (ta, tb)), shape=(count, count))
    _, labels = csgraph.connected_components(graph, directed=False)
    window = windows_of(zone, 0)
    if len(np.unique(window)) == 1:
        return solve_offsets(count, ta, tb, tk, w, tbound), labels, len(w)
    f = np.zeros(count)
    for value in np.unique(window):
        free = window == value
        inner = free[ta] & free[tb]
        local = np.full(count, -1, dtype=np.int64)
        local[free] = np.arange(int(free.sum()))
        f[free] = solve_offsets(int(free.sum()), local[ta[inner]], local[tb[inner]], tk[inner], w[inner], tbound[inner])
    same = window[ta] == window[tb]
    inner_graph = sparse.coo_matrix(
        (np.ones(int(same.sum()), dtype=np.int8), (ta[same], tb[same])), shape=(count, count)
    )
    _, block = csgraph.connected_components(inner_graph, directed=False)
    cross = block[ta] != block[tb]
    stitched, total = aggregate(
        block[ta[cross]],
        block[tb[cross]],
        (tk[cross] + f[ta[cross]] - f[tb[cross]]),
        w[cross],
        tbound[cross],
        int(block.max()) + 1,
    )
    g = solve_offsets(
        int(block.max()) + 1,
        stitched[:, 0],
        stitched[:, 1],
        stitched[:, 2].astype(np.float64),
        total,
        stitched[:, 3].astype(np.float64),
    )
    f = f + g[block]
    best = cost_of(f, ta, tb, tk, w, tbound)
    step = max(WINDOW_ZONES // 2, 1)
    for repair in range(REPAIR_PASSES):
        shifted = windows_of(zone, step * (repair % 2 + 1) if WINDOW_ZONES > 2 else repair % 2 + 1)
        for value in np.unique(shifted):
            free = shifted == value
            f[free] = solve_subset(free, ta, tb, tk, w, tbound, f)
        now = cost_of(f, ta, tb, tk, w, tbound)
        if best - now < 1e-5 * best:
            break
        best = now
    return f, labels, len(w)
