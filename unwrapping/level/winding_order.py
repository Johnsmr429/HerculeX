"""The curl of a unit: the sense of winding that costs least over its quick pieces, solved exactly over them."""

import numpy as np

from unwrapping.level.l1_synchronization import cost_of, synchronize
from unwrapping.level.sheet_graph import QUALITY_SPAN, RADIAL_RANKS

MIN_ORDERED_NODES = 400
RAY_WEIGHT = 1.0


def votes(nodes, root, potential, member, between, pairs, sign, chirality):
    """Integer relations between pieces."""
    atoms = np.unique(root[member])
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[atoms] = np.arange(len(atoms), dtype=np.int32)
    a, b = between["a"], between["b"]
    keep = member[a] & member[b] & (root[a] != root[b])
    a, b, step, weight = a[keep], b[keep], between["step"][keep], between["weight"][keep]
    along = (
        lookup[root[a]],
        lookup[root[b]],
        potential[b] - potential[a] - step,
        weight,
        np.zeros(len(a), dtype=np.int8),
    )
    first = pairs if RADIAL_RANKS == 1 else pairs[pairs["rank"] == 1]
    u, v = first["a"], first["b"]
    keep = member[u] & member[v] & (root[u] != root[v])
    u, v = u[keep], v[keep]
    del keep
    reversed_ray = sign[u] < 0
    u, v = np.where(reversed_ray, v, u), np.where(reversed_ray, u, v)
    del reversed_ray
    reference = float(np.median(nodes["prominence"]))
    strength = np.minimum(nodes["prominence"][u], nodes["prominence"][v]).astype(np.float32) / (
        2 * QUALITY_SPAN * reference
    )
    radial = (
        lookup[root[u]],
        lookup[root[v]],
        potential[v] - potential[u] - chirality,
        strength.clip(0, 1) * RAY_WEIGHT,
        np.full(len(u), chirality, dtype=np.int8),
    )
    join = [np.concatenate([x, y]) for x, y in zip(along, radial)]
    return (
        atoms,
        join[0].astype(np.int32),
        join[1].astype(np.int32),
        join[2].astype(np.int32),
        join[3].astype(np.float32),
        join[4],
    )


def best_curl(nodes, root, potential, between, pairs, sign):
    """The curl that costs least, the cost of each, and that curl's (offsets, groups, atoms, terms), with which pieces are ordered."""
    sizes = np.bincount(root, minlength=len(nodes))
    member = sizes[root] >= MIN_ORDERED_NODES
    del sizes
    results = {}
    for chirality in (1, -1):
        atoms, a, b, k, w, bound = votes(nodes, root, potential, member, between, pairs, sign, chirality)
        total_weight = float(w.sum(dtype=np.float64))
        if total_weight == 0:
            results[chirality] = (0.0, np.zeros(len(atoms)), np.arange(len(atoms), dtype=np.int32), atoms, 0)
            continue
        f, labels, terms = synchronize(a, b, k, w, len(atoms), bound, np.zeros(len(atoms), dtype=np.int64))
        results[chirality] = (
            cost_of(np.round(f), a, b, k, w.astype(np.float64), bound) / total_weight,
            f,
            labels,
            atoms,
            terms,
        )
    chirality = min(results, key=lambda c: results[c][0])
    return chirality, {c: results[c][0] for c in results}, (*results[chirality][1:], member)
