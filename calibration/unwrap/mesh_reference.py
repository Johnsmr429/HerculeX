"""Reference vertices of one band from the published meshes."""

import json
import re
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

VOXEL_UM = 7.91
TOLERANCE_UM = 40.0
LATERAL_UM = 160.0
GROUP_PAIRS = 200
LINK_PAIRS = 1000
LINK_SHARE = 0.2
LINK_AGREEMENT = 0.95
TIER_A = [
    "20230702185753",
    "20230929220926",
    "20231005123336",
    "20231007101619",
    "20231012184424",
    "20231016151002",
    "20231022170901",
    "20231031143852",
    "20231106155351",
    "20231210121321",
    "20231221180251",
]
EXCLUDED = ["20260926044226-spiral-output-generation-19"]


def named_windings(segment):
    match = re.search(r"-w(\d+)-(\d+)", segment)
    return (int(match.group(1)), int(match.group(2))) if match else None


def points_of(ref):
    return np.stack([ref["x"], ref["y"], ref["z"]], axis=1).astype(np.float64)


def normals_of(ref):
    return np.stack([ref["nx"], ref["ny"], ref["nz"]], axis=1).astype(np.float64)


def neighbours_on_sheet(points, normals, tree, other, tolerance, lateral, k=12, batch=400_000):
    """For each point, (query index, found index) pairs of tree points within the lateral reach and the normal tolerance."""
    reach = float(np.hypot(tolerance, lateral))
    qs, fs = [], []
    for start in range(0, len(points), batch):
        p, n = points[start : start + batch], normals[start : start + batch]
        distance, index = tree.query(p, k=k, distance_upper_bound=reach)
        found = np.isfinite(distance)
        rows, cols = np.nonzero(found)
        j = index[rows, cols]
        offset = other[j] - p[rows]
        depth = np.abs((offset * n[rows]).sum(axis=1))
        side = np.sqrt(np.maximum((offset**2).sum(axis=1) - depth**2, 0))
        ok = (depth <= tolerance) & (side <= lateral)
        qs.append(rows[ok] + start)
        fs.append(j[ok])
    return np.concatenate(qs), np.concatenate(fs)


def coincidence(ref):
    """Vertex pairs of two units lying within the tolerance, and per unit pair the modal integer winding offset, count and agreement."""
    p, n = points_of(ref), normals_of(ref)
    tolerance, lateral = TOLERANCE_UM / VOXEL_UM, LATERAL_UM / VOXEL_UM
    qi, fj = neighbours_on_sheet(p, n, cKDTree(p), p, tolerance, lateral)
    unit = ref["unit"]
    cross = unit[qi] != unit[fj]
    qi, fj = qi[cross], fj[cross]
    units = len(ref["unit_mesh"])
    step = np.round(ref["W"][qi] - ref["W"][fj]).astype(np.int64)
    pair_key = unit[qi].astype(np.int64) * units + unit[fj]
    order_pairs = np.lexsort((step, pair_key))
    keys, starts = np.unique(pair_key[order_pairs], return_index=True)
    ends = np.append(starts[1:], len(order_pairs))
    sorted_step = step[order_pairs]
    edges = {}
    for key, s0, s1 in zip(keys, starts, ends):
        values, counts = np.unique(sorted_step[s0:s1], return_counts=True)
        edges[(int(key // units), int(key % units))] = (
            int(values[np.argmax(counts)]),
            int(s1 - s0),
            float(counts.max() / (s1 - s0)),
        )
    ref["pair_q"], ref["pair_f"] = qi.astype(np.int64), fj.astype(np.int64)
    return edges


def name_conflict(ref, a, b):
    na, nb = (
        named_windings(str(ref["names"][ref["unit_mesh"][a]])),
        named_windings(str(ref["names"][ref["unit_mesh"][b]])),
    )
    return bool(na and nb and not (na[0] <= nb[1] and nb[0] <= na[1]))


def named_offset(ref, u):
    """Whole turns that put a named unit's windings inside its name's range."""
    window = named_windings(str(ref["names"][ref["unit_mesh"][u]]))
    turns = np.floor(ref["W"][ref["unit"] == u])
    best, score = 0, -1
    for c in range(window[0] - int(turns.max()) - 2, window[1] - int(turns.min()) + 3):
        inside = int(((turns + c >= window[0]) & (turns + c <= window[1])).sum())
        if inside > score:
            best, score = c, inside
    return best


def frame(ref, edges, tiers, by_name):
    """Duplicates, groups and group windings of the units of the given tiers."""
    unit = ref["unit"]
    units = len(ref["unit_mesh"])
    member_tier = ref["tiers"][ref["unit_mesh"]]
    inside = np.isin(member_tier, list(tiers))
    sizes = np.bincount(unit, minlength=units)
    rank = np.lexsort((-sizes, member_tier != "A"))
    order = np.empty_like(rank)
    order[rank] = np.arange(units)
    qi, fj = ref["pair_q"], ref["pair_f"]
    usable = inside[unit[qi]] & inside[unit[fj]]
    earlier = usable & (order[unit[fj]] < order[unit[qi]])
    duplicate = np.zeros(len(unit), dtype=bool)
    duplicate[qi[earlier]] = True
    links, flagged = {}, []
    for (a, b), (s, count, agreement) in edges.items():
        if not (inside[a] and inside[b]):
            continue
        if name_conflict(ref, a, b):
            if count >= GROUP_PAIRS:
                flagged.append((a, b, count))
            continue
        if count >= LINK_PAIRS and count >= LINK_SHARE * min(sizes[a], sizes[b]) and agreement >= LINK_AGREEMENT:
            links.setdefault(a, []).append((b, s))
    named = (
        [u for u in range(units) if inside[u] and named_windings(str(ref["names"][ref["unit_mesh"][u]]))]
        if by_name
        else []
    )
    hub = units
    for u in named:
        c = named_offset(ref, u)
        links.setdefault(hub, []).append((u, c))
        links.setdefault(u, []).append((hub, -c))
    group = np.full(units + 1, -1, dtype=np.int64)
    offset = np.zeros(units + 1, dtype=np.int64)
    conflicts, label = [], 0
    for start in ([hub] if named else []) + [u for u in rank if inside[u]]:
        if group[start] >= 0:
            continue
        group[start] = label
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b, s in links.get(a, []):
                if group[b] < 0:
                    group[b] = label
                    offset[b] = offset[a] + s
                    queue.append(b)
                elif offset[b] != offset[a] + s and a < b:
                    conflicts.append((int(a), int(b), int(offset[b] - offset[a] - s)))
        label += 1
    keep = inside[unit] & ~duplicate
    return {
        "keep": keep,
        "group": group[unit].astype(np.int32),
        "Wg": ref["W"] + offset[unit],
        "duplicate": duplicate & inside[unit],
        "conflicts": conflicts,
        "flagged": flagged,
        "groups": label,
    }


def unresolved(ref, key):
    """Vertices with a surface of another sheet (or of an unlinked mesh) within twice the tolerance."""
    p, n = points_of(ref), normals_of(ref)
    inside = np.isin(ref["tiers"][ref["mesh"]], list(key))
    index = np.flatnonzero(inside)
    q, f = neighbours_on_sheet(
        p[index], n[index], cKDTree(p[index]), p[index], 2 * TOLERANCE_UM / VOXEL_UM, LATERAL_UM / VOXEL_UM, k=16
    )
    q, f = index[q], index[f]
    group, wg = ref[f"group_{key}"], ref[f"Wg_{key}"]
    depth = np.abs(((p[f] - p[q]) * n[q]).sum(axis=1))
    other = ((group[q] == group[f]) & (np.abs(wg[q] - wg[f]) > 0.5)) | (
        (group[q] != group[f]) & (depth > TOLERANCE_UM / VOXEL_UM)
    )
    out = np.zeros(len(p), dtype=bool)
    out[q[other]] = True
    return out


def save(path, ref, audit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ref)
    path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=1))


def load(path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
