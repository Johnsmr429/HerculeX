"""One frame for a whole scan: the level of every flattened piece, seam turns compared about one segment where both units took a core segment."""

from pathlib import Path

import numpy as np

from runner import stage_queue
from storage.data_roots import stage_folder
from unwrapping.level import kernel_bindings as kb
from unwrapping.level.piece_flattening import UNIT_NEEDS
from unwrapping.level.piece_joiner import seam_field

UNITS = 64
HIDDEN = 0.5
FIRST_STEP = 64
TAU = 2 * np.pi
CURLS = {1: "plus", -1: "minus"}


def boundary_of(unit):
    """A unit's seam nodes sorted by id."""
    data = np.load(unit / "boundary.npz")
    order = np.argsort(data["ids"])
    return data["ids"][order], data["piece"][order], (data["potential"] + data["theta"] / TAU)[order]


def seam_segment(units, seam_index):
    """The mean of the two units' core segments nearest their seam, or None unless both units took the core segment."""
    rows = []
    for unit, row in ((units[seam_index], -1), (units[seam_index + 1], 0)):
        centres = stage_queue.record_of(unit).get("centres") or {}
        if centres.get("rule") != "segment" or not centres.get("segments"):
            return None
        rows.append(np.array(centres["segments"][row][1:6], dtype=np.float64))
    if rows[0][2:4] @ rows[1][2:4] < 0:
        rows[1][2:4] = -rows[1][2:4]
    mean = (rows[0] + rows[1]) / 2
    mean[2:4] /= max(np.hypot(*mean[2:4]), 1e-9)
    return rows, mean


def angle_about(x, y, segment):
    """The angle of points about their nearest point of a segment (cx, cy, dx, dy, half)."""
    cx, cy, dx, dy, half = segment
    t = np.clip((x - cx) * dx + (y - cy) * dy, -half, half)
    return np.arctan2(y - (cy + t * dy), x - (cx + t * dx))


def seam_crossings(chunks, units, seam_index, first):
    """What crosses a seam between pieces, lower end first."""
    seam = stage_queue.record_of(chunks[seam_index])["cell_planes"][1]
    sides = [boundary_of(units[seam_index]), boundary_of(units[seam_index + 1])]
    _, side, edges, pairs, certainty, ids, nodes = seam_field(
        chunks[seam_index : seam_index + 2], seam, [s[:2] for s in sides]
    )
    shared = seam_segment(units, seam_index)
    own, turns = np.full(len(ids), -1, dtype=np.int64), np.zeros(len(ids))
    for s, (known, labels, along) in enumerate(sides):
        here = np.flatnonzero(side == s)
        index = np.minimum(np.searchsorted(known, ids[here]), len(known) - 1)
        found = (known[index] == ids[here]) & (labels[index] >= 0)
        own[here[found]] = labels[index[found]] + first[seam_index + s]
        turns[here[found]] = along[index[found]]
        if shared is not None:
            x, y = (nodes[c][here[found]].astype(np.float64) for c in "xy")
            moved = angle_about(x, y, shared[1]) - angle_about(x, y, shared[0][s])
            turns[here[found]] += np.angle(np.exp(1j * moved)) / TAU
    parts = []
    for a, b, weight, rise, share in (
        (edges["a"], edges["b"], edges["weight"], 0, 1.0),
        (pairs["a"], pairs["b"], certainty, 1, HIDDEN),
    ):
        across = (side[a] != side[b]) & (own[a] >= 0) & (own[b] >= 0) & (weight > 0)
        a, b, weight = a[across], b[across], weight[across].astype(np.float64)
        swap = side[a] > side[b]
        parts.append(
            (
                np.where(swap, own[b], own[a]),
                np.where(swap, own[a], own[b]),
                np.where(swap, turns[b] - turns[a], turns[a] - turns[b]),
                np.where(swap, -rise, rise),
                np.where(swap, weight, weight * share),
                np.where(swap, weight * share, weight),
            )
        )
    return [np.concatenate(values) for values in zip(*parts)]


def weighted_median(values, weights):
    order = np.argsort(values)
    return values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]


def summed(p, q, k, up, dn):
    """One term per pair of pieces and offset."""
    unique, inverse = np.unique(np.stack([p, q, k], axis=1), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    costs = [
        np.maximum(np.rint(np.bincount(inverse, weights=w, minlength=len(unique)) * UNITS), 1).astype(np.int32)
        for w in (up, dn)
    ]
    return (*[np.ascontiguousarray(unique[:, c], dtype=np.int64) for c in range(3)], *costs)


def fitted(curl, data, first, crossings, levels):
    """The scan's exact levels under one curl."""
    name = CURLS[curl]
    start = levels.copy()
    seams, report = [], []
    for s, (p, q, turns, rise, up, dn) in enumerate(crossings):
        k = np.rint(curl * turns).astype(np.int64) + rise
        shift = int(weighted_median(k + start[p] - start[q], dn)) if len(p) else 0
        start[first[s + 1] :] += shift
        seams.append(summed(p, q, k, up, dn))
        report.append({"crossing": len(p), "shift": shift})
    p_s, q_s, k_s = (np.concatenate([s[c] for s in seams]) for c in range(3))
    up_s, dn_s = (np.concatenate([s[c] for s in seams]) for c in (3, 4))
    p = np.concatenate([d[f"p_{name}"].astype(np.int64) + o for d, o in zip(data, first)] + [p_s]).astype(np.int32)
    q = np.concatenate([d[f"q_{name}"].astype(np.int64) + o for d, o in zip(data, first)] + [q_s]).astype(np.int32)
    k = np.concatenate([d[f"k_{name}"] for d in data] + [k_s]).astype(np.int32)
    up = np.concatenate([d[f"up_{name}"] for d in data] + [up_s]).astype(np.int32)
    dn = np.concatenate([d[f"dn_{name}"] for d in data] + [dn_s]).astype(np.int32)
    x, log = kb.minimize(int(first[-1]), p, q, k, up, dn, start.astype(np.int32), FIRST_STEP)
    x = x.astype(np.int64)

    def held(y):
        return float(np.average((y[q_s] - y[p_s]) == k_s, weights=dn_s))

    moved = x - start
    moved -= int(np.median(moved))
    return (
        x,
        int(log[2]),
        int(log[0]),
        (p_s, q_s, k_s, dn_s),
        {
            "seams": report,
            "terms": len(p),
            "seam kept before": held(start),
            "seam kept": held(x),
            "pieces moved": float((moved != 0).mean()),
        },
    )


def solve(scan, method, same_output=()):
    """A scan's frame, once all its units are done."""
    chunks = sorted(stage_folder(scan, "nodes").glob("rows-*"))
    units = [scan / "pieces" / c.name for c in chunks]
    if len(units) < 2 or not all(stage_queue.done(u, method, same_output, UNIT_NEEDS) for u in units):
        return None
    out = scan / "pieces" / "frame"
    digest = stage_queue.method_of(Path(__file__), *[u / stage_queue.RECORD for u in units])
    if stage_queue.done(out, digest):
        return out
    data = [np.load(u / "levels.npz") for u in units]
    first = np.cumsum([0, *[len(d["level"]) for d in data]])
    grown_with = {int(d["chirality"]) for d in data}
    levels = np.concatenate([d["level"] for d in data]).astype(np.int64)
    crossings = [seam_crossings(chunks, units, s, first) for s in range(len(units) - 1)]
    fits = {
        curl: fitted(curl, data, first, crossings, levels if grown_with == {curl} else np.zeros_like(levels))
        for curl in CURLS
    }
    curl = next(iter(grown_with)) if len(grown_with) == 1 else min(fits, key=lambda c: fits[c][1])
    x, energy, cuts, (p_s, q_s, k_s, weight), report = fits[curl]
    agrees = (x[q_s] - x[p_s]) == k_s
    pairs, inverse = np.unique(p_s * int(first[-1]) + q_s, return_inverse=True)
    inverse = inverse.ravel()
    standing = pairs[
        np.bincount(inverse, weights=weight * agrees, minlength=len(pairs))
        > np.bincount(inverse, weights=weight * ~agrees, minlength=len(pairs))
    ]
    with stage_queue.building(out) as part:
        np.savez(
            part / "levels.npz",
            level=x,
            first=first,
            curl=curl,
            held_p=standing // int(first[-1]),
            held_q=standing % int(first[-1]),
        )
        summary = {
            "units": [u.name for u in units],
            "pieces": int(first[-1]),
            "curl": curl,
            "grown with": sorted(grown_with),
            "energy": energy,
            "energy by curl": {str(c): f[1] for c, f in fits.items()},
            "cuts": cuts,
            "method": digest,
        } | report
        stage_queue.finish(part, summary)
    print(scan.parts[-2], "one frame", {k: v for k, v in summary.items() if k not in ("units", "method")}, flush=True)
    return out
