"""Levels a unit solve gains along straight lines through the stack, against the sheets counted in the raw scan on the same lines."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ni
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from sampling import column_sampling as sampler
from storage.data_roots import LEVEL_ROUTE, calibration_output, stage_folder
from unwrapping.level import level_pieces

OUT = calibration_output("raw_sheet_ruler")
LINES, HALF_MM, SEED = 150, 1.6, 7
SIDE, TANGENT, REACH, SAME, NEAR = 4.0, 0.7, 6.0, 2.0, 3.0
PROMINENCE, FEWEST_SHEETS = 0.15, 8
RIGHT, LOST, NOISY = (0.8, 1.25), 0.5, 0.25


def solved(sample, volume, unit, planes):
    """Nodes of a slab and the winding coordinate the route's unit solve gives each."""
    scan = LEVEL_ROUTE / sample / volume
    curl = json.loads((scan / "curl.json").read_text())["curl"] if (scan / "curl.json").exists() else None
    nodes, theta, _, _, pairs, record, _ = level_pieces.prepare([stage_folder(scan, "nodes") / unit], planes)
    level_pieces.grow_pieces(nodes, theta, pairs, record, curl)
    winding = record["node_level"] + (curl or record["piece_terms"]["chirality"]) * theta.astype(np.float64) / (
        2 * np.pi
    )
    return nodes, winding, record


def lines_of(nodes, xyz, normal, um):
    """Straight lines along the normals of random nodes, each as start, direction and two directions across."""
    half = HALF_MM / um
    low, high = nodes["z"].min() + SIDE, nodes["z"].max() - SIDE
    for seed in np.random.default_rng(SEED).permutation(len(nodes))[: 4 * LINES]:
        way = normal[seed]
        start = xyz[seed] - half * way
        if not low <= start[2] <= high or not low <= start[2] + 2 * half * way[2] <= high:
            continue
        across = np.cross(way, [0.0, 0.0, 1.0] if abs(way[2]) < 0.9 else [1.0, 0.0, 0.0])
        across /= np.linalg.norm(across)
        yield start, way, (across, np.cross(way, across))


def crossings(tree, xyz, normal, winding, start, way, length):
    """Where sheets cross a line, as distance along it and winding coordinate, nodes of one sheet merged."""
    points = start + np.arange(0, length + SIDE, SIDE)[:, None] * way
    near = np.unique(
        np.concatenate([np.asarray(row, dtype=np.int64) for row in tree.query_ball_point(points, 1.5 * SIDE)])
    )
    along = (xyz[near] - start) @ way
    aside = xyz[near] - start - along[:, None] * way
    cut = normal[near] @ way
    good = (np.linalg.norm(aside, axis=1) <= SIDE) & (np.abs(cut) >= TANGENT)
    where = along[good] + (normal[near][good] * aside[good]).sum(axis=1) / cut[good]
    good[good] = (np.abs(where - along[good]) <= REACH) & (where >= 0) & (where <= length)
    where = along[good] + (normal[near][good] * aside[good]).sum(axis=1) / cut[good]
    order = np.argsort(where)
    merged = []
    for at, value in zip(where[order], winding[near][good][order]):
        if merged and at - merged[-1][0] <= SAME and abs(value - merged[-1][1]) < 0.5:
            continue
        merged.append((at, value))
    return np.array(merged, dtype=np.float64).reshape(-1, 2)


def shifted(start, way, across, length):
    """Voxels along a line and along its four neighbours one voxel aside."""
    steps = np.arange(int(length) + 1)[:, None] * way
    return [
        np.rint(start + shift + steps).astype(int) for shift in (0 * way, across[0], -across[0], across[1], -across[1])
    ]


def bricks_of(store, voxels):
    return {(z, y, x) for x, y, z in set(map(tuple, (np.concatenate(voxels) // np.array(store.brick)[::-1]).tolist()))}


def profile_of(store, voxels, cache):
    """The raw scan along a line, the mean of the line and its four neighbours."""
    total = np.zeros(len(voxels[0]))
    for points in voxels:
        for i, (x, y, z) in enumerate(points):
            key = (z // store.brick[0], y // store.brick[1], x // store.brick[2])
            if key not in cache:
                if len(cache) > 24:
                    cache.clear()
                box = [(k * b, min(k * b + b, s)) for k, b, s in zip(key, store.brick, store.shape)]
                cache[key] = store.read(*box)
            total[i] += cache[key][z % store.brick[0], y % store.brick[1], x % store.brick[2]]
    return ni.gaussian_filter1d(total / len(voxels), 1.0)


def line_score(profile, cross):
    """Raw sheets between a line's first and last sheet with a node, and the levels gained over them with one fold allowed."""
    spread = np.percentile(profile, 90) - np.percentile(profile, 10)
    peaks = find_peaks(profile, prominence=PROMINENCE * spread)[0].astype(np.float64)
    if len(peaks) < FEWEST_SHEETS or len(cross) < FEWEST_SHEETS:
        return None
    nearest = np.abs(cross[:, 0][:, None] - peaks[None, :]).argmin(axis=1)
    on = np.abs(cross[:, 0] - peaks[nearest]) <= NEAR
    sheet, first = np.unique(nearest[on], return_index=True)
    if len(sheet) < 2 or sheet[-1] - sheet[0] < FEWEST_SHEETS:
        return None
    winding = cross[on, 1][first]
    span = float(sheet[-1] - sheet[0])
    folded = np.abs(winding - winding[0]) + np.abs(winding[-1] - winding)
    turn = int(folded.argmax())
    varied = float(np.abs(np.diff(winding)).sum())
    return {
        "raw_sheets": span,
        "levels_per_raw_sheet": float(folded[turn]) / span,
        "noise_per_raw_sheet": (varied - float(folded[turn])) / span,
        "folded": bool(min(abs(winding[turn] - winding[0]), abs(winding[-1] - winding[turn])) >= 3),
        "raw_sheets_with_a_node": len(sheet) / (span + 1),
        "nodes_per_raw_sheet": len(cross) / max(len(peaks), 1),
    }


def summary(rows, um):
    """What the lines of one slab say together."""
    gain, noise = (np.array([row[key] for row in rows]) for key in ("levels_per_raw_sheet", "noise_per_raw_sheet"))
    right = (gain >= RIGHT[0]) & (gain <= RIGHT[1]) & (noise <= NOISY)
    return {
        "lines": len(rows),
        "levels_per_raw_sheet": [round(float(v), 3) for v in np.percentile(gain, [10, 25, 50, 75, 90])],
        "right": round(float(right.mean()), 3),
        "lost": round(float((gain < LOST).mean()), 3),
        "noisy": round(float((noise > NOISY).mean()), 3),
        "folded": round(float(np.mean([row["folded"] for row in rows])), 3),
        "raw_sheets_per_mm": round(float(np.median([row["raw_sheets"] for row in rows]) / (2 * HALF_MM)), 2),
        "raw_sheets_with_a_node": round(float(np.median([row["raw_sheets_with_a_node"] for row in rows])), 3),
        "nodes_per_raw_sheet": round(float(np.median([row["nodes_per_raw_sheet"] for row in rows])), 2),
        "node_um": round(um * 1000, 3),
    }


def scored(nodes, winding, store, um, fetch=None):
    """The score of every usable line, the raw bricks they cross fetched first."""
    xyz = np.stack([nodes[c].astype(np.float64) for c in "xyz"], axis=1)
    normal = np.stack([nodes[c].astype(np.float64) for c in ("nx", "ny", "nz")], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    length = 2 * HALF_MM / um
    tree = cKDTree(xyz)
    lines = [
        (start, way, shifted(start, way, across, length)) for start, way, across in lines_of(nodes, xyz, normal, um)
    ]
    inside = [all((v >= 0).all() and (v < np.array(store.shape)[::-1]).all() for v in line[2]) for line in lines]
    lines = [line for line, kept in zip(lines, inside) if kept][:LINES]
    if fetch:
        fetch(set().union(*[bricks_of(store, line[2]) for line in lines]))
    cache, rows = {}, []
    for start, way, voxels in lines:
        found = line_score(profile_of(store, voxels, cache), crossings(tree, xyz, normal, winding, start, way, length))
        if found:
            rows.append(found)
    return rows


def run(sample, volume, unit, plane0, plane1, out):
    nodes, winding, record = solved(sample, volume, unit, (int(plane0), int(plane1)))
    um = record["scan_voxel_um"] * 2 ** record["level"] / 1000
    store, url = sampler.scan_store(sample, volume, OUT / sample / "bricks", record["level"])

    def fetch(wanted):
        todo = [i for i in sorted(wanted) if store.offsets[i] == -1]
        failures = sampler.fetched_all(url, [(i, "/".join(map(str, i))) for i in todo], store.put)
        store.save()
        if failures:
            raise RuntimeError(f"{len(failures)} bricks not fetched. Rerun to resume.")

    rows = scored(nodes, winding, store, um, fetch)
    kept = {"sample": sample, "unit": unit, "planes": [int(plane0), int(plane1)], "nodes": len(nodes)}
    kept |= {"refused_share": round(record["level_route"]["refused_share"], 4)} | summary(rows, um) | {"rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(kept, indent=1))
    print(sample, unit, plane0, plane1, {k: v for k, v in kept.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:7])
