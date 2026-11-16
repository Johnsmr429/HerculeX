"""Every sheet control unwrapped and scored for recovery, purity and frame agreement."""

import json
import shutil
import sys
import time

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from runner.stage_queue import done, record_of
from storage.data_roots import SHEET_CONTROLS, SURFACE_RECORDS, calibration_output
from storage.node_store import read_nodes
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level.piece_joiner import SEAM_PLANES, join, joins_of, seam_field
from unwrapping.level.run_levels import extract, piece_name
from unwrapping.level.sheet_graph import MIN_PROMINENCE

TOLERANCE_UM = 40.0
USABLE_MM2 = 100.0
REACH_CELLS = 1.5
SCRATCH = calibration_output("sheet_study")


def upsampled(xyz, locked):
    """Grid points and the midpoints between locked neighbours."""
    parts = [
        (xyz, locked),
        ((xyz[:, :, :-1] + xyz[:, :, 1:]) / 2, locked[:, :-1] & locked[:, 1:]),
        ((xyz[:, :-1] + xyz[:, 1:]) / 2, locked[:-1] & locked[1:]),
        (
            (xyz[:, :-1, :-1] + xyz[:, 1:, 1:] + xyz[:, :-1, 1:] + xyz[:, 1:, :-1]) / 4,
            locked[:-1, :-1] & locked[1:, 1:] & locked[:-1, 1:] & locked[1:, :-1],
        ),
    ]
    return np.concatenate([p[:, m].T for p, m in parts])


def piece_points(pieces, summary, level):
    """Locked points of every written piece at the sheet level, each with its piece."""
    points, owner = [], []
    scale = 2**level
    for row in summary["pieces"]:
        if not row["written"]:
            continue
        folder = pieces / piece_name(row["piece"])
        xyz = np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"]).astype(np.float64)
        p = (upsampled(xyz, np.load(folder / "quality.npy") == 1) - 0.5 * (scale - 1)) / scale
        points.append(p)
        owner.append(np.full(len(p), row["piece"], dtype=np.int32))
    return np.concatenate(points), np.concatenate(owner)


def nearest_on_sheet(query, normals, tree, points, reach, tolerance):
    """Index of the piece point nearest each query along its normal, within reach and tolerance."""
    near, index = tree.query(query, k=8, distance_upper_bound=reach)
    found = np.isfinite(near)
    safe = np.where(found, index, 0)
    depth = np.abs(((points[safe] - query[:, None, :]) * normals[:, None, :]).sum(axis=2))
    depth = np.where(found, depth, np.inf)
    best = np.argmin(depth, axis=1)
    rows = np.arange(len(query))
    return np.where(depth[rows, best] <= tolerance, safe[rows, best], -1)


def conflicts(truth):
    """Mesh pairs known to be different sheets."""
    M = len(truth["names"])
    named = np.array([truth["wrap"][truth["mesh"] == k][0] if (truth["mesh"] == k).any() else -1 for k in range(M)])
    return (named[:, None] >= 0) & (named[None, :] >= 0) & (named[:, None] != named[None, :])


def mesh_score(control, pieces, summary):
    """Recovery, coverage, its null half a gap away, purity, node detection and frame agreement against the meshes."""
    record = record_of(control)
    truth = np.load(control / "truth.npz")
    voxel = record["voxel_um"]
    points, owner = piece_points(pieces, summary, record["level"])
    areas = {r["piece"]: r["area_mm2"] for r in summary["pieces"]}
    windings = {r["piece"]: r["winding"] for r in summary["pieces"]}
    frames = {r["piece"]: r["frame"] for r in summary["pieces"]}
    stride = (
        next(
            json.loads((pieces / piece_name(r["piece"]) / "meta.json").read_text())["stride"]
            for r in summary["pieces"]
            if r["written"]
        )
        / 2 ** record["level"]
    )
    tree = cKDTree(points)
    p, n, mesh = truth["points"].astype(np.float64), truth["normals"].astype(np.float64), truth["mesh"]
    tolerance = TOLERANCE_UM / voxel
    hit = nearest_on_sheet(p, n, tree, points, REACH_CELLS * stride, tolerance)
    shifted = nearest_on_sheet(
        p + n * (summary.get("gap_um", 160.0) / 2 / voxel), n, tree, points, REACH_CELLS * stride, tolerance
    )
    piece = np.where(hit >= 0, owner[np.maximum(hit, 0)], -1)
    usable = np.array([areas.get(int(k), 0.0) >= USABLE_MM2 for k in piece]) & (piece >= 0)
    modal = {}
    for k in np.unique(piece[usable]):
        values, counts = np.unique(mesh[usable & (piece == k)], return_counts=True)
        modal[int(k)] = int(values[np.argmax(counts)])
    home = np.array([modal.get(int(k), -2) for k in piece])
    duplicate = (
        np.isin(np.arange(len(p), dtype=np.int64) * len(truth["names"]) + np.maximum(home, 0), truth["coincide"])
        if "coincide" in truth
        else False
    )
    right = usable & ((home == mesh) | duplicate)
    different = conflicts(truth)[np.maximum(home, 0), mesh] & (home >= 0)
    wrong = usable & ~right & different
    unknown = usable & ~right & ~different
    nodes = read_nodes(control)
    nodes = nodes[nodes["prominence"] / 2 >= MIN_PROMINENCE]
    node_xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64)
    detected = nearest_on_sheet(p, n, cKDTree(node_xyz), node_xyz, 1.5 * 8, tolerance) >= 0
    out = {
        "vertices": len(p),
        "meshes": len(truth["names"]),
        "detected": float(detected.mean()),
        "covered": float((hit >= 0).mean()),
        "covered_null": float((shifted >= 0).mean()),
        "in_usable_pieces": float(usable.mean()),
        "recovery": float((right | unknown).mean()),
        "recovery_strict": float(right.mean()),
        "purity": float(right.sum() / max((right | wrong).sum(), 1)),
        "unjudged": float(unknown.mean()),
    }
    named = usable & (truth["wrap"] >= 0)
    if named.sum() > 100:
        main = np.bincount([frames[int(k)] + 1 for k in piece[named]]).argmax() - 1
        framed = named & np.array([frames.get(int(k), -9) == main for k in piece])
        offset = np.array([windings[int(k)] for k in piece[framed]]) - summary["chirality"] * truth["wrap"][framed]
        values, counts = np.unique(offset, return_counts=True)
        out["frame_agreement"] = float(counts.max() / max(named.sum(), 1))
    return out


def seam_pairs(truth, seam, reach):
    """Truth vertex pairs of one mesh straddling the seam."""
    p, mesh = truth["points"].astype(np.float64), truth["mesh"]
    plane = p[:, 2] / CELL
    below = np.flatnonzero((plane >= seam - SEAM_PLANES) & (plane < seam))
    above = np.flatnonzero((plane >= seam) & (plane < seam + SEAM_PLANES))
    pairs = []
    for k in np.unique(mesh[below]):
        up = above[mesh[above] == k]
        if not len(up):
            continue
        down = below[mesh[below] == k]
        d, i = cKDTree(p[up]).query(p[down], distance_upper_bound=reach)
        ok = np.isfinite(d)
        pairs.append(np.stack([down[ok], up[i[ok]]], axis=1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)


def wrap_near(xyz, truth, reach, tolerance):
    """Named wrap of the truth vertex nearest each point along that vertex's normal, within reach and tolerance."""
    named = truth["wrap"] >= 0
    if not named.any():
        return np.full(len(xyz), -1)
    p, n, wrap = (
        truth["points"][named].astype(np.float64),
        truth["normals"][named].astype(np.float64),
        truth["wrap"][named],
    )
    near, index = cKDTree(p).query(xyz, k=8, distance_upper_bound=reach)
    found = np.isfinite(near)
    safe = np.where(found, index, 0)
    depth = np.where(found, np.abs(((p[safe] - xyz[:, None, :]) * n[safe]).sum(axis=2)), np.inf)
    best = np.argmin(depth, axis=1)
    rows = np.arange(len(xyz))
    return np.where(depth[rows, best] <= tolerance, wrap[safe[rows, best]], -1)


def joint_score(control):
    """The control cut at its middle into two chunks unwrapped alone and joined across the seam."""
    record = record_of(control)
    truth = np.load(control / "truth.npz")
    lo, hi = record["cell_planes"]
    seam = (lo + hi) // 2
    windows, labels = [], []
    for name, planes, shared in (
        ("lower", (lo, seam), (seam - SEAM_PLANES, seam)),
        ("upper", (seam, hi), (seam, seam + SEAM_PLANES)),
    ):
        folder = SCRATCH / f"{control.name}-{name}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        summary, _ = extract(
            [control],
            folder,
            record["sample"],
            record["volume"],
            record["scan_voxel_um"],
            record["level"],
            planes=planes,
            shared=[shared],
        )
        boundary = np.load(folder / "boundary.npz")
        order = np.argsort(boundary["ids"])
        labels.append((boundary["ids"][order], boundary["piece"][order]))
        windows.append((folder, summary, int(boundary["pieces"])))
    piece, side, edges, pairs, certainty, _, nodes = seam_field([control], seam, labels)
    joins = joins_of(piece, side, edges, pairs, certainty)
    global_id = join([w[2] for w in windows], [joins])
    p, n = truth["points"].astype(np.float64), truth["normals"].astype(np.float64)
    tolerance = TOLERANCE_UM / record["voxel_um"]
    covering = []
    for folder, summary, _ in windows:
        points, owner = piece_points(folder, summary, record["level"])
        hit = nearest_on_sheet(p, n, cKDTree(points), points, REACH_CELLS * CELL, tolerance)
        covering.append(np.where(hit >= 0, owner[np.maximum(hit, 0)], -1))
    pairs_of_truth = seam_pairs(truth, seam, 2.5 * CELL)
    below, above = covering[0][pairs_of_truth[:, 0]], covering[1][pairs_of_truth[:, 1]]
    both = (below >= 0) & (above >= 0)
    joined = global_id[below[both]] == global_id[windows[0][2] + above[both]]
    wrap = wrap_near(
        np.stack([nodes[c] for c in "xyz"], axis=1).astype(np.float64), truth, REACH_CELLS * CELL, tolerance
    )
    a, b = edges["a"], edges["b"]
    lower_node, upper_node = np.where(side[a] == 0, a, b), np.where(side[a] == 0, b, a)
    on = (side[a] != side[b]) & (piece[a] >= 0) & (piece[b] >= 0) & (wrap[lower_node] >= 0) & (wrap[upper_node] >= 0)
    span = int(piece.max()) + 1
    keys, inverse = np.unique(piece[lower_node[on]] * span + piece[upper_node[on]], return_inverse=True)
    differ = np.bincount(
        inverse, weights=(wrap[lower_node[on]] != wrap[upper_node[on]]).astype(np.float64), minlength=len(keys)
    )
    differ_share = dict(zip(keys.tolist(), (differ / np.bincount(inverse, minlength=len(keys))).tolist()))
    judged = [count for x, y, count in joins if x * span + y in differ_share]
    false = [count for x, y, count in joins if differ_share.get(x * span + y, 0.0) > 0.5]
    for folder, _, _ in windows:
        shutil.rmtree(folder, ignore_errors=True)
    return {
        "control": control.name,
        "seam_pairs": len(pairs_of_truth),
        "covered_pairs": int(both.sum()),
        "joins": len(joins),
        "joint_accuracy": round(float(joined.mean()), 4) if both.any() else None,
        "judged_joins": len(judged),
        "false_joins": len(false),
        "false_join_share": round(sum(false) / sum(judged), 4) if judged else None,
    }


def score(control):
    record = record_of(control)
    pieces = SCRATCH / control.name
    shutil.rmtree(pieces, ignore_errors=True)
    pieces.mkdir(parents=True)
    start = time.time()
    summary, _ = extract(
        [control], pieces, record["sample"], record["volume"], record["scan_voxel_um"], record["level"]
    )
    seconds = time.time() - start
    result = mesh_score(control, pieces, summary)
    shutil.rmtree(pieces, ignore_errors=True)
    return {
        "control": control.name,
        "sample": record["sample"],
        "voxel_um": record["voxel_um"],
        "seconds": round(seconds, 1),
        "area_cm2": round(summary["area_cm2"], 1),
        "area_in_1cm2_pieces": round(summary["area_in_1cm2_pieces"], 1),
    } | {k: round(v, 4) for k, v in result.items()}


def study(names=()):
    rows = []
    for control in sorted(p for p in SHEET_CONTROLS.iterdir() if p.is_dir() and done(p) and (p / "truth.npz").exists()):
        if names and control.name not in names and record_of(control)["sample"] not in names:
            continue
        rows.append(score(control))
        print(json.dumps(rows[-1]), flush=True)
    real = [r for r in rows if "covered" in r]
    result = {
        "controls": rows,
        "mean_recovery": round(float(np.mean([r["recovery"] for r in rows])), 4) if rows else None,
        "real_mean_recovery": round(float(np.mean([r["recovery"] for r in real])), 4) if real else None,
    }
    if not names:
        SURFACE_RECORDS.mkdir(exist_ok=True)
        (SURFACE_RECORDS / "sheet-study.json").write_text(json.dumps(result, indent=1))
    print({k: v for k, v in result.items() if k != "controls"}, flush=True)
    return result


def joint_study(names=()):
    rows = []
    for control in sorted(p for p in SHEET_CONTROLS.iterdir() if p.is_dir() and done(p) and (p / "truth.npz").exists()):
        if names and control.name not in names and record_of(control)["sample"] not in names:
            continue
        rows.append(joint_score(control))
        print(json.dumps(rows[-1]), flush=True)
    scored = [r for r in rows if r["joint_accuracy"] is not None]
    result = {
        "controls": rows,
        "joint_accuracy": round(float(np.mean([r["joint_accuracy"] for r in scored])), 4) if scored else None,
        "false_join_share": round(
            float(np.mean([r["false_join_share"] for r in scored if r["false_join_share"] is not None])), 4
        )
        if any(r["false_join_share"] is not None for r in scored)
        else None,
    }
    if not names:
        (SURFACE_RECORDS / "joint-study.json").write_text(json.dumps(result, indent=1))
    print({k: v for k, v in result.items() if k != "controls"}, flush=True)
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["joint"]:
        joint_study(sys.argv[2:])
    else:
        study(sys.argv[1:])
