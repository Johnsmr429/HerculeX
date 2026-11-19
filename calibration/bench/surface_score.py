"""Extracted sheet surfaces scored against the truth: how far each vertex lies from the meshes, which wrap it follows,
how much of a surface leaves its dominant wrap (sheet switching) and how much correct area is connected.

A vertex is on the truth when a mesh vertex lies within 40 µm along that mesh's normal (v1's tolerance) and within
0.75 of the truth's vertex spacing across it (v1 allowed 160 µm; --lateral-um 160 restores that). Along a
surface the truth's continuous winding minus the surface's own unwrapped turn about the axis is constant while the
surface stays on one sheet; per 4-connected part of a surface its commonest whole-turn offset (per truth frame) is the
dominant wrap and every other offset is a switch. Each truth mesh has its own phase, as in winding_score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from calibration.bench import results
from calibration.bench import truth as truths
from calibration.bench.nearest import on_sheet
from calibration.bench.winding_score import modal, phases

TOLERANCE_UM = 40.0
LATERAL_SHARE = 0.75
FAR_UM = 300.0
NULL_SHIFT_UM = 80.0
BLOB_MM2 = 16.0
FLOORS_CM2 = (1.0, 4.0)
RECALL_SAMPLE = 500_000


def load_surface(path, mask="known"):
    """A surface as (rows, columns, 3) level-0 x, y, z with its mask: a tifxyz folder (x.tif, y.tif, z.tif and
    optional quality.npy: 1 locked, 2 interpolated) or an .npz with xyz (or x, y, z) and optional mask."""
    path = Path(path)
    if path.is_dir():
        xyz = np.stack([tifffile.imread(path / f"{a}.tif") for a in "xyz"], axis=-1).astype(np.float32)
        valid = np.all(xyz > 0, axis=-1) & np.all(np.isfinite(xyz), axis=-1)
        quality = path / "quality.npy"
        if quality.exists():
            q = np.load(quality)
            valid &= (q == 1) if mask == "locked" else (q > 0)
        return {"name": path.name, "xyz": xyz, "mask": valid}
    data = np.load(path)
    xyz = data["xyz"] if "xyz" in data.files else np.stack([data[a] for a in "xyz"], axis=-1)
    xyz = xyz.astype(np.float32)
    valid = np.all(np.isfinite(xyz), axis=-1) & np.all(xyz > 0, axis=-1)
    if "mask" in data.files:
        valid &= data["mask"].astype(bool)
    return {"name": path.stem, "xyz": xyz, "mask": valid}


def surfaces_in(path):
    """Surface paths under a path: itself, its piece-* folders, or its .npz files."""
    path = Path(path)
    if path.suffix == ".npz" or (path / "x.tif").exists():
        return [path]
    found = sorted(p for p in path.glob("piece-*") if (p / "x.tif").exists())
    return found or sorted(path.glob("*.npz"))


def vertex_area(xyz, mask):
    """Area per vertex (level-0 voxels squared): every complete grid quad split over its four corners."""
    full = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    diagonal_a = xyz[1:, 1:] - xyz[:-1, :-1]
    diagonal_b = xyz[1:, :-1] - xyz[:-1, 1:]
    quad = np.where(full, 0.5 * np.linalg.norm(np.cross(diagonal_a, diagonal_b), axis=-1), 0.0)
    area = np.zeros(mask.shape)
    for dr in (0, 1):
        for dc in (0, 1):
            area[dr : dr + quad.shape[0], dc : dc + quad.shape[1]] += quad / 4
    return area


def unwrapped(theta, mask):
    """Angle unwrapped over each 4-connected part of a grid (spanning tree, pointer jumping); and the part labels."""
    index = np.full(mask.shape, -1, dtype=np.int64)
    n = int(mask.sum())
    index[mask] = np.arange(n)
    pairs = []
    for a, b in ((index[:, :-1], index[:, 1:]), (index[:-1, :], index[1:, :])):
        both = (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[both], b[both]], axis=1))
    edges = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    count, part = connected_components(graph, directed=False)
    roots = np.unique(part, return_index=True)[1]
    rows = np.concatenate([edges[:, 0], np.full(count, n)])
    columns = np.concatenate([edges[:, 1], roots])
    tree = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(n + 1, n + 1)).tocsr()
    _, parent = breadth_first_order(tree, n, directed=False, return_predecessors=True)
    t = theta[mask]
    parent = parent[:n]
    parent[roots] = roots
    step = np.angle(np.exp(1j * (t - t[parent])))
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            break
        step = step + step[parent]
        parent = grand
    return t[parent] + step, part


def followed(mesh, area, correct, truth, top=4):
    """The truth meshes a surface lies on, by area, with the share of each on the dominant wrap."""
    out = []
    for m in np.unique(mesh[mesh >= 0]):
        mine = mesh == m
        out.append(
            {
                "mesh": str(truth["names"][m]),
                "wrap": int(truth["mesh_wrap"][m]),
                "cm2": round(float(area[mine].sum()), 4),
                "correct_share": round(float(area[mine & correct].sum() / max(area[mine].sum(), 1e-12)), 4),
            }
        )
    return sorted(out, key=lambda r: -r["cm2"])[:top]


def score_grid(surface, labels, truth, voxel_um):
    """Per-surface metrics from per-vertex labels (on truth, offset, depth) on its grid."""
    mask = surface["mask"]
    area = vertex_area(surface["xyz"], mask) * (voxel_um * 1e-4) ** 2
    a = area[mask]
    matched, winding, frame = labels["matched"], labels["winding"], labels["frame"]
    theta, _ = truths.angle(truth["knots"], surface["xyz"][mask].astype(np.float64))
    grid_theta = np.zeros(mask.shape)
    grid_theta[mask] = theta
    turn, part = unwrapped(grid_theta, mask)
    judged = matched & np.isfinite(winding)
    correct = np.zeros(len(a), dtype=bool)
    if judged.any():
        o = winding[judged] - truth["sense"] * turn[judged] / (2 * np.pi)
        key = part[judged].astype(np.int64) * (int(frame[judged].max()) + 1) + frame[judged]
        phase, _ = phases(o, key, labels["mesh"][judged])
        offset = np.round(o - phase).astype(np.int64)
        _, right = modal(offset, key)
        correct[np.flatnonzero(judged)[right]] = True
    wrong = judged & ~correct
    grid_correct = np.zeros(mask.shape, dtype=bool)
    grid_correct[mask] = correct
    parts, count = ndimage.label(grid_correct, structure=np.ones((3, 3)))
    sizes = ndimage.sum(area, parts, index=np.arange(1, count + 1)) if count else np.zeros(0)
    grid_wrong = np.zeros(mask.shape, dtype=bool)
    grid_wrong[mask] = wrong
    blobs, blob_count = ndimage.label(grid_wrong, structure=np.ones((3, 3)))
    blob_sizes = ndimage.sum(area, blobs, index=np.arange(1, blob_count + 1)) if blob_count else np.zeros(0)
    total = float(a.sum())
    right, bad = float(a[correct].sum()), float(a[wrong].sum())
    resolved = ~labels["pressed"]
    right_resolved, bad_resolved = float(a[correct & resolved].sum()), float(a[wrong & resolved].sum())
    return {
        "surface": surface["name"],
        "vertices": int(mask.sum()),
        "parts": int(part.max() + 1) if len(part) else 0,
        "area_cm2": round(total, 4),
        "on_truth_cm2": round(float(a[matched].sum()), 4),
        "off_truth_cm2": round(float(a[labels["near"] & ~matched].sum()), 4),
        "no_truth_cm2": round(float(a[~labels["near"]].sum()), 4),
        "pressed_cm2": round(float(a[labels["pressed"]].sum()), 4),
        "correct_cm2": round(right, 4),
        "wrong_cm2": round(bad, 4),
        "correct_resolved_cm2": round(right_resolved, 4),
        "wrong_resolved_cm2": round(bad_resolved, 4),
        "wrong_rate": round(bad / (right + bad), 4) if right + bad > 0 else None,
        "largest_correct_cm2": round(float(sizes.max()), 4) if len(sizes) else 0.0,
        **{f"correct_in_{f:g}cm2_parts": round(float(sizes[sizes >= f].sum()), 4) for f in FLOORS_CM2},
        "wrong_blobs": int((blob_sizes >= BLOB_MM2 / 100).sum()),
        "wrong_in_blobs_cm2": round(float(blob_sizes[blob_sizes >= BLOB_MM2 / 100].sum()), 4),
        "follows": followed(labels["mesh"], a, correct, truth),
        "_correct": correct,
        "_area": a,
    }


def quantiles(values):
    if not len(values):
        return None
    return [round(float(v), 2) for v in np.percentile(values, (50, 90, 99))]


def score(truth, paths, mask="known", recall=True, keep_rows=True, lateral_um=None, labels_path=None):
    """Metrics of a set of surfaces (paths of tifxyz folders or .npz files) on one truth; a vertex matches a truth
    vertex within 40 µm along its normal and lateral_um across (default 0.75 of the truth's vertex spacing)."""
    voxel_um = truth["scan_voxel_um"]
    surfaces = [load_surface(p, mask) for p in paths]
    surfaces = [s for s in surfaces if s["mask"].sum() >= 4]
    if not surfaces:
        return {"error": "no surface vertices"}
    vertices = np.concatenate([s["xyz"][s["mask"]] for s in surfaces])
    lateral = (lateral_um or LATERAL_SHARE * truth["spacing_um"]) / voxel_um
    reach = np.hypot(FAR_UM / voxel_um, lateral)
    near_box = truths.in_box(truth["points"], (vertices.min(axis=0) - reach, vertices.max(axis=0) + reach))
    if near_box.mean() < 0.8:
        truth = truth | {k: truth[k][near_box] for k in ("points", "normals", "mesh", "winding", "frame")}
    index, depth = on_sheet(
        truth["points"], vertices, truth["normals"], FAR_UM / voxel_um, lateral, keep=2, normal_of="source"
    )
    nearest = np.where(index[:, 0] >= 0, index[:, 0], 0)
    matched = depth[:, 0] <= TOLERANCE_UM / voxel_um
    near = index[:, 0] >= 0
    winding = np.where(matched, truth["winding"][nearest], np.nan)
    frame = np.where(matched, truth["frame"][nearest], -1).astype(np.int64)
    mesh = np.where(matched, truth["mesh"][nearest], -1).astype(np.int64)
    second = np.where(index[:, 1] >= 0, index[:, 1], 0)
    pressed = (
        matched
        & (index[:, 1] >= 0)
        & (depth[:, 1] <= 2 * TOLERANCE_UM / voxel_um)
        & (np.abs(truth["winding"][second] - truth["winding"][nearest]) >= 0.5)
    )
    rows, start = [], 0
    for s in surfaces:
        n = int(s["mask"].sum())
        part = slice(start, start + n)
        labels = {
            "matched": matched[part],
            "near": near[part],
            "pressed": pressed[part],
            "winding": winding[part],
            "frame": frame[part],
            "mesh": mesh[part],
        }
        rows.append(score_grid(s, labels, truth, voxel_um))
        start += n
    correct = np.concatenate([r.pop("_correct") for r in rows])
    area = np.concatenate([r.pop("_area") for r in rows])
    depth_um = depth[:, 0] * voxel_um
    if labels_path:
        truths.write_npz(
            labels_path,
            xyz=vertices,
            surface=np.repeat(np.arange(len(surfaces)), [int(s["mask"].sum()) for s in surfaces]),
            names=np.array([s["name"] for s in surfaces]),
            depth_um=depth_um.astype(np.float32),
            matched=matched,
            mesh=mesh,
            winding=winding.astype(np.float32),
            correct=correct,
        )
    total = float(area.sum())
    right = sum(r["correct_cm2"] for r in rows)
    bad = sum(r["wrong_cm2"] for r in rows)
    right_resolved = sum(r["correct_resolved_cm2"] for r in rows)
    bad_resolved = sum(r["wrong_resolved_cm2"] for r in rows)
    big = [r for r in rows if r["area_cm2"] >= FLOORS_CM2[0]]
    right_big, bad_big = sum(r["correct_cm2"] for r in big), sum(r["wrong_cm2"] for r in big)
    out = {
        "surfaces": len(rows),
        "vertices": len(vertices),
        "area_cm2": round(total, 3),
        "on_truth_share": round(float(area[matched].sum() / total), 4),
        "off_truth_share": round(float(area[near & ~matched].sum() / total), 4),
        "no_truth_share": round(float(area[~near].sum() / total), 4),
        "pressed_share_of_on_truth": round(float(area[pressed].sum() / max(area[matched].sum(), 1e-12)), 4),
        "correct_cm2": round(right, 3),
        "wrong_cm2": round(bad, 3),
        "wrong_rate": round(bad / (right + bad), 4) if right + bad > 0 else None,
        "wrong_rate_resolved": round(bad_resolved / (right_resolved + bad_resolved), 4)
        if right_resolved + bad_resolved > 0
        else None,
        "dominant_share": round(right / (right + bad), 4) if right + bad > 0 else None,
        "largest_correct_cm2": round(max(r["largest_correct_cm2"] for r in rows), 3) if right + bad > 0 else None,
        "correct_in_1cm2_parts": round(sum(r["correct_in_1cm2_parts"] for r in rows), 3) if right + bad > 0 else None,
        "correct_in_4cm2_parts": round(sum(r["correct_in_4cm2_parts"] for r in rows), 3) if right + bad > 0 else None,
        "surfaces_1cm2": len(big),
        "correct_cm2_in_1cm2_surfaces": round(right_big, 3),
        "wrong_rate_in_1cm2_surfaces": round(bad_big / (right_big + bad_big), 4) if right_big + bad_big else None,
        "wrong_blobs": sum(r["wrong_blobs"] for r in rows),
        "depth_um_on_truth_p50_p90_p99": quantiles(depth_um[matched]),
        "depth_um_near_p50_p90_p99": quantiles(depth_um[near]),
        "lateral_reach_um": round(lateral * voxel_um, 1),
    }
    if recall:
        out |= truth_recall(truth, vertices, correct, surfaces, voxel_um)
    if keep_rows:
        out["per_surface"] = rows
    return out


def truth_recall(truth, vertices, correct, surfaces, voxel_um):
    """Share of the truth (within the surfaces' bounding box) that some surface vertex covers, covers correctly, and covers
    when the truth is moved 80 µm along its normal (false coverage)."""
    margin = TOLERANCE_UM / voxel_um
    low, high = vertices.min(axis=0) - margin, vertices.max(axis=0) + margin
    inside = np.flatnonzero(truths.in_box(truth["points"], (low, high)))
    inside = inside[:: max(1, len(inside) // RECALL_SAMPLE)]
    steps = [np.linalg.norm(np.diff(s["xyz"], axis=1)[s["mask"][:, 1:] & s["mask"][:, :-1]], axis=-1) for s in surfaces]
    step = float(np.median(np.concatenate(steps))) if any(len(s) for s in steps) else 1.0
    lateral = LATERAL_SHARE * step
    out = {"truth_recall_sample": len(inside)}
    for name, shift in (("recall", 0.0), ("recall_null", NULL_SHIFT_UM / voxel_um)):
        points = truth["points"][inside] + shift * truth["normals"][inside]
        index, _ = on_sheet(vertices, points, truth["normals"][inside], TOLERANCE_UM / voxel_um, lateral, k=8)
        hit = index[:, 0] >= 0
        out[name] = round(float(hit.mean()), 4) if len(hit) else None
        if name == "recall":
            out["recall_correct"] = round(float(correct[index[hit, 0]].sum() / max(len(hit), 1)), 4)
    return out


def headline(out):
    keys = (
        "area_cm2",
        "on_truth_share",
        "correct_cm2",
        "wrong_cm2",
        "wrong_rate",
        "wrong_rate_resolved",
        "largest_correct_cm2",
        "correct_in_1cm2_parts",
        "correct_in_4cm2_parts",
        "correct_cm2_in_1cm2_surfaces",
        "wrong_rate_in_1cm2_surfaces",
        "recall",
        "recall_correct",
        "recall_null",
        "surfaces",
    )
    flat = {k: out[k] for k in keys if out.get(k) is not None}
    if out.get("depth_um_on_truth_p50_p90_p99"):
        flat["depth_um_p50"], flat["depth_um_p90"], _ = out["depth_um_on_truth_p50_p90_p99"]
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control")
    parser.add_argument("surfaces", nargs="+", help="tifxyz folders, folders of piece-* folders, or .npz files")
    parser.add_argument("--mask", default="known", choices=("known", "locked"), help="tifxyz cells to score")
    parser.add_argument("--truth", default="auto", choices=("auto", "names", "groups"))
    parser.add_argument("--every", type=int, default=1, help="use every n-th truth vertex")
    parser.add_argument("--no-recall", action="store_true")
    parser.add_argument("--lateral-um", type=float, help="lateral reach to a truth vertex (v1 used 160)")
    parser.add_argument("--labels", help="write per-vertex labels (mesh followed, depth, correct) to this .npz")
    parser.add_argument("--rows", action="store_true", help="print one line per surface")
    parser.add_argument("--method", default="surfaces")
    parser.add_argument("--system", default="network")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    paths = [p for given in args.surfaces for p in surfaces_in(given)]
    truth = truths.load(args.control, args.truth, args.every)
    out = score(truth, paths, args.mask, not args.no_recall, args.rows, args.lateral_um, args.labels)
    out |= {"control": args.control, "truth": truth["source"], "split": truths.split(args.control)}
    print(json.dumps(out))
    if not args.no_record and "error" not in out:
        parameters = {
            "surfaces": [str(p) for p in args.surfaces],
            "mask": args.mask,
            "truth": truth["source"],
            "every": args.every,
        }
        run = results.record(
            args.control,
            args.system,
            args.method,
            "surface",
            headline(out),
            parameters,
            detail={k: v for k, v in out.items() if k != "per_surface"},
        )
        print(f"recorded run {run}")


if __name__ == "__main__":
    main()
