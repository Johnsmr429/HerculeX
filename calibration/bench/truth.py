"""Published-mesh truth of the v1 sheet controls in level-0 voxels, with a continuous winding per vertex.

Two sources. 'names': v1's truth.npz (every mesh grid vertex nearest 100 µm apart, which is the 20-voxel mesh grid
itself on 7.9-9.4 µm scans); a wrap-named mesh's winding is its name's wrap plus the turns travelled along the mesh
grid from where the mesh starts. Meshes start at different angles, so each mesh is then moved by less than a turn
until winding - sense*theta/2pi has one fraction on every mesh, the cut between fractions placed in their widest
gap: all named meshes then form one winding that rises by exactly 1 per wrap. A multi-wrap mesh is shifted by whole
turns to agree with the meshes it lies on; a vertex repeating a larger mesh's sheet at the same winding is dropped.
Where a scroll's names miscount their wraps, NAME_TURNS moves the meshes concerned by whole turns, and the audit
(names.json) records it.
'groups': v1's wrap truth (vertices nearest 150 µm apart, Wg continuous within coincidence groups whose mutual
offsets are unknown; touching wraps are often merged into one group, so it under-counts sheet switches).
"""

import argparse
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from storage.data_roots import CALIBRATION_OUTPUTS, OPEN_DATA_MIRROR
from storage.data_roots import SHEET_CONTROLS as V1_CONTROLS
from storage.data_roots import SURFACE_BENCH as BENCH

WRAP_TRUTH = CALIBRATION_OUTPUTS / "geometry_wrap" / "truth"
OPEN_DATA = OPEN_DATA_MIRROR
MANIFEST = Path(__file__).with_name("controls.json")
CACHE = BENCH / "truth"
VERSION = 2
TRUTH_UM = 100.0
GROUPS_UM = 150.0
SAME_SHEET_UM = 40.0
RADIAL_UM = 400.0
MERGED_TURNS = 1.6
START_QUANTILE = 0.001
ANCHOR_LEAST = 250
ANCHOR_SHARE = 0.01
CHUNK = 500_000
# Whole turns added to named meshes whose names miscount their wrap, per v1 control: {control: {(first, last): turns}}.
# PHerc0139's names count one wrap too many from w051: w050 spans 0.45 turns, and w050 and w051 lie 2 µm apart along the
# normal at z 4952-5063. Against the bench packages, which chain wraps by radial votes, w051-w059 are one turn high on
# every vertex. Agreed by surface, ml and data, 17 Nov 2026.
NAME_TURNS = {
    "PHerc0139-20250728140407-rows-0007-0019": {(51, 59): -1},
    "PHerc0139-20250728140407-rows-0019-0031": {(51, 59): -1},
}


def controls():
    return sorted(p.parent.name for p in V1_CONTROLS.glob("*/record.json"))


def record(control):
    return json.loads((V1_CONTROLS / control / "record.json").read_text())


def split(control):
    """Split of a control: a package name of calibration/bench/controls.json, or a v1 band SAMPLE-SCAN-rows-... judged by its scan:
    'mixed' where the scan holds the dev zone (its bands run through dev, buffer and holdout-B), 'holdout-A' for the
    other scans of the manifest, 'train' for training scans (train and val zones only), 'excluded' for scans outside
    it."""
    rules = json.loads(MANIFEST.read_text())
    if control in rules["controls"]:
        control = rules["controls"][control]["scan"]
    scan = "-".join(control.split("-")[:2])
    if scan not in rules["scans"]:
        return "excluded"
    zones = {zone for _, _, zone in rules["scans"][scan]["zones"]}
    if zones <= {"train", "val"}:
        return "train"
    return "mixed" if "dev" in zones else "holdout-A"


def wrap_of(name):
    match = re.search(r"-w(\d+)_", str(name))
    return int(match.group(1)) if match else -1


def available(control):
    """Truth sources on disk: 'names' (truth.npz) and 'groups' (v1 wrap truth)."""
    found = []
    if (V1_CONTROLS / control / "truth.npz").exists():
        found.append("names")
    if (WRAP_TRUTH / f"{control}.npz").exists():
        found.append("groups")
    return found


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.replace(temporary, path)


def write_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def angle(knots, xyz):
    """Angle about the axis and radius; knots are (z, cx, cy) rows in the points' units."""
    cx = np.interp(xyz[:, 2], knots[:, 0], knots[:, 1])
    cy = np.interp(xyz[:, 2], knots[:, 0], knots[:, 2])
    dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
    return np.arctan2(dy, dx), np.hypot(dx, dy)


def centre_of_normals(p, u):
    """Robust point closest to the lines through p along the unit 2D directions u."""
    c = p.mean(axis=0)
    weight = np.ones(len(p))
    for _ in range(12):
        a = np.eye(2)[None] - u[:, :, None] * u[:, None, :]
        lhs = (weight[:, None, None] * a).sum(axis=0)
        rhs = (weight[:, None, None] * a @ p[:, :, None])[..., 0].sum(axis=0)
        c = np.linalg.solve(lhs, rhs)
        miss = (c[0] - p[:, 0]) * u[:, 1] - (c[1] - p[:, 1]) * u[:, 0]
        scale = max(2.0 * float(np.median(np.abs(miss))), 1.0)
        weight = 1.0 / (1.0 + (miss / scale) ** 2)
    return c


def fitted_axis(points, normals, slabs=8, sample=200_000, seed=0):
    """Axis knots from the truth's own normals, one robust centre per z slab."""
    rng = np.random.default_rng(seed)
    horizontal = np.hypot(normals[:, 0], normals[:, 1])
    usable = np.flatnonzero(horizontal > 0.7)
    z = points[usable, 2]
    edges = np.quantile(z, np.linspace(0, 1, slabs + 1))
    knots = []
    for low, high in itertools.pairwise(edges):
        inside = usable[(z >= low) & (z <= high)]
        if len(inside) < 500:
            continue
        pick = rng.choice(inside, min(sample, len(inside)), replace=False)
        u = normals[pick, :2] / horizontal[pick, None]
        c = centre_of_normals(points[pick, :2].astype(np.float64), u.astype(np.float64))
        knots.append(((low + high) / 2, c[0], c[1]))
    return np.array(knots, dtype=np.float64)


def v1_axis(control):
    """v1's axis knots for a control, scaled from node voxels to level-0 voxels; None when absent."""
    for name in (control, f"{control}-tier-a"):
        path = WRAP_TRUTH / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            scale = 2 ** int(data["level"])
            return data["knots"] * scale + 0.5 * (scale - 1)
    return None


def axis(control, points=None, normals=None):
    """Axis knots (z, cx, cy) in level-0 voxels: v1's where it fitted one, else fitted to the truth normals; cached."""
    path = CACHE / control / "axis.json"
    if path.exists():
        cached = json.loads(path.read_text())
        return np.array(cached["knots"]), cached["source"]
    knots, source = v1_axis(control), "v1 wrap truth knots"
    if knots is None:
        if points is None:
            truth = np.load(V1_CONTROLS / control / "truth.npz")
            points, normals = truth["points"] * 2 ** record(control)["level"], truth["normals"]
        knots, source = fitted_axis(points, normals), "fitted to truth normals"
    write_json(path, {"knots": knots.tolist(), "source": source})
    return knots, source


def mesh_folders(sample, volume):
    found = sorted(OPEN_DATA.glob(f"{sample}/segments/*/mesh/*-on-{volume}-*um.tifxyz"))
    return {path.parent.parent.name: path for path in found if (path / "z.tif").exists()}


def spacing_um(rec, names, mesh, thinned_um):
    """Vertex-weighted median spacing of the thinned mesh grids: v1 kept every round(thinned_um * scale / voxel)-th
    vertex of grids whose vertices are 1/scale voxels apart."""
    folders = mesh_folders(rec["sample"], rec["volume"])
    counts = np.bincount(mesh, minlength=len(names))
    values, weights = [], []
    for m, name in enumerate(names):
        if counts[m] and str(name) in folders:
            scale = json.loads((folders[str(name)] / "meta.json").read_text())["scale"][0]
            values.append(max(1, round(thinned_um * scale / rec["scan_voxel_um"])) / scale * rec["scan_voxel_um"])
            weights.append(counts[m])
    if not values:
        return thinned_um
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def mesh_turns(folder, knots, voxel_um, z_range):
    """Vertices of a published mesh grid near the band, thinned like the truth, and their angle unwrapped along the
    winding (the grid direction along which the angle changes)."""
    scale = json.loads((folder / "meta.json").read_text())["scale"][0]
    step = max(1, round(TRUTH_UM * scale / voxel_um))
    grid = np.stack([tifffile.imread(folder / f"{a}.tif")[::step, ::step] for a in "xyz"], axis=-1)
    valid = np.all(grid > 0, axis=-1) & np.all(np.isfinite(grid), axis=-1)
    valid &= (grid[..., 2] >= z_range[0]) & (grid[..., 2] <= z_range[1])
    rows, columns = np.nonzero(valid)
    xyz = grid[rows, columns]
    del grid
    theta, _ = angle(knots, xyz)
    unit = np.exp(1j * theta.astype(np.float32))
    del theta
    lines = []
    for line in (columns, rows):
        total = np.bincount(line, weights=unit.real) + 1j * np.bincount(line, weights=unit.imag)
        count = np.bincount(line)
        lines.append((float(np.sum(np.abs(total))) / max(int(count.sum()), 1), line, total, count > 0))
    _, line, total, used = max(lines, key=lambda item: item[0])
    centre = np.full(len(total), np.nan)
    centre[used] = np.unwrap(np.angle(total[used]))
    unit *= np.exp(-1j * centre[line]).astype(np.complex64)
    return xyz, centre[line] + np.angle(unit)


def handedness(phi, radius, mesh, meshes):
    """+1 when the sheet runs outward along +theta: where a mesh passes its start angle again, it lies further out."""
    votes = []
    window = 0.05 * 2 * np.pi
    for m in meshes:
        mine = mesh == m
        p, r = phi[mine], radius[mine]
        low, high = np.quantile(p, START_QUANTILE), np.quantile(p, 1 - START_QUANTILE)
        for head, tail in (
            ((p >= low) & (p < low + window), (p >= low + 2 * np.pi) & (p < low + 2 * np.pi + window)),
            ((p > high - 2 * np.pi - window) & (p <= high - 2 * np.pi), (p > high - window) & (p <= high)),
        ):
            if head.sum() >= 50 and tail.sum() >= 50:
                votes.append(np.sign(np.median(r[tail]) - np.median(r[head])) * min(head.sum(), tail.sum()))
    return (1 if sum(votes) >= 0 else -1), len(votes)


def group_handedness(control):
    """+1 when, within v1's wrap truth groups, the sheet index Wg - theta/2pi grows outward."""
    data = np.load(WRAP_TRUTH / f"{control}.npz")
    points = np.stack([data["x"], data["y"], data["z"]], axis=1)[::4]
    scale = 2 ** int(data["level"])
    theta, radius = angle(data["knots"] * scale + 0.5 * (scale - 1), points)
    layer = np.round(data["Wg"][::4] - theta / (2 * np.pi))
    group = data["group"][::4]
    vote = 0.0
    for g in np.unique(group):
        mine = group == g
        if mine.sum() > 1000 and np.ptp(layer[mine]) >= 1:
            vote += np.corrcoef(layer[mine], radius[mine])[0, 1] * mine.sum()
    return 1 if vote >= 0 else -1


def name_corrections(control, names):
    """Whole turns added to each named mesh's winding (NAME_TURNS), 0 for the rest."""
    out = np.zeros(len(names))
    for (first, last), turns in NAME_TURNS.get(control, {}).items():
        for m, name in enumerate(names):
            if first <= wrap_of(name) <= last:
                out[m] += turns
    return out


def named_winding(control, points, mesh, names, knots, voxel_um):
    """Continuous winding per truth vertex from names and mesh grids (NaN where unknown), and an audit."""
    rec = record(control)
    folders = mesh_folders(rec["sample"], rec["volume"])
    margin = 2000.0 / voxel_um
    z_range = (float(points[:, 2].min()) - margin, float(points[:, 2].max()) + margin)
    phi = np.full(len(points), np.nan)
    turns, missing = {}, []
    for m, name in enumerate(names):
        mine = np.flatnonzero(mesh == m)
        if wrap_of(name) < 0 or not len(mine):
            continue
        if str(name) not in folders:
            missing.append(str(name))
            continue
        xyz, along = mesh_turns(folders[str(name)], knots, voxel_um, z_range)
        distance, index = cKDTree(xyz).query(points[mine], distance_upper_bound=1.5, workers=4)
        hit = np.isfinite(distance)
        phi[mine[hit]] = along[index[hit]]
        turns[m] = float(np.ptp(phi[mine[hit]]) / (2 * np.pi)) if hit.any() else 0.0
    wraps = np.array([wrap_of(n) for n in names])
    single = [m for m, t in turns.items() if t < MERGED_TURNS]
    merged = [m for m, t in turns.items() if t >= MERGED_TURNS]
    known = np.flatnonzero(np.isfinite(phi))[::4]
    _, radius = angle(knots, points[known].astype(np.float64))
    hand, votes = handedness(phi[known], radius, mesh[known], list(turns))
    hand_source = "mesh overlaps past their start"
    if not votes and (WRAP_TRUTH / f"{control}.npz").exists():
        hand, hand_source = group_handedness(control), "v1 wrap truth groups"
    middle = [float(np.median(radius[mesh[known] == m])) for m in single]
    outward = 1 if len(single) < 2 or np.corrcoef(middle, wraps[single])[0, 1] >= 0 else -1
    del radius
    sense = hand * outward
    tau = sense * phi / (2 * np.pi)
    start = np.zeros(len(names))
    for m in turns:
        values = tau[mesh == m]
        start[m] = np.quantile(values[np.isfinite(values)], START_QUANTILE)
    winding = wraps[mesh] + tau - start[mesh]
    del tau
    shifts, arc = aligned(points, mesh, winding, knots, sense, list(turns))
    anchors = anchored(points, mesh, winding, single, merged, voxel_um)
    corrections = name_corrections(control, names)
    winding += corrections[mesh]
    audit = {
        "version": VERSION,
        "fraction_shifts": {str(names[m]): round(s, 4) for m, s in shifts.items()},
        "fraction_arc": round(arc, 4),
        "sense": int(sense),
        "handedness": int(hand),
        "handedness_votes": votes,
        "handedness_source": hand_source if votes or hand_source.startswith("v1") else "default (no evidence)",
        "names_outward": int(outward),
        "turns": {str(names[m]): round(t, 3) for m, t in turns.items()},
        "merged": [str(names[m]) for m in merged],
        "anchors": {str(names[m]): a for m, a in anchors.items()},
        "name_corrections": {str(names[m]): int(t) for m, t in enumerate(corrections) if t},
        "missing_meshes": missing,
        "matched": round(float(np.isfinite(phi[np.isin(mesh, list(turns))]).mean()) if turns else 0.0, 4),
    }
    return winding, audit


def aligned(points, mesh, winding, knots, sense, meshes):
    """In place, each mesh moved by less than a turn so that winding - sense*theta/2pi has one fraction on all of
    them, the cut between fractions in their widest gap; returns the shifts and the width of the fractions' arc."""
    known = np.flatnonzero(np.isfinite(winding))
    theta, _ = angle(knots, points[known].astype(np.float64))
    unit = np.exp(2j * np.pi * (winding[known] - sense * theta / (2 * np.pi)))
    del theta
    size = int(mesh.max()) + 1
    total = np.bincount(mesh[known], unit.real, size) + 1j * np.bincount(mesh[known], unit.imag, size)
    fraction = {m: float(np.angle(total[m]) / (2 * np.pi)) % 1.0 for m in meshes if abs(total[m]) > 0}
    if not fraction:
        return {}, 0.0
    values = np.sort(np.array(list(fraction.values())))
    gaps = np.diff(np.append(values, values[0] + 1.0))
    start = float(values[(int(np.argmax(gaps)) + 1) % len(values)])
    unwrapped = {m: start + (f - start) % 1.0 for m, f in fraction.items()}
    width = max(unwrapped.values()) - start
    shifts = {m: u - start - width / 2 for m, u in unwrapped.items()}
    for m, s in shifts.items():
        winding[mesh == m] -= s
    return shifts, width


def consistency(points, normals, mesh, winding, voxel_um, spacing, sample=50_000, seed=0):
    """Self-check of a winding on its own vertices: pairs on two meshes at one place should differ by 0, the nearest
    vertex of another sheet along the normal by exactly 1."""
    rng = np.random.default_rng(seed)
    known = np.flatnonzero(np.isfinite(winding) & np.all(np.isfinite(normals), axis=1))
    pick = rng.choice(known, min(sample, len(known)), replace=False)
    tree = cKDTree(points[known])
    distance, found = tree.query(points[pick], k=48, distance_upper_bound=RADIAL_UM / voxel_um, workers=4)
    hit = np.isfinite(distance)
    other = known[np.where(hit, found, 0)]
    offset = points[other].astype(np.float64) - points[pick][:, None, :]
    along = np.abs((offset * normals[pick][:, None, :]).sum(axis=2))
    across = np.sqrt(np.maximum((offset**2).sum(axis=2) - along**2, 0))
    step = winding[other] - winding[pick][:, None]
    coincide = hit & (mesh[other] != mesh[pick][:, None]) & (along <= SAME_SHEET_UM / voxel_um)
    coincide &= across <= spacing / voxel_um
    radial = hit & (along > 2 * SAME_SHEET_UM / voxel_um) & (along >= 0.8 * distance)
    nearest = np.argmin(np.where(radial, distance, np.inf), axis=1)
    has = radial[np.arange(len(pick)), nearest]
    jump = np.abs(step[np.arange(len(pick)), nearest][has])
    same = np.abs(step[coincide])
    return {
        "coincident_pairs": int(coincide.sum()),
        "coincident_within_0.25": round(float(np.mean(same < 0.25)), 4) if len(same) else None,
        "radial_pairs": int(has.sum()),
        "radial_one": round(float(np.mean(np.round(jump) == 1)), 4) if len(jump) else None,
        "radial_within_0.15_of_whole": round(float(np.mean(np.abs(jump - np.round(jump)) <= 0.15)), 4)
        if len(jump)
        else None,
    }


def anchored(points, mesh, winding, single, merged, voxel_um):
    """Whole-turn shift of each multi-wrap mesh (largest first) agreeing best with the meshes already placed that it
    lies on; in place on winding. Few pairs (touching neighbours, not the same sheet) leave a mesh at its name."""
    out = {}
    placed = list(single)
    sizes = {m: int((mesh == m).sum()) for m in merged}
    for m in sorted(merged, key=lambda k: -sizes[k]):
        reference = np.flatnonzero(np.isin(mesh, placed) & np.isfinite(winding))
        mine = np.flatnonzero((mesh == m) & np.isfinite(winding))[::4]
        placed.append(m)
        if not len(reference) or not len(mine):
            out[m] = {"offset": None, "pairs": 0, "agreement": None}
            continue
        distance, index = cKDTree(points[reference]).query(
            points[mine], distance_upper_bound=SAME_SHEET_UM / voxel_um, workers=4
        )
        hit = np.isfinite(distance)
        if hit.sum() < max(ANCHOR_LEAST, ANCHOR_SHARE * len(mine)):
            out[m] = {"offset": None, "pairs": int(hit.sum()), "agreement": None}
            continue
        delta = np.round(winding[reference[index[hit]]] - winding[mine[hit]]).astype(np.int64)
        values, counts = np.unique(delta, return_counts=True)
        best = int(np.argmax(counts))
        out[m] = {
            "offset": int(values[best]),
            "pairs": int(hit.sum()),
            "agreement": round(float(counts[best] / hit.sum()), 4),
        }
        winding[mesh == m] += values[best]
    return out


def duplicates(points, mesh, winding, voxel_um):
    """Vertices lying on a larger mesh at the same winding (within 40 µm and half a turn): the same sheet twice."""
    known = np.flatnonzero(np.isfinite(winding))
    rank = np.argsort(np.argsort(-np.bincount(mesh, minlength=int(mesh.max()) + 1), kind="stable"), kind="stable")
    tree = cKDTree(points[known])
    out = np.zeros(len(points), dtype=bool)
    for start in range(0, len(known), CHUNK):
        rows = known[start : start + CHUNK]
        distance, found = tree.query(points[rows], k=8, distance_upper_bound=SAME_SHEET_UM / voxel_um, workers=4)
        hit = np.isfinite(distance)
        other = known[np.where(hit, found, 0)]
        larger = rank[mesh[other]] < rank[mesh[rows]][:, None]
        same = np.abs(winding[other] - winding[rows][:, None]) < 0.5
        out[rows] = (hit & larger & same).any(axis=1)
    return out


def grid_normals(points, mesh, row, col):
    """Unit normals of vertices on their mesh's thinned grid from row and column neighbours; NaN where none."""
    out = np.full((len(points), 3), np.nan, dtype=np.float32)
    for m in np.unique(mesh):
        index = np.flatnonzero(mesh == m)
        r, c = row[index] + 1, col[index] + 1
        ids = np.full((int(r.max()) + 2, int(c.max()) + 2), -1, dtype=np.int64)
        ids[r, c] = index
        tangents = []
        for dr, dc in ((1, 0), (0, 1)):
            ahead, behind = ids[r + dr, c + dc], ids[r - dr, c - dc]
            t = points[np.where(ahead >= 0, ahead, index)].astype(np.float64)
            t -= points[np.where(behind >= 0, behind, index)]
            t[(ahead < 0) & (behind < 0)] = np.nan
            tangents.append(t)
        n = np.cross(tangents[0], tangents[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[index] = n / np.linalg.norm(n, axis=1, keepdims=True)
    return out


def in_box(points, box):
    if box is None:
        return np.ones(len(points), dtype=bool)
    return np.all((points >= np.asarray(box[0])) & (points < np.asarray(box[1])), axis=1)


def names_truth(control, every=1, box=None, dedupe=True):
    rec = record(control)
    data = np.load(V1_CONTROLS / control / "truth.npz")
    names, mesh = data["names"], data["mesh"]
    points = data["points"] * np.float32(2 ** rec["level"])
    knots, axis_source = axis(control, points, data["normals"])
    mesh_wrap = np.array([wrap_of(n) for n in names], dtype=np.int64)
    spacing = spacing_um(rec, names, mesh, TRUTH_UM)
    winding, audit = np.full(len(points), np.nan), {}
    twice = np.zeros(len(points), dtype=bool)
    if (mesh_wrap >= 0).any():
        folder = CACHE / control
        cached, doubled, audit_path = (
            folder / "names-winding.npy",
            folder / "names-duplicate.npy",
            folder / "names.json",
        )
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if audit.get("version") != VERSION or not cached.exists() or not doubled.exists():
            voxel_um = rec["scan_voxel_um"]
            winding, audit = named_winding(control, points, mesh, names, knots, voxel_um)
            double = duplicates(points, mesh, winding, voxel_um)
            kept = np.where(double, np.nan, winding)
            audit["consistency"] = consistency(points, data["normals"], mesh, kept, voxel_um, spacing)
            write_array(cached, winding)
            write_array(doubled, double)
            write_json(audit_path, audit)
        winding = np.load(cached)
        twice = np.load(doubled) if dedupe else twice
        audit = audit | {"duplicates_dropped": int(twice.sum())}
    if box is None and every == 1 and not twice.any():
        keep = slice(None)
    else:
        keep = np.flatnonzero(in_box(points, box) & ~twice)[::every]
    merged = np.isin(names, audit.get("merged", []))
    return {
        "control": control,
        "source": "names",
        "points": points[keep],
        "normals": data["normals"][keep],
        "mesh": mesh[keep],
        "names": names,
        "mesh_wrap": mesh_wrap,
        "merged": merged,
        "winding": winding[keep],
        "sense": audit.get("sense", 1),
        "frame": np.where(np.isfinite(winding[keep]), 0, -1).astype(np.int32),
        "unresolved": None,
        "knots": knots,
        "axis_source": axis_source,
        "audit": audit,
        "scan_voxel_um": float(rec["scan_voxel_um"]),
        "spacing_um": spacing * every**0.5,
        "record": rec,
    }


def groups_truth(control, every=1, box=None):
    rec = record(control)
    data = np.load(WRAP_TRUTH / f"{control}.npz")
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    names, mesh = data["names"], data["mesh"]
    spacing = spacing_um(rec, names, mesh, GROUPS_UM)
    normals = grid_normals(points, mesh, data["row"], data["col"])
    keep = np.flatnonzero(in_box(points, box))[::every]
    scale = 2 ** int(data["level"])
    mesh, points = mesh[keep], points[keep]
    return {
        "control": control,
        "source": "groups",
        "points": points,
        "normals": normals[keep],
        "mesh": mesh,
        "names": names,
        "mesh_wrap": np.array([wrap_of(n) for n in names], dtype=np.int64),
        "merged": np.zeros(len(names), dtype=bool),
        "winding": data["Wg"][keep].astype(np.float64),
        "sense": 1,
        "frame": data["group"][keep].astype(np.int32),
        "unresolved": data["unresolved"][keep],
        "knots": data["knots"] * scale + 0.5 * (scale - 1),
        "axis_source": "v1 wrap truth knots",
        "audit": {},
        "scan_voxel_um": float(rec["scan_voxel_um"]),
        "spacing_um": spacing * every**0.5,
        "record": rec,
    }


def default_source(control):
    found = available(control)
    if "names" in found:
        names = np.load(V1_CONTROLS / control / "truth.npz")["names"]
        if max(wrap_of(n) for n in names) >= 0 or "groups" not in found:
            return "names"
    return "groups" if "groups" in found else None


def load(control, source="auto", every=1, box=None, dedupe=True):
    """Truth vertices of a control, optionally thinned and boxed (box = ((x, y, z) low, high), level-0 voxels); with
    wrap names, vertices repeating a larger mesh's sheet at the same winding are dropped unless dedupe is False."""
    source = default_source(control) if source == "auto" else source
    if source == "names":
        return names_truth(control, every, box, dedupe)
    if source == "groups":
        return groups_truth(control, every, box)
    raise FileNotFoundError(f"no truth for {control}")


def layers(truth, theta, psi):
    """Box-local sheet index (constant on a sheet within a box that does not encircle the axis) and the angle used."""
    relative = np.angle(np.exp(1j * (theta - psi)))
    return truth["winding"] - truth["sense"] * relative / (2 * np.pi), relative


def summary(control, source="auto"):
    truth = load(control, source)
    known = np.isfinite(truth["winding"])
    out = {
        "control": control,
        "split": split(control),
        "source": truth["source"],
        "available": available(control),
        "vertices": len(truth["points"]),
        "meshes": len(truth["names"]),
        "named_meshes": int((truth["mesh_wrap"] >= 0).sum()),
        "with_winding": round(float(known.mean()), 4),
        "frames": int(len(np.unique(truth["frame"][truth["frame"] >= 0]))),
        "axis": truth["axis_source"],
        "scan_voxel_um": truth["scan_voxel_um"],
        "spacing_um": round(truth["spacing_um"], 1),
        "z_range": [float(truth["points"][:, 2].min()), float(truth["points"][:, 2].max())],
    }
    audit = truth["audit"]
    if audit:
        out |= {
            k: audit.get(k)
            for k in (
                "fraction_arc",
                "consistency",
                "sense",
                "handedness",
                "handedness_votes",
                "handedness_source",
                "names_outward",
                "merged",
                "anchors",
                "missing_meshes",
                "matched",
                "duplicates_dropped",
            )
        }
    if truth["unresolved"] is not None:
        out["unresolved"] = round(float(truth["unresolved"].mean()), 4)
    return out


def agreement(control, reach_um=SAME_SHEET_UM):
    """Name windings against v1's group windings at the same places: share on each group's modal offset."""
    names, groups = load(control, "names"), load(control, "groups")
    known = np.flatnonzero(np.isfinite(names["winding"]))
    distance, index = cKDTree(groups["points"]).query(
        names["points"][known], distance_upper_bound=reach_um / names["scan_voxel_um"], workers=4
    )
    hit = np.isfinite(distance)
    mine, theirs = known[hit], index[hit]
    offset = np.round(names["winding"][mine] - names["sense"] * groups["winding"][theirs]).astype(np.int64)
    frame = groups["frame"][theirs]
    key = offset * 100_000 + frame
    right = sum(np.bincount(offset[frame == g] - offset[frame == g].min()).max() for g in np.unique(frame))
    by_mesh = {}
    for m in np.unique(names["mesh"][mine]):
        s = names["mesh"][mine] == m
        by_mesh[str(names["names"][m])[:24]] = round(float(np.unique(key[s], return_counts=True)[1].max() / s.sum()), 4)
    return {
        "control": control,
        "matched": int(hit.sum()),
        "named_with_winding": len(known),
        "share_on_group_offset": round(float(right / max(hit.sum(), 1)), 4),
        "groups": int(len(np.unique(frame))),
        "by_mesh": by_mesh,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", nargs="*", help="control names (default: all)")
    parser.add_argument("--source", default="auto", choices=("auto", "names", "groups"))
    parser.add_argument("--list", action="store_true", help="only list controls, their split and truth files")
    parser.add_argument("--agree", action="store_true", help="name windings against v1's group windings")
    args = parser.parse_args()
    for control in args.controls or controls():
        if args.list:
            print(f"{control:48s} {split(control):9s} {','.join(available(control))}")
        elif args.agree:
            print(json.dumps(agreement(control)), flush=True)
        else:
            print(json.dumps(summary(control, args.source)), flush=True)


if __name__ == "__main__":
    main()
