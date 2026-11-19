"""A winding field scored at the truth's mesh vertices: does W step by exactly one per wrap, everywhere it is valid?

The field and the truth are compared as box-local sheet indices. For a 'level' field (sheets are level sets
W = integer + const) the index is W itself; for an 'angle' field (W = theta/2pi + integer, v1's convention) it is
W - theta/2pi with theta taken about the scroll axis relative to the box's mean direction. The truth's index comes
from its continuous winding the same way (truth.layers). d = field index - chirality * truth index then sits on
integer + phase where the field is right; each mesh has its own phase (its truth winding starts where its segmenter
started), and the offset of a vertex is round(d - its mesh's phase). The chirality with the larger share is kept.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from calibration.bench import results
from calibration.bench import truth as truths

CHUNK = 1_000_000
PHASE_BINS = 20
CURVE_POINTS = 20
WRONG_RATES = (0.001, 0.01, 0.05)
SAMPLE = 100_000
PRESSED_UM = 80.0
LATERAL_UM = 160.0
NEIGHBOUR_UM = 400.0
ENCIRCLING_DEG = 300.0
CUT_STEP_DEG = 5.0
CUT_SAMPLE = 200_000


def grid_field(values, origin_zyx, spacing, valid=None, quality=None, nearest=False):
    """Callable sampling a (z, y, x) grid whose sample [0, 0, 0] sits at origin_zyx (level-0 voxels) with the given
    level-0 spacing; trilinear unless nearest, valid where every used sample is finite (and in `valid`)."""
    origin = np.asarray(origin_zyx, dtype=np.float64)
    shape = np.array(values.shape)

    def sample(xyz, normals=None):
        out = {
            "W": np.full(len(xyz), np.nan),
            "valid": np.zeros(len(xyz), dtype=bool),
            "inside": np.zeros(len(xyz), dtype=bool),
            "quality": None if quality is None else np.full(len(xyz), np.nan, dtype=np.float32),
        }
        for start in range(0, len(xyz), CHUNK):
            part = slice(start, start + CHUNK)
            g = (xyz[part][:, ::-1].astype(np.float64) - origin) / spacing
            inside = np.all((g >= 0) & (g <= shape - 1), axis=1)
            g = np.clip(g, 0, shape - 1)
            if nearest:
                corners = [(np.rint(g).astype(np.int64), np.ones(len(g)))]
            else:
                base = np.minimum(np.floor(g).astype(np.int64), np.maximum(shape - 2, 0))
                frac = g - base
                corners = []
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            step = np.array([dz, dy, dx])
                            weight = np.prod(np.where(step, frac, 1 - frac), axis=1)
                            corners.append((np.minimum(base + step, shape - 1), weight))
            w = np.zeros(len(g))
            q = np.zeros(len(g))
            ok = inside.copy()
            for at, weight in corners:
                v = values[at[:, 0], at[:, 1], at[:, 2]].astype(np.float64)
                used = weight > 0
                ok &= ~used | np.isfinite(v)
                if valid is not None:
                    ok &= ~used | valid[at[:, 0], at[:, 1], at[:, 2]]
                w += np.where(used, weight * np.nan_to_num(v), 0)
                if quality is not None:
                    q += weight * quality[at[:, 0], at[:, 1], at[:, 2]]
            out["W"][part] = np.where(ok, w, np.nan)
            out["valid"][part] = ok
            out["inside"][part] = inside
            if quality is not None:
                out["quality"][part] = q
        return out

    return sample


def circular(values):
    """Phase in [-0.5, 0.5) and resultant length of values taken modulo 1."""
    resultant = np.mean(np.exp(2j * np.pi * values))
    return float(np.angle(resultant) / (2 * np.pi)), float(np.abs(resultant))


def group_means(unit, key):
    """Circular mean phase per element's group and the group's resultant length."""
    _, inverse = np.unique(key, return_inverse=True)
    total = np.bincount(inverse, unit.real) + 1j * np.bincount(inverse, unit.imag)
    count = np.bincount(inverse)
    return (np.angle(total) / (2 * np.pi))[inverse], (np.abs(total) / count)[inverse]


def phases(d, coarse, fine):
    """Phase of d per fine group (a truth mesh carries its own fraction) placed within half a turn of its coarse
    group's phase, so one cut serves the whole coarse group; and the mean within-fine-group strength."""
    unit = np.exp(2j * np.pi * d)
    coarse = np.asarray(coarse, dtype=np.int64)
    reference, _ = group_means(unit, coarse)
    value, length = group_means(unit, coarse * (int(np.max(fine)) + 1) + fine)
    shift = value - reference
    return reference + shift - np.round(shift), float(length.mean())


def modal(offset, key):
    """Per key, its commonest offset; returns (mode per element, right mask)."""
    order = np.lexsort((offset, key))
    k, o = key[order], offset[order]
    boundary = np.flatnonzero((np.diff(k) != 0) | (np.diff(o) != 0)) + 1
    starts = np.concatenate([[0], boundary])
    lengths = np.diff(np.append(starts, len(k)))
    run_key, run_offset = k[starts], o[starts]
    best = {}
    for rk, ro, n in zip(run_key.tolist(), run_offset.tolist(), lengths.tolist()):
        if n > best.get(rk, (None, -1))[1]:
            best[rk] = (ro, n)
    keys = np.array(list(best), dtype=np.int64)
    modes = np.array([v[0] for v in best.values()], dtype=np.int64)
    lookup = modes[np.searchsorted(keys, key)] if len(keys) else np.zeros(0, dtype=np.int64)
    return lookup, offset == lookup


def distribution(error):
    values, counts = np.unique(np.clip(error, -3, 3), return_counts=True)
    total = max(int(counts.sum()), 1)
    return {("other" if abs(v) == 3 else str(int(v))): round(int(c) / total, 4) for v, c in zip(values, counts)}


def risk_coverage(quality, wrong, boxed):
    """Coverage of the box's truth against wrong rate as the quality floor rises."""
    order = np.argsort(-quality, kind="stable")
    wrong_sorted = wrong[order]
    counts = np.arange(1, len(order) + 1)
    rate = np.cumsum(wrong_sorted) / counts
    curve = []
    for share in np.linspace(1.0, 1.0 / CURVE_POINTS, CURVE_POINTS):
        n = max(1, int(round(share * len(order))))
        curve.append(
            {
                "quality_floor": round(float(quality[order[n - 1]]), 5),
                "coverage": round(n / boxed, 4),
                "wrong_rate": round(float(rate[n - 1]), 5),
            }
        )
    at = {}
    for limit in WRONG_RATES:
        fine = np.flatnonzero(rate <= limit)
        at[str(limit)] = round(float((fine.max() + 1) / boxed), 4) if len(fine) else 0.0
    return {"curve": curve, "coverage_at_wrong_rate": at}


def neighbours(points, normals, index_truth, voxel_um, chirality, index_field, rng):
    """Radial truth neighbours on the next meshed sheet, and vertices with another sheet within 80 µm."""
    pick = rng.choice(len(points), min(SAMPLE, len(points)), replace=False)
    tree = cKDTree(points.astype(np.float64))
    distance, found = tree.query(
        points[pick].astype(np.float64), k=32, distance_upper_bound=NEIGHBOUR_UM / voxel_um, workers=4
    )
    hit = np.isfinite(distance)
    other = np.where(hit, found, 0)
    offset = points[other].astype(np.float64) - points[pick][:, None, :]
    along = np.abs((offset * normals[pick][:, None, :]).sum(axis=2))
    across = np.sqrt(np.maximum((offset**2).sum(axis=2) - along**2, 0))
    step = index_truth[other] - index_truth[pick][:, None]
    other_sheet = hit & (np.abs(step) >= 0.5)
    pressed = (other_sheet & (along <= PRESSED_UM / voxel_um) & (across <= LATERAL_UM / voxel_um)).any(axis=1)
    radial = other_sheet & (np.abs(np.round(step)) == 1) & (along >= 0.8 * np.sqrt(along**2 + across**2))
    radial_distance = np.where(radial, distance, np.inf)
    best = np.argmin(radial_distance, axis=1)
    has = np.isfinite(radial_distance[np.arange(len(pick)), best])
    q = other[np.arange(len(pick)), best][has]
    p = pick[has]
    expected = chirality * np.round(index_truth[q] - index_truth[p])
    got = np.round(index_field[q] - index_field[p]) * expected
    values, counts = np.unique(np.clip(got, -1, 3), return_counts=True)
    names = {-1: "backward", 0: "same sheet", 1: "one", 2: "two", 3: "three or more"}
    return (
        pick,
        pressed,
        {
            "pairs": int(has.sum()),
            "exact": round(float(np.mean(got == 1)) if has.any() else 0.0, 4),
            "steps": {names[int(v)]: round(int(c) / max(int(has.sum()), 1), 4) for v, c in zip(values, counts)},
            "sampled": len(pick),
        },
    )


def trial(d, sub):
    """Share on each frame's modal offset of d once each mesh's phase is removed, with what it was computed from."""
    phase, strength = phases(d, sub["frame"], sub["mesh"])
    offset = np.round(d - phase).astype(np.int64)
    _, right = modal(offset, sub["frame"].astype(np.int64))
    return float(right.mean()), d, phase, strength, offset


def level_cut(W, sub, theta, rng):
    """Direction psi whose opposite, as the truth's branch cut, best matches a level field around the whole axis: a
    level field must jump somewhere on every turn, and where is the field's choice, not an error."""
    pick = rng.choice(len(W), min(CUT_SAMPLE, len(W)), replace=False)
    part = {k: v[pick] if isinstance(v, np.ndarray) else v for k, v in sub.items()}
    best = (-1.0, 0.0)
    for psi in np.radians(np.arange(0.0, 360.0, CUT_STEP_DEG)):
        index_truth, _ = truths.layers(part, theta[pick], psi)
        share = max(trial(W[pick] - c * index_truth, part)[0] for c in (1, -1))
        best = max(best, (share, psi))
    return best[1]


def v1_footing(offset, frame, resolved):
    """v1's ruled-roll one-offset share (error_map, quick_truth): modes taken on resolved truth only, in the largest
    group or per group; vertex-weighted, where v1 counted each truth node once."""
    frame = frame.astype(np.int64)
    largest = np.bincount(frame[resolved] - frame.min()).argmax() + frame.min()
    big = resolved & (frame == largest)
    _, right_big = modal(offset[big], np.zeros(int(big.sum()), dtype=np.int64))
    _, right_all = modal(offset[resolved], frame[resolved])
    return {
        "share_largest_frame_resolved": round(float(right_big.mean()), 4),
        "share_frames_resolved": round(float(right_all.mean()), 4),
        "resolved_vertices": int(resolved.sum()),
    }


def score(truth, field, convention="level", errors=None, seed=0, cut=None):
    """Metrics of a field on a truth (truth.load); field(xyz, normals) returns W, valid, inside and quality (or None).
    cut: the truth's branch cut angle in degrees about the axis, 'search' for the one matching the field best, or None
    (opposite the box's mean direction, searched for a level field around the whole axis)."""
    rng = np.random.default_rng(seed)
    known = np.flatnonzero(np.isfinite(truth["winding"]))
    sampled = field(truth["points"][known], truth["normals"][known])
    inside = sampled["inside"]
    boxed = int(inside.sum())
    use = known[sampled["valid"]]
    out = {"truth_in_box": boxed, "truth_valid": len(use), "coverage": round(len(use) / max(boxed, 1), 4)}
    if len(use) < 100:
        return out | {"error": "fewer than 100 truth vertices with a valid field"}
    W = sampled["W"][sampled["valid"]]
    quality = None if sampled["quality"] is None else sampled["quality"][sampled["valid"]]
    points = truth["points"][use]
    theta, _ = truths.angle(truth["knots"], points.astype(np.float64))
    psi = float(np.angle(np.mean(np.exp(1j * theta))))
    sub = {k: truth[k][use] for k in ("winding", "mesh", "frame")} | {"sense": truth["sense"]}
    span = float(np.degrees(np.ptp(truths.layers(sub, theta, psi)[1])))
    encircles = span > ENCIRCLING_DEG
    searched = cut == "search" or (cut is None and convention == "level" and encircles)
    if searched:
        psi = level_cut(W, sub, theta, rng)
    elif cut is not None:
        psi = float(np.radians(cut)) - np.pi
    index_truth, relative = truths.layers(sub, theta, psi)
    index_field = W - relative / (2 * np.pi) if convention == "angle" else W
    trials = {c: trial(index_field - c * index_truth, sub) for c in (1, -1)}
    chirality = max(trials, key=lambda c: trials[c][0])
    _, d, phase, strength, offset = trials[chirality]
    residual = d - phase - offset
    phase_all, strength_all = circular(d)
    frame_mode, frame_right = modal(offset, sub["frame"].astype(np.int64))
    global_mode, global_right = modal(offset, np.zeros(len(offset), dtype=np.int64))
    mesh_mode, mesh_right = modal(offset, sub["mesh"].astype(np.int64))
    histogram = np.histogram(residual, bins=PHASE_BINS, range=(-0.5, 0.5))[0]
    out |= {
        "convention": convention,
        "chirality": chirality,
        "share_other_chirality": round(trials[-chirality][0], 4),
        "box_angle_span_deg": round(span, 1),
        "encircles_axis": encircles,
        "truth_cut_deg": round(float(np.degrees(np.angle(np.exp(1j * (psi + np.pi))))), 1),
        "truth_cut_searched": searched,
        "phase": round(phase_all, 4),
        "phase_strength": round(strength_all, 4),
        "phase_strength_within_mesh": round(strength, 4),
        "phase_within_0.1": round(float(np.mean(np.abs(residual) <= 0.1)), 4),
        "phase_within_0.25": round(float(np.mean(np.abs(residual) <= 0.25)), 4),
        "phase_histogram": (histogram / histogram.sum()).round(4).tolist(),
        "share_global": round(float(global_right.mean()), 4),
        "share_frames": round(float(frame_right.mean()), 4),
        "share_mesh": round(float(mesh_right.mean()), 4),
        "frames": int(len(np.unique(sub["frame"]))),
        "offsets_global": distribution(offset - global_mode),
    }
    frames = np.unique(sub["frame"], return_counts=True)
    largest = frames[0][np.argmax(frames[1])]
    out["share_largest_frame"] = round(float(frame_right[sub["frame"] == largest].mean()), 4)
    per_mesh = []
    for m in np.unique(sub["mesh"]):
        mine = sub["mesh"] == m
        per_mesh.append(
            {
                "mesh": str(truth["names"][m]),
                "wrap": int(truth["mesh_wrap"][m]),
                "vertices": int(mine.sum()),
                "share_on_global": round(float(global_right[mine].mean()), 4),
                "share_on_own": round(float(mesh_right[mine].mean()), 4),
                "own_minus_global": int(mesh_mode[mine][0] - global_mode[mine][0]),
            }
        )
    out["per_mesh"] = sorted(per_mesh, key=lambda r: (r["wrap"], r["mesh"]))
    pick, pressed, adjacency = neighbours(
        points, truth["normals"][use], index_truth, truth["scan_voxel_um"], chirality, index_field, rng
    )
    out["adjacency"] = adjacency
    if truth["unresolved"] is not None:
        pressed_all = truth["unresolved"][use]
        out["pressed"] = {"source": "v1 unresolved", "share_of_truth": round(float(pressed_all.mean()), 4)}
    else:
        pressed_all = None
        out["pressed"] = {"source": f"another truth sheet within {PRESSED_UM:.0f} µm, sampled", "sampled": len(pick)}
        out["pressed"]["share_of_truth"] = round(float(pressed.mean()), 4)
    flags = pressed_all if pressed_all is not None else np.zeros(len(use), dtype=bool)
    rows = np.arange(len(use)) if pressed_all is not None else pick
    if pressed_all is None:
        flags[pick] = pressed
    for name, mask in (("resolved", ~flags[rows]), ("pressed", flags[rows])):
        out["pressed"][f"share_frames_{name}"] = round(float(frame_right[rows][mask].mean()), 4) if mask.any() else None
    if pressed_all is not None and (~pressed_all).any():
        out["v1_footing"] = v1_footing(offset, sub["frame"], ~pressed_all)
    if quality is not None:
        out["risk_coverage_frames"] = risk_coverage(quality, ~frame_right, boxed)
        out["risk_coverage_mesh"] = risk_coverage(quality, ~mesh_right, boxed)
    if errors:
        truths.write_npz(
            errors,
            points=points,
            mesh=sub["mesh"],
            error=np.clip(offset - frame_mode, -127, 127).astype(np.int8),
            d=d.astype(np.float32),
            quality=quality if quality is not None else np.zeros(0, np.float32),
        )
    return out


def headline(out):
    """Flat metrics stored in the results database."""
    keys = (
        "coverage",
        "truth_in_box",
        "truth_valid",
        "chirality",
        "phase_strength",
        "phase_strength_within_mesh",
        "share_global",
        "share_frames",
        "share_mesh",
        "share_largest_frame",
    )
    flat = {k: float(out[k]) for k in keys if k in out}
    if "adjacency" in out:
        flat["adjacent_exact"] = out["adjacency"]["exact"]
    for k, v in out.get("v1_footing", {}).items():
        if k.startswith("share"):
            flat[f"v1_{k}"] = v
    if "pressed" in out:
        for k in ("share_of_truth", "share_frames_resolved", "share_frames_pressed"):
            if out["pressed"].get(k) is not None:
                flat[f"pressed_{k}"] = out["pressed"][k]
    if "risk_coverage_frames" in out:
        for rate, value in out["risk_coverage_frames"]["coverage_at_wrong_rate"].items():
            flat[f"coverage_at_wrong_{rate}"] = value
    return flat


def parse_triple(text):
    return tuple(float(v) for v in text.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control")
    parser.add_argument("--field", required=True, help="W grid .npy, axes (z, y, x), float32, NaN where invalid")
    parser.add_argument(
        "--origin",
        required=True,
        type=parse_triple,
        help="level-0 z,y,x of grid sample [0,0,0] (voxel i of a level-L array is centred at i*2^L + (2^L-1)/2)",
    )
    parser.add_argument(
        "--cut",
        help="truth branch cut in degrees about the axis, or 'search' (default: searched for a level field "
        "around the whole axis, else opposite the box's mean direction)",
    )
    parser.add_argument("--level", type=int, default=0, help="grid level: spacing 2**level level-0 voxels")
    parser.add_argument("--spacing", type=float, help="grid spacing in level-0 voxels (overrides --level)")
    parser.add_argument("--valid", help="bool .npy of the grid's shape")
    parser.add_argument("--quality", help="float .npy of the grid's shape (risk-coverage)")
    parser.add_argument("--convention", default="level", choices=("level", "angle"))
    parser.add_argument("--truth", default="auto", choices=("auto", "names", "groups"))
    parser.add_argument("--nearest", action="store_true", help="nearest-sample instead of trilinear")
    parser.add_argument("--every", type=int, default=1, help="use every n-th truth vertex")
    parser.add_argument("--errors", help="write the per-vertex error map to this .npz")
    parser.add_argument("--method", help="name of the run in the results database (default: field file stem)")
    parser.add_argument("--system", default="network")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    values = np.load(args.field, mmap_mode="r")
    spacing = args.spacing or 2.0**args.level
    origin = np.array(args.origin)
    high = origin + spacing * (np.array(values.shape) - 1)
    box = (origin[::-1], high[::-1] + 1e-3)
    truth = truths.load(args.control, args.truth, args.every, box)
    field = grid_field(
        values,
        origin,
        spacing,
        None if args.valid is None else np.load(args.valid, mmap_mode="r"),
        None if args.quality is None else np.load(args.quality, mmap_mode="r"),
        args.nearest,
    )
    cut = args.cut if args.cut in (None, "search") else float(args.cut)
    out = score(truth, field, args.convention, args.errors, cut=cut)
    out |= {"control": args.control, "truth": truth["source"], "split": truths.split(args.control)}
    print(json.dumps(out))
    if not args.no_record and "error" not in out:
        parameters = vars(args) | {"box_xyz": [list(map(float, b)) for b in box], "spacing": spacing}
        run = results.record(
            args.control,
            args.system,
            args.method or Path(args.field).stem,
            "winding",
            headline(out),
            parameters,
            detail=out,
        )
        print(f"recorded run {run}")


if __name__ == "__main__":
    main()
