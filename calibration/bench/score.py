"""Surfaces scored against a control package (bench.package): distance, winding, area, joins, folds.

Input: tifxyz folders (or folders of them), level-0 x, y, z; quality.npy optional (1 locked, 2 interpolated). A
surface's winding is, per vertex, w.tif when present, else meta.json's integer `winding` plus the turn travelled
about the axis from a cut angle: W = winding + chirality * ((angle - cut) mod one turn), the turn unwrapped along the
surface's grid and counted from the cut its median lies past. `cut_deg` and `chirality` in meta.json are used when
given; otherwise the cut is fitted in 5 degree steps and the chirality (sign of `winding` against the truth) is
fitted too. Without any winding a surface only gets its own offset (local mode: switches inside a surface).

Matching. A truth vertex is covered by a surface vertex within 40 µm along the truth normal and within 0.75 of the
surface's grid step across; a surface vertex is on truth when a truth vertex lies within 40 µm along that truth
vertex's normal and 0.75 of the truth spacing across. The rank error of a matched pair is
round(W_truth - a * W_surface - c) minus its frame's modal value: one global offset per truth frame (named truth has
one frame), a = +-1, c the common fraction. Truth vertices flagged conflict or order (the truth contradicts itself)
are unjudged; duplicates count once in areas.

Metrics, over the zones scored (dev unless --holdout):
- distance: truth to surface along the truth normal (covered truth, and all truth within 300 µm), surface to truth,
  median and p90 in voxels
- winding: share of judged covered truth with rank error 0, pressed and loose apart; crossings the same
- area: surface cm2 on truth with rank error 0 (correct), with another (wrong), near truth but off it, far from
  truth; truth cm2 covered and covered correctly; largest 8-connected correct cm2 of one surface
- recall: share of the truth area covered, covered on the right winding, and covered where the truth is pressed
  (another winding within 80 µm) and where it is loose: gaps at contacts show as a lower pressed recall
- joins: share of surface grid edges between two judged vertices whose rank errors differ (wrong joins)
- folds: grid steps more than 10% off the surface's median step (non-isometry); vertices whose normal turns against
  the surface's majority side of the axis (flipped), and grid edges whose two normals point apart (folded)

Writes truth_rank.npy (int8 per truth vertex of truth.npz, 127 not covered or not scored), truth_depth.npy,
crossing_rank.npy (per crossing of crossings.npz), surface_labels.npz and summary.json into the score folder.
Holdout zones need --holdout and a clean commit; every holdout run is appended to holdout-log.jsonl.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from calibration.bench import package as packages
from calibration.bench import results
from calibration.bench import truth as truths
from calibration.bench.nearest import on_sheet
from calibration.bench.surface_score import load_surface, surfaces_in, unwrapped, vertex_area
from storage.data_roots import HERE as CODE
from storage.data_roots import SURFACE_BENCH as BENCH

TOLERANCE_UM = 40.0
FAR_UM = 300.0
LATERAL_SHARE = 0.75
ISOMETRY = 0.10
CUT_STEP_DEG = 5.0
UNSCORED = 127
KEEP = 3
SCORES = BENCH / "scores"
HOLDOUT_LOG = BENCH / "holdout-log.jsonl"
SKIP = packages.CONFLICT | packages.ORDER


def load(path, mask):
    surface = load_surface(path, mask)
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    surface["meta"] = meta
    surface["w"] = tifffile.imread(path / "w.tif").astype(np.float64) if (path / "w.tif").exists() else None
    return surface


def grid_normals(xyz, mask):
    return packages.grid_normals(xyz.astype(np.float64), mask)


def toward_axis(knots, xyz, normals):
    cx = np.interp(xyz[:, 2], knots[:, 0], knots[:, 1])
    cy = np.interp(xyz[:, 2], knots[:, 0], knots[:, 2])
    return (cx - xyz[:, 0]) * normals[:, 0] + (cy - xyz[:, 1]) * normals[:, 1]


def modal(values, key):
    """Modal value per key and a boolean of entries equal to their key's mode."""
    out = np.zeros(len(values), dtype=bool)
    modes = {}
    for k in np.unique(key):
        mine = key == k
        v, c = np.unique(values[mine], return_counts=True)
        modes[int(k)] = int(v[np.argmax(c)])
        out[mine] = values[mine] == modes[int(k)]
    return modes, out


def circular_fraction(x):
    angle = np.angle(np.mean(np.exp(2j * np.pi * x)))
    return float(angle / (2 * np.pi))


def surface_winding(s, turn, sense, a, cut):
    """Per masked vertex: W of the surface for chirality a (sign of its integer against the truth) and cut angle."""
    if s["w"] is not None:
        return s["w"][s["mask"]]
    if "winding" not in s["meta"]:
        return None
    u = a * sense * (turn - cut) / (2 * np.pi)
    u -= np.floor(np.median(u))
    return float(s["meta"]["winding"]) + u


def fit_alignment(surfaces, pairs, truth, sense, given):
    """Chirality a, cut angle and the common fraction maximising the share of pairs on their frame's modal offset."""
    best = None
    chiralities = [given["chirality"]] if given.get("chirality") is not None else [1, -1]
    if given.get("cut_deg") is not None:
        cuts = [np.deg2rad(given["cut_deg"])]
    elif all(s["w"] is not None for s in surfaces):
        cuts = [0.0]
    else:
        cuts = np.deg2rad(np.arange(0, 360, CUT_STEP_DEG))
    surface_of, vertex_of, truth_of = pairs
    w_truth = truth["winding"][truth_of]
    frame = truth["frame"][truth_of]
    for a in chiralities:
        for cut in cuts:
            w = np.full(len(surface_of), np.nan)
            for i, s in enumerate(surfaces):
                mine = surface_of == i
                if not mine.any():
                    continue
                ws = surface_winding(s, s["turn"], sense, a, cut)
                if ws is None:
                    return None
                w[mine] = ws[vertex_of[mine]]
            o = w_truth - a * w
            c = circular_fraction(o)
            offset = np.round(o - c).astype(np.int64)
            _, right = modal(offset, frame)
            share = float(right.mean()) if len(right) else 0.0
            if best is None or share > best["share"]:
                best = {"share": share, "chirality": a, "cut": float(cut), "fraction": c}
    return best


def quantiles(values, q=(50, 90)):
    return [round(float(v), 2) for v in np.percentile(values, q)] if len(values) else None


def guard_holdout():
    dirty = subprocess.run(
        ["git", "-C", str(CODE), "status", "--porcelain", "--", *results.LANE],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "holdout scoring needs a clean commit of the ml module and the bench (git status shows changes)"
        )


def score(
    control, paths, zones=("dev",), mask="known", out_dir=None, label="surfaces", ignore_winding=False, z_range=None
):
    truth = packages.load(control)
    z_lo, z_hi = z_range or (-np.inf, np.inf)
    record = truth["record"]
    voxel_um = float(truth["voxel_um"])
    knots, sense = truth["knots"], int(truth["sense"])
    zone_ids = [packages.ZONES.index(z) for z in zones]
    scored_truth = np.isin(truth["zone"], zone_ids)
    scored_truth &= (truth["points"][:, 2] >= z_lo) & (truth["points"][:, 2] < z_hi)
    flags = truth["flags"]
    surfaces = [load(p, mask) for p in paths]
    surfaces = [s for s in surfaces if s["mask"].sum() >= 4]
    if ignore_winding:
        for s in surfaces:
            s["w"] = None
            s["meta"] = {k: v for k, v in s["meta"].items() if k not in ("winding", "cut_deg", "chirality")}
    if not surfaces:
        raise SystemExit("no surface vertices")
    for s in surfaces:
        xyz = s["xyz"][s["mask"]].astype(np.float64)
        theta, _ = truths.angle(knots, xyz)
        grid = np.zeros(s["mask"].shape)
        grid[s["mask"]] = theta
        s["turn"], s["part"] = unwrapped(grid, s["mask"])
        s["normals"] = grid_normals(s["xyz"], s["mask"])
        s["area"] = vertex_area(s["xyz"].astype(np.float64), s["mask"]) * (voxel_um * 1e-4) ** 2
        s["zone"] = packages.zone_codes(record["scan"], xyz[:, 2])
        steps = np.linalg.norm(np.diff(s["xyz"], axis=1), axis=-1)[s["mask"][:, 1:] & s["mask"][:, :-1]]
        s["step"] = float(np.median(steps)) if len(steps) else 1.0
    vertices = np.concatenate([s["xyz"][s["mask"]] for s in surfaces]).astype(np.float32)
    surface_of = np.concatenate([np.full(int(s["mask"].sum()), i) for i, s in enumerate(surfaces)])
    vertex_of = np.concatenate([np.arange(int(s["mask"].sum())) for s in surfaces])
    in_zone = np.isin(np.concatenate([s["zone"] for s in surfaces]), zone_ids)
    in_zone &= (vertices[:, 2] >= z_lo) & (vertices[:, 2] < z_hi)
    spacing = float(truth["spacing_um"]) / voxel_um
    tolerance = TOLERANCE_UM / voxel_um

    to_truth, depth_to_truth = on_sheet(
        truth["points"],
        vertices,
        truth["normals"],
        FAR_UM / voxel_um,
        LATERAL_SHARE * spacing,
        keep=KEEP,
        normal_of="source",
    )
    t_of = to_truth[:, 0]
    on_truth = (t_of >= 0) & (depth_to_truth[:, 0] <= tolerance)
    t_safe = np.maximum(to_truth, 0)
    judgeable = (
        (to_truth >= 0)
        & (depth_to_truth <= tolerance)
        & scored_truth[t_safe]
        & ((flags[t_safe] & SKIP) == 0)
        & np.isfinite(truth["winding"][t_safe])
        & (truth["frame"][t_safe] >= 0)
        & in_zone[:, None]
    )
    within = (to_truth >= 0) & (depth_to_truth <= tolerance)
    judgeable &= ~(within & ((flags[t_safe] & SKIP) > 0)).any(axis=1)[:, None]
    judged = judgeable.any(axis=1)
    w_cand = truth["winding"][t_safe]
    spread = np.max(np.where(judgeable, w_cand, -np.inf), axis=1) - np.min(np.where(judgeable, w_cand, np.inf), axis=1)
    ambiguous = judged & (spread >= 0.5)

    step = float(np.median([s["step"] for s in surfaces]))
    to_surface, depth_to_surface = on_sheet(
        vertices, truth["points"], truth["normals"], FAR_UM / voxel_um, LATERAL_SHARE * step, keep=KEEP
    )
    p_of = to_surface[:, 0]
    covered = (p_of >= 0) & (depth_to_surface[:, 0] <= tolerance)

    has_winding = all(s["w"] is not None or "winding" in s["meta"] for s in surfaces)
    first = judgeable[:, 0]
    pairs = (surface_of[first], vertex_of[first], t_of[first])
    alignment = fit_alignment(surfaces, pairs, truth, sense, surfaces[0]["meta"]) if has_winding else None
    mode = "global" if alignment else "local"
    windings = []
    for s in surfaces:
        if alignment:
            windings.append(surface_winding(s, s["turn"], sense, alignment["chirality"], alignment["cut"]))
        else:
            windings.append(sense * s["turn"] / (2 * np.pi))
    w_surface = np.concatenate(windings)
    part_of = np.concatenate([s["part"] for s in surfaces]).astype(np.int64)
    chirality = alignment["chirality"] if alignment else 1

    def offsets(p_idx, w_truth, frame):
        """Raw integer offsets of (surface vertex, truth winding) pairs and their keys: the truth frame, or in local
        mode frame x surface part."""
        o = w_truth - chirality * w_surface[p_idx]
        frame = frame.astype(np.int64)
        if alignment:
            return np.round(o - alignment["fraction"]).astype(np.int64), frame
        c = circular_fraction(o) if len(o) else 0.0
        key = (frame * len(surfaces) + surface_of[p_idx]) * 100_000 + part_of[p_idx]
        return np.round(o - c).astype(np.int64), key

    raw, key = offsets(np.flatnonzero(first), truth["winding"][t_of[first]], truth["frame"][t_of[first]])
    modes, _ = modal(raw, key)

    def ranks(p_idx, w_truth, frame, valid):
        """Rank error per row from KEEP candidate pairs: the candidate nearest its frame's mode wins, so where truth
        or surfaces touch within the tolerance any of them may be the one; UNSCORED where no candidate is valid."""
        flat = np.flatnonzero(valid)
        raw_, key_ = offsets(p_idx[valid], w_truth[valid], frame[valid])
        known = np.array([int(k) in modes for k in key_], dtype=bool)
        mode_ = np.array([modes.get(int(k), 0) for k in key_], dtype=np.int64)
        rank = np.full(valid.shape, 10_000, dtype=np.int64)
        rank.flat[flat[known]] = raw_[known] - mode_[known]
        best = np.argmin(np.abs(rank), axis=1)
        chosen = np.take_along_axis(rank, best[:, None], axis=1)[:, 0]
        strict = rank[:, 0]
        return (
            np.where(np.abs(chosen) < 10_000, np.clip(chosen, -100, 100), UNSCORED),
            np.where(np.abs(strict) < 10_000, np.clip(strict, -100, 100), UNSCORED),
        )

    rows = np.repeat(np.arange(len(vertices))[:, None], KEEP, axis=1)
    rank_vertex, _ = ranks(rows, truth["winding"][t_safe], truth["frame"][t_safe], judgeable)

    usable = scored_truth & ((flags & SKIP) == 0) & np.isfinite(truth["winding"]) & (truth["frame"] >= 0)
    valid_t = (to_surface >= 0) & (depth_to_surface <= tolerance) & usable[:, None]
    p_safe = np.maximum(to_surface, 0)
    w_t = np.repeat(truth["winding"][:, None], KEEP, axis=1)
    f_t = np.repeat(truth["frame"][:, None], KEEP, axis=1)
    rank_truth, rank_truth_strict = ranks(p_safe, w_t, f_t, valid_t)
    rank_truth, rank_truth_strict = rank_truth.astype(np.int8), rank_truth_strict.astype(np.int8)

    cross = np.load(packages.PACKAGES / control / "crossings.npz")
    scored_cross = np.isin(cross["zone"], zone_ids) & ((cross["flags"] & SKIP) == 0)
    scored_cross &= (cross["z"] >= z_lo) & (cross["z"] < z_hi)
    cxyz = np.column_stack([cross["xy"], cross["z"]]).astype(np.float32)
    rank_cross = np.full(len(cxyz), UNSCORED, dtype=np.int8)
    ci = np.flatnonzero(scored_cross)
    if len(ci):
        c_to, c_depth = on_sheet(vertices, cxyz[ci], cross["normal"][ci], tolerance, LATERAL_SHARE * step, keep=KEEP)
        mesh_frame = np.full(int(truth["mesh"].max()) + 1, -1, dtype=np.int64)
        mesh_frame[truth["mesh"]] = truth["frame"]
        w_c = np.repeat(cross["winding"][ci][:, None], KEEP, axis=1)
        f_c = np.repeat(mesh_frame[cross["mesh"][ci]][:, None], KEEP, axis=1)
        chosen, _ = ranks(np.maximum(c_to, 0), w_c, f_c, (c_to >= 0) & np.isfinite(c_depth))
        rank_cross[ci] = chosen.astype(np.int8)

    area = np.concatenate([s["area"][s["mask"]] for s in surfaces])
    correct = judged & (rank_vertex == 0)
    wrong = judged & (rank_vertex != 0)
    near = (t_of >= 0) & ~on_truth
    largest, edges_judged, edges_wrong = 0.0, 0, 0
    steps_total, steps_off, flipped, folded, normal_edges = 0, 0, 0, 0, 0
    start = 0
    per_surface = []
    for s in surfaces:
        n = int(s["mask"].sum())
        sl = slice(start, start + n)
        start += n
        grid = np.full(s["mask"].shape, UNSCORED, dtype=np.int16)
        grid[s["mask"]] = rank_vertex[sl]
        good = np.zeros(s["mask"].shape, dtype=bool)
        good[s["mask"]] = correct[sl]
        parts, count = ndimage.label(good, structure=np.ones((3, 3)))
        if count:
            sizes = ndimage.sum(s["area"], parts, index=np.arange(1, count + 1))
            largest = max(largest, float(sizes.max()))
        zone_grid = np.zeros(s["mask"].shape, dtype=bool)
        zone_grid[s["mask"]] = in_zone[sl]
        for axis in (0, 1):
            a_ = np.take(grid, range(grid.shape[axis] - 1), axis=axis)
            b_ = np.take(grid, range(1, grid.shape[axis]), axis=axis)
            both = (a_ != UNSCORED) & (b_ != UNSCORED)
            edges_judged += int(both.sum())
            edges_wrong += int((both & (a_ != b_)).sum())
            xa = np.take(s["xyz"], range(s["xyz"].shape[axis] - 1), axis=axis)
            xb = np.take(s["xyz"], range(1, s["xyz"].shape[axis]), axis=axis)
            ma = np.take(s["mask"], range(s["mask"].shape[axis] - 1), axis=axis)
            mb = np.take(s["mask"], range(1, s["mask"].shape[axis]), axis=axis)
            za = np.take(zone_grid, range(zone_grid.shape[axis] - 1), axis=axis)
            pair = ma & mb & za
            length = np.linalg.norm(xb - xa, axis=-1)[pair]
            if len(length):
                median = np.median(length)
                steps_total += len(length)
                steps_off += int((np.abs(length - median) > ISOMETRY * median).sum())
            na = np.take(s["normals"], range(s["normals"].shape[axis] - 1), axis=axis)
            nb = np.take(s["normals"], range(1, s["normals"].shape[axis]), axis=axis)
            dots = np.einsum("...k,...k->...", na, nb)[pair]
            dots = dots[np.isfinite(dots)]
            normal_edges += len(dots)
            folded += int((dots < 0).sum())
        normals = s["normals"][s["mask"]]
        side = np.sign(toward_axis(knots, s["xyz"][s["mask"]].astype(np.float64), normals))
        side = side[in_zone[sl] & np.isfinite(side)]
        if len(side):
            majority = 1 if (side > 0).sum() >= (side < 0).sum() else -1
            flipped += int((side == -majority).sum())
        per_surface.append(
            {
                "surface": s["name"],
                "winding": s["meta"].get("winding"),
                "area_cm2": round(float(area[sl][in_zone[sl]].sum()), 3),
                "correct_cm2": round(float(area[sl][correct[sl]].sum()), 3),
                "wrong_cm2": round(float(area[sl][wrong[sl]].sum()), 3),
            }
        )
    zone_vertices = int(in_zone.sum())
    truth_area = truth["area_mm2"] / 100.0
    unique = (flags & packages.DUPLICATE) == 0
    in_scope = scored_truth & unique
    pressed = (flags & packages.PRESSED) > 0
    touching = (flags & packages.TOUCHING) > 0
    tj = rank_truth != UNSCORED
    cj = rank_cross != UNSCORED
    cross_pressed = (cross["flags"] & packages.PRESSED) > 0
    cross_overlap = (cross["flags"] & (packages.OVERLAP_NEXT | packages.OVERLAP_PREVIOUS)) > 0
    near_truth = (p_of >= 0) & in_scope
    out = {
        "control": control,
        "zones": list(zones),
        "label": label,
        "mode": mode,
        "alignment": None
        if not alignment
        else {
            "chirality": alignment["chirality"],
            "cut_deg": round(float(np.rad2deg(alignment["cut"])), 1),
            "fraction": round(alignment["fraction"], 4),
            "fit_share": round(alignment["share"], 4),
        },
        "surfaces": len(surfaces),
        "surface_cm2": round(float(area[in_zone].sum()), 3),
        "truth_cm2": round(float(truth_area[in_scope].sum()), 3),
        "distance_vox": {
            "truth_to_surface_covered_p50_p90": quantiles(depth_to_surface[covered & in_scope, 0]),
            "truth_to_surface_within_300um_p50_p90": quantiles(depth_to_surface[near_truth, 0]),
            "surface_to_truth_p50_p90": quantiles(depth_to_truth[on_truth & in_zone, 0]),
            "surface_near_truth_p50_p90": quantiles(depth_to_truth[(t_of >= 0) & in_zone, 0]),
        },
        "winding": {
            "judged_truth": int(tj.sum()),
            "right_share": round(float((rank_truth[tj] == 0).mean()), 4) if tj.any() else None,
            "right_share_pressed": round(float((rank_truth[tj & pressed] == 0).mean()), 4)
            if (tj & pressed).any()
            else None,
            "right_share_loose": round(float((rank_truth[tj & ~pressed] == 0).mean()), 4)
            if (tj & ~pressed).any()
            else None,
            "right_share_untouched_strict": round(float((rank_truth_strict[tj & ~touching] == 0).mean()), 4)
            if (tj & ~touching).any()
            else None,
            "right_share_touching": round(float((rank_truth[tj & touching] == 0).mean()), 4)
            if (tj & touching).any()
            else None,
            "touching_share": round(float((tj & touching).sum() / max(tj.sum(), 1)), 4),
            "rank_error_share_crossings": round(float((rank_cross[cj] != 0).mean()), 4) if cj.any() else None,
            "rank_error_share_crossings_pressed": round(float((rank_cross[cj & cross_pressed] != 0).mean()), 4)
            if (cj & cross_pressed).any()
            else None,
            "rank_error_share_crossings_loose": round(float((rank_cross[cj & ~cross_pressed] != 0).mean()), 4)
            if (cj & ~cross_pressed).any()
            else None,
            "rank_error_share_crossings_overlap": round(float((rank_cross[cj & cross_overlap] != 0).mean()), 4)
            if (cj & cross_overlap).any()
            else None,
            "judged_crossings": int(cj.sum()),
            "rank_error_histogram": {
                str(k): int(v) for k, v in zip(*np.unique(rank_truth[tj], return_counts=True), strict=True)
            },
        },
        "area_cm2": {
            "correct": round(float(area[correct].sum()), 3),
            "wrong": round(float(area[wrong].sum()), 3),
            "on_truth_unjudged": round(float(area[on_truth & in_zone & ~judged].sum()), 3),
            "near_off_truth": round(float(area[near & in_zone].sum()), 3),
            "far_from_truth": round(float(area[(t_of < 0) & in_zone].sum()), 3),
            "largest_connected_correct": round(largest, 3),
            "truth_covered": round(float(truth_area[covered & in_scope].sum()), 3),
            "truth_covered_correct": round(float(truth_area[(rank_truth == 0) & in_scope].sum()), 3),
        },
        "truth_recall": round(float(truth_area[covered & in_scope].sum() / max(truth_area[in_scope].sum(), 1e-12)), 4),
        "truth_recall_correct": round(
            float(truth_area[(rank_truth == 0) & in_scope].sum() / max(truth_area[in_scope].sum(), 1e-12)), 4
        ),
        "truth_recall_pressed": round(
            float(truth_area[covered & in_scope & pressed].sum() / max(truth_area[in_scope & pressed].sum(), 1e-12)), 4
        ),
        "truth_recall_loose": round(
            float(truth_area[covered & in_scope & ~pressed].sum() / max(truth_area[in_scope & ~pressed].sum(), 1e-12)),
            4,
        ),
        "wrong_join_share": round(edges_wrong / edges_judged, 5) if edges_judged else None,
        "surface_on_touching_truth_share": round(float(ambiguous[in_zone].sum() / max(judged[in_zone].sum(), 1)), 4),
        "non_isometric_share": round(steps_off / steps_total, 4) if steps_total else None,
        "flipped_share": round(flipped / zone_vertices, 5) if zone_vertices else None,
        "folded_edge_share": round(folded / normal_edges, 5) if normal_edges else None,
        "per_surface": per_surface,
    }
    folder = Path(out_dir) if out_dir else SCORES / control / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "truth_rank.npy", rank_truth)
    np.save(folder / "truth_depth.npy", np.where(p_of >= 0, depth_to_surface[:, 0], np.inf).astype(np.float32))
    np.save(folder / "crossing_rank.npy", rank_cross)
    np.savez_compressed(
        folder / "surface_labels.npz",
        surface=surface_of.astype(np.int32),
        vertex=vertex_of.astype(np.int32),
        truth_vertex=t_of,
        depth_vox=depth_to_truth[:, 0],
        rank=rank_vertex,
        names=np.array([s["name"] for s in surfaces]),
    )
    out["folder"] = str(folder)
    (folder / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def headline(out):
    flat = {
        "truth_recall": out["truth_recall"],
        "truth_recall_correct": out["truth_recall_correct"],
        "winding_right_share": out["winding"]["right_share"],
        "winding_right_pressed": out["winding"]["right_share_pressed"],
        "winding_right_loose": out["winding"]["right_share_loose"],
        "rank_error_crossings": out["winding"]["rank_error_share_crossings"],
        "correct_cm2": out["area_cm2"]["correct"],
        "wrong_cm2": out["area_cm2"]["wrong"],
        "largest_correct_cm2": out["area_cm2"]["largest_connected_correct"],
        "wrong_join_share": out["wrong_join_share"],
        "non_isometric_share": out["non_isometric_share"],
        "flipped_share": out["flipped_share"],
        "folded_edge_share": out["folded_edge_share"],
    }
    d = out["distance_vox"]
    if d["truth_to_surface_covered_p50_p90"]:
        flat["truth_to_surface_p50"], flat["truth_to_surface_p90"] = d["truth_to_surface_covered_p50_p90"]
    if d["surface_to_truth_p50_p90"]:
        flat["surface_to_truth_p50"], flat["surface_to_truth_p90"] = d["surface_to_truth_p50_p90"]
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", help="a package name from calibration/bench/controls.json, e.g. PHerc0139-dev")
    parser.add_argument("surfaces", nargs="+", help="tifxyz folders or folders holding them")
    parser.add_argument("--holdout", action="store_true", help="score the holdout zones (clean commit; logged)")
    parser.add_argument("--zones", help="comma-separated zones to score instead (dev, holdout-A, holdout-B)")
    parser.add_argument("--mask", default="known", choices=("known", "locked"))
    parser.add_argument("--label", default="surfaces", help="short name of what is scored (folder and record)")
    parser.add_argument("--system", default="network")
    parser.add_argument("--out", help="score folder (default: bench/scores/CONTROL/TIME-LABEL under the data root)")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--ignore-winding", action="store_true", help="local mode: each surface its own offset")
    parser.add_argument("--z", help="z0:z1, score only truth and surface in these slices (a short run's band)")
    args = parser.parse_args()
    if args.zones:
        zones = tuple(args.zones.split(","))
    elif args.holdout:
        zones = ("holdout-A", "holdout-B")
    else:
        zones = ("dev",)
    holdout = any(z.startswith("holdout") for z in zones)
    if holdout and not args.holdout:
        raise SystemExit("scoring holdout zones needs --holdout")
    if holdout:
        guard_holdout()
    paths = [p for given in args.surfaces for p in surfaces_in(given)]
    paths += [p for given in args.surfaces for p in sorted(Path(given).glob("*.tifxyz")) if (p / "x.tif").exists()]
    paths = list(dict.fromkeys(paths))
    z_range = tuple(int(v) for v in args.z.split(":")) if args.z else None
    out = score(args.control, paths, zones, args.mask, args.out, args.label, args.ignore_winding, z_range)
    if holdout:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "control": args.control,
            "zones": list(zones),
            "label": args.label,
            "commit": results.commit_id(),
            "surfaces": [str(p) for p in args.surfaces],
            "folder": out["folder"],
            "headline": headline(out),
        }
        HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(HOLDOUT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    printed = {k: v for k, v in out.items() if k != "per_surface"}
    print(json.dumps(printed, indent=1))
    if not args.no_record:
        run = results.record(
            args.control,
            args.system,
            args.label,
            "score",
            headline(out),
            {"surfaces": [str(p) for p in args.surfaces], "zones": list(zones), "mask": args.mask},
            detail=printed,
            split=",".join(zones),
        )
        print(f"recorded run {run}")


if __name__ == "__main__":
    main()
