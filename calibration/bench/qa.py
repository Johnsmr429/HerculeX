"""Training-truth QA of bench packages: numbers per package and per mesh, the truth against the CT per mesh, level-0 CT
planes with the crossings drawn, and a per-mesh inclusion table.

./run.sh calibration.bench.qa vertices NAME...                BENCH/qa/NAME/vertices.npz: every truth vertex in zero CT or in air
./run.sh calibration.bench.qa numbers NAME...                 BENCH/qa/NAME/numbers.json
./run.sh calibration.bench.qa planes NAME [--z Z ...]         BENCH/qa/NAME/z{Z}.png, z{Z}-overview.png, z{Z}-tile{i}.png
./run.sh calibration.bench.qa table NAME... [--notes FILE.md]  BENCH/qa/training-truth.json and training-truth.md

Run bench.ridge NAME --chunks 360 first; vertices and numbers read its sample, numbers reads vertices.npz.

Numbers. Areas count non-duplicate vertices (bench.package). Flag shares are over a mesh's (or the package's)
non-duplicate vertices, the duplicate share over all of them.
- Gaps: on every crossing plane and one-degree sector, the windings of one frame between its innermost and outermost
  traced sheet that no mesh covers.
- Same-surface pairs: two meshes of one frame within 16 µm along the normal whose windings differ there; one of the
  two is numbered wrong on that surface.
- Per mesh, from bench.ridge's sample:
  - the distance to the nearest CT ridge (Gaussian sigma 3 voxels) along the normal, and its null (the same search
    from 6 voxels off the mesh);
  - the contrast of the mean CT profile, (max - min) / mean over -12..+12 voxels, in each 128^3 chunk, median over
    the mesh's chunks: 0.25-0.6 on a sheet, under 0.1 off it. Pooled over the whole mesh it also drops where the
    mesh's offset to its sheet drifts;
  - the share of vertices in zero CT (outside the published volume's mask), and the share with no papyrus within 3
    voxels (in air). bench.qa vertices measures both on every vertex.
- The flattest (mesh, 128^3 chunk) cells of that sample: where a mesh leaves its sheet.

Planes. The level-0 CT at one crossing plane over the box of the truth (64 voxels around it), chunks never fetched
left black, with each crossing as a 3 x 3 dot coloured by its winding: red, green, blue for windings 0, 1, 2 mod 3
of the frame holding most of the truth (outward the colours run red, green, blue, red ...), yellow, magenta, cyan
in other frames (unnamed meshes, other runs of names), white where the truth is flagged conflict or order. Tiles are
640 x 640 crops at 1:1 around the sheets where flagged crossings are densest, and one at each of four angles.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from calibration.bench import package as packages
from calibration.bench.ridge import CHECKS, NULL_VOX, ridges
from scans.chunks.volume import Volume
from storage.data_roots import SURFACE_BENCH as BENCH
from storage.data_roots import WINDING_LABELS

QA = BENCH / "qa"
SECTORS = 360
TILE = 640
FIRST = [(235, 50, 50), (40, 210, 40), (60, 120, 255)]
OTHER = [(250, 220, 30), (230, 60, 230), (30, 225, 225)]
FLAGGED = (255, 255, 255)
RIDGE_SAMPLED = 200
RULES = {
    "exclude": [
        "CT profile contrast under 0.08, or under 0.15 and half the package's median (not on papyrus)",
        "more than half of its sampled vertices lie in zero CT (outside the published volume's mask)",
        "more than 35% of its vertices have no papyrus within 3 voxels of the package's papyrus (PHerc0139: 2-17%)",
        "median distance to the CT ridge more than 3 voxels above the package's median",
        "conflict or order on more than 25% of its non-duplicate vertices",
        "more than 90% of its vertices repeat a larger mesh (a copy; the larger one carries the labels)",
    ],
    "lower weight (0.5)": [
        "CT profile contrast under 0.15, or under 0.2 and three quarters of the package's median (on a sheet 0.25-0.6)",
        "10-50% of its sampled vertices lie in zero CT",
        "20-35% of its vertices have no papyrus within 3 voxels",
        "median distance to the CT ridge 1.5 to 3 voxels above the package's median",
        "conflict or order on 5-25% of its non-duplicate vertices",
        "more than half of its vertices repeat a larger mesh",
        "30% or more of its vertices lie on another mesh's surface numbered another winding (one of the two is a copy "
        "under the wrong number, and the winding next to them is untraced there)",
    ],
    "include": ["everything else; ridge figures need at least 200 sampled vertices to count"],
}


def share(mask, within):
    return round(float((mask & within).sum() / within.sum()), 4) if within.any() else None


def sheet_index(crossings, knots, sense):
    """Integer winding of each crossing's sheet at its own angle: winding - sense * angle / 2pi, rounded; and angle."""
    z, xy = crossings["z"], crossings["xy"].astype(np.float64)
    cx = np.interp(z, knots[:, 0], knots[:, 1])
    cy = np.interp(z, knots[:, 0], knots[:, 2])
    theta = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    return np.round(crossings["winding"] - sense * theta / (2 * np.pi)), theta


def frames_of(truth):
    """Frame per mesh (the most common over its vertices), -1 where it has none."""
    out = np.full(len(truth["names"]), -1, dtype=np.int64)
    for m in range(len(out)):
        f = truth["frame"][truth["mesh"] == m]
        f = f[f >= 0]
        if len(f):
            out[m] = np.bincount(f).argmax()
    return out


def covered(crossings, knots, sense):
    """(plane, sector, sheet, mesh) for every one-degree sector a mesh covers on a plane: the sectors between two
    consecutive crossings of one mesh (neighbouring grid columns), each with the sheet index at its own angle."""
    z, w, mesh, col = crossings["z"], crossings["winding"], crossings["mesh"], crossings["col"]
    _, theta = sheet_index(crossings, knots, sense)
    pair = np.flatnonzero(
        (z[1:] == z[:-1]) & (mesh[1:] == mesh[:-1]) & (col[1:] == col[:-1] + 1) & np.isfinite(w[1:] + w[:-1])
    )
    a = (theta[pair] + np.pi) / (2 * np.pi) * SECTORS
    b = a + np.angle(np.exp(1j * (theta[pair + 1] - theta[pair]))) / (2 * np.pi) * SECTORS
    low, high = np.floor(np.minimum(a, b)).astype(np.int64), np.floor(np.maximum(a, b)).astype(np.int64)
    count = high - low + 1
    row = np.repeat(np.arange(len(pair)), count)
    j = low[row] + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    span = b[row] - a[row]
    t = np.clip(np.where(np.abs(span) > 1e-9, (j + 0.5 - a[row]) / np.where(np.abs(span) > 1e-9, span, 1), 0), 0, 1)
    wj = w[pair][row] + t * (w[pair + 1][row] - w[pair][row])
    sheet = np.round(wj - sense * ((j + 0.5) / SECTORS * 2 * np.pi - np.pi) / (2 * np.pi)).astype(np.int64)
    return np.round(z[pair][row]).astype(np.int64), np.mod(j, SECTORS), sheet, mesh[pair][row]


def gaps(crossings, knots, sense, frame_of_mesh, top=6):
    """Per frame: (plane, sector) cells, windings spanned between the innermost and outermost sheet covered, missing
    ones, and the windings missing most often (cells, z range)."""
    z, sector, sheet, mesh = covered(crossings, knots, sense)
    frame = frame_of_mesh[mesh]
    out = {}
    for f in np.unique(frame[frame >= 0]):
        s = frame == f
        cells, cell = np.unique(z[s] * SECTORS + sector[s], return_inverse=True)
        nn = sheet[s]
        low = nn.min()
        present = np.zeros((len(cells), nn.max() - low + 1), dtype=bool)
        present[cell.ravel(), nn - low] = True
        index = np.arange(present.shape[1])
        first = present.argmax(axis=1)
        last = present.shape[1] - 1 - present[:, ::-1].argmax(axis=1)
        missing = (index >= first[:, None]) & (index <= last[:, None]) & ~present
        per_n = missing.sum(axis=0)
        cell_z = cells // SECTORS
        worst = []
        for j in np.argsort(-per_n)[:top]:
            if per_n[j]:
                zs = cell_z[missing[:, j]]
                worst.append({"winding": int(low + j), "cells": int(per_n[j]), "z": [int(zs.min()), int(zs.max())]})
        out[int(f)] = {
            "cells": len(cells),
            "spanned": int((last - first + 1).sum()),
            "missing": int(missing.sum()),
            "most_missing": worst,
        }
    return out


def same_surface(truth, sample=300_000, least=100, seed=0):
    """Pairs of meshes of one frame tracing one surface (vertices within 16 µm along the normal, lateral within 0.75
    grid spacing) with windings that differ there: one of the two is numbered wrong on that surface. Per pair: the
    contacts, the share whose windings differ, and the most common difference."""
    points, normals = truth["points"].astype(np.float64), truth["normals"].astype(np.float64)
    mesh, winding, frame = truth["mesh"], truth["winding"], truth["frame"]
    voxel_um, spacing_um = float(truth["voxel_um"]), float(truth["spacing_um"])
    usable = np.flatnonzero(np.isfinite(winding) & np.all(np.isfinite(normals), axis=1) & (frame >= 0))
    pick = np.random.default_rng(seed).choice(usable, min(sample, len(usable)), replace=False)
    lateral = packages.LATERAL_SHARE * spacing_um / voxel_um
    radius = float(np.hypot(packages.SAME_SURFACE_UM / voxel_um, lateral))
    keys = []
    for rows, other, along, across, _ in packages.neighbours(points, normals, pick, radius, 24):
        o = np.maximum(other, 0)
        hit = (other >= 0) & (mesh[o] != mesh[rows][:, None]) & (frame[o] == frame[rows][:, None])
        hit &= np.isfinite(winding[o]) & (np.abs(along) <= packages.SAME_SURFACE_UM / voxel_um) & (across <= lateral)
        a, j = np.nonzero(hit)
        d = np.round(winding[o[a, j]] - winding[rows[a]]).astype(np.int64)
        lo, hi = np.minimum(mesh[rows[a]], mesh[o[a, j]]), np.maximum(mesh[rows[a]], mesh[o[a, j]])
        keys.append(np.stack([lo, hi, np.where(mesh[rows[a]] == lo, d, -d)], axis=1))
    if not keys:
        return []
    found, counts = np.unique(np.concatenate(keys), axis=0, return_counts=True)
    out = []
    for a, b in np.unique(found[:, :2], axis=0):
        s = (found[:, 0] == a) & (found[:, 1] == b)
        total = int(counts[s].sum())
        differ = int(counts[s & (found[:, 2] != 0)].sum())
        if total >= least:
            d, c = found[s][np.argmax(counts[s]), 2], counts[s].max()
            out.append(
                {
                    "mesh_index": [int(a), int(b)],
                    "meshes": [str(truth["names"][a]), str(truth["names"][b])],
                    "wraps": [int(truth["mesh_wrap"][a]), int(truth["mesh_wrap"][b])],
                    "contacts": total,
                    "differ_share": round(differ / total, 3),
                    "mode_difference": int(d),
                    "mode_share": round(float(c / total), 3),
                }
            )
    return sorted(out, key=lambda r: -r["differ_share"] * r["contacts"])


def coincident(truth, pairs, least_share=0.5, least_contacts=300):
    """For same-surface pairs numbered differently: the share of each mesh's vertices that lie on the other's surface
    (within 16 µm along the normal, lateral within 0.75 grid spacing) with a different winding."""
    points, normals = truth["points"].astype(np.float64), truth["normals"].astype(np.float64)
    mesh, winding = truth["mesh"], truth["winding"]
    voxel_um, spacing_um = float(truth["voxel_um"]), float(truth["spacing_um"])
    radius = packages.SAME_SURFACE_UM / voxel_um + packages.LATERAL_SHARE * spacing_um / voxel_um
    out = []
    for p in pairs:
        if p["differ_share"] < least_share or p["contacts"] < least_contacts:
            continue
        shares = []
        for a, b in (p["mesh_index"], p["mesh_index"][::-1]):
            sa, sb = np.flatnonzero(mesh == a), np.flatnonzero(mesh == b)
            distance, j = cKDTree(points[sb]).query(points[sa], k=1, distance_upper_bound=radius, workers=-1)
            hit = np.isfinite(distance)
            other = sb[np.where(hit, j, 0)]
            along = np.einsum("ij,ij->i", points[other] - points[sa], normals[sa])
            on = hit & (np.abs(along) <= packages.SAME_SURFACE_UM / voxel_um)
            on &= np.round(winding[other] - winding[sa]) != 0
            shares.append(round(float(on.mean()), 3))
        out.append(p | {"coincident_share": shares})
    return out


def contrast(mean):
    """(max - min) / mean of a mean CT profile; None where the CT is empty (outside the scroll's mask)."""
    level = float(mean.mean())
    return round(float((mean.max() - mean.min()) / level), 3) if level > 1 else None


def papyrus_threshold(profiles, empty):
    """Halfway from the package's air (5th percentile of the smoothed profiles) to its papyrus (their median
    maximum), from bench.ridge's sample; None without CT."""
    smooth = ndimage.gaussian_filter1d(profiles, 2.0, axis=1)[~empty]
    if not len(smooth):
        return None
    air = np.percentile(smooth[smooth > 0], 5)
    return float(air + 0.5 * (np.median(smooth.max(axis=1)) - air))


def papyrus_offset(profiles, offsets, empty):
    """Where the package's papyrus is brightest along the normal (the mean profile's peak): meshes on a sheet's face
    sit a few voxels from it, the same way across a package."""
    known = profiles[~empty]
    return float(offsets[int(np.argmax(known.mean(axis=0)))]) if len(known) else 0.0


def in_air(profiles, offsets, empty):
    """Vertices with no papyrus within 3 voxels of where the package's papyrus is brightest along the normal: the
    smoothed CT there stays under papyrus_threshold. About 0.09 of PHerc0139's truth."""
    threshold = papyrus_threshold(profiles, empty)
    if threshold is None:
        return np.zeros(len(profiles), dtype=bool)
    centre = papyrus_offset(profiles, offsets, empty)
    smooth = ndimage.gaussian_filter1d(profiles, 2.0, axis=1)
    return (smooth[:, np.abs(offsets - centre) <= 3].max(axis=1) < threshold) & ~empty


def vertex_pass(name, block=(2, 4, 4)):
    """Every truth vertex against the CT: QA/NAME/vertices.npz with qa (uint8 per vertex: 1 zero CT within 3 voxels
    along the normal, outside the volume's mask; 2 no papyrus within 3 voxels of where the package's papyrus is
    brightest, under papyrus_threshold of bench.ridge's sample; 0 on papyrus; 254 CT never fetched there; 255 no
    normal) and peak (that smoothed CT maximum). Reads the store only."""
    truth = packages.load(name)
    entry = truth["record"]["scan"]
    sample = np.load(CHECKS / f"{name}-ridge.npz")
    profiles, offsets = sample["profiles"].astype(np.float32), sample["offsets_vox"]
    sampled_empty = np.all(profiles[:, np.abs(offsets) <= 3] == 0, axis=1)
    threshold = papyrus_threshold(profiles, sampled_empty)
    centre = papyrus_offset(profiles, offsets, sampled_empty)
    points = truth["points"][:, ::-1].astype(np.float64)
    normals = truth["normals"][:, ::-1].astype(np.float64)
    ok = np.all(np.isfinite(normals), axis=1)
    steps = np.arange(-9, 9.01, 0.5)
    inner = np.abs(steps - centre) <= 3
    near = np.abs(steps) <= 3
    peak = np.zeros(len(points), dtype=np.float32)
    empty = np.zeros(len(points), dtype=bool)
    unread = np.zeros(len(points), dtype=bool)
    with Volume.for_scan(entry["sample"], entry["scan"], "ct", 0) as volume:
        size = np.asarray(volume.chunks) * np.asarray(block)
        key = np.floor(points / size).astype(np.int64)
        cells, inverse = np.unique(key[ok], axis=0, return_inverse=True)
        members = np.split(
            np.flatnonzero(ok)[np.argsort(inverse.ravel(), kind="stable")], np.cumsum(np.bincount(inverse.ravel()))[:-1]
        )
        for cell, idx in zip(cells, members, strict=True):
            low, high = cell * size - 8, (cell + 1) * size + 8
            ct, unknown = volume.read_with_unknown((low[0], high[0], low[1], high[1], low[2], high[2]))
            path = points[idx][:, None, :] + steps[None, :, None] * normals[idx][:, None, :] - low
            if unknown.any():
                first = np.floor(low / volume.chunks).astype(np.int64)
                at = np.floor((path + low) / volume.chunks).astype(np.int64) - first
                at = np.clip(at, 0, np.array(unknown.shape) - 1)
                unread[idx] = unknown[at[..., 0], at[..., 1], at[..., 2]].any(axis=1)
            profile = ndimage.map_coordinates(ct.astype(np.float32), path.reshape(-1, 3).T, order=1, mode="nearest")
            profile = ndimage.gaussian_filter1d(profile.reshape(len(idx), -1), 2.0, axis=1)
            peak[idx] = profile[:, inner].max(axis=1)
            empty[idx] = np.all(profile[:, near] == 0, axis=1)
    qa = np.where(empty, 1, np.where(peak < threshold, 2, 0)).astype(np.uint8)
    qa[unread] = 254
    qa[~ok] = 255
    folder = QA / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "vertices.npz",
        qa=qa,
        peak=np.clip(peak, 0, 255).astype(np.uint8),
        threshold=threshold,
        papyrus_offset_vox=centre,
    )
    judged = qa < 254
    return {
        "vertices": len(qa),
        "judged": int(judged.sum()),
        "zero_ct": round(float((qa[judged] == 1).mean()), 4) if judged.any() else None,
        "in_air": round(float((qa[judged] == 2).mean()), 4) if judged.any() else None,
    }


def mesh_ridges(name, truth):
    """Per mesh from bench.ridge's sample, over vertices with CT around them: n, |offset| p50 and p90, signed p50,
    null p50, CT profile contrast (the median over the mesh's chunks, and pooled over the mesh), the share in air;
    and the share of its sampled vertices in zero CT (masked out)."""
    path = CHECKS / f"{name}-ridge.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    rows, found = data["truth_index"], data["ridge_sigma3_vox"]
    profiles, offsets = data["profiles"].astype(np.float32), data["offsets_vox"]
    mesh = truth["mesh"][rows]
    points = truth["points"][rows][:, ::-1]
    empty = np.all(profiles[:, np.abs(offsets) <= 3] == 0, axis=1)
    air = in_air(profiles, offsets, empty)
    out = {}
    for m in np.unique(mesh):
        every = mesh == m
        s = every & ~empty
        ok = s & np.isfinite(found)
        if ok.sum() < 20:
            continue
        null = np.concatenate([ridges(profiles[s], offsets, 3.0, side) for side in (NULL_VOX, -NULL_VOX)])
        mean = profiles[s].mean(axis=0)
        cell = np.unique(np.floor(points[s] / 128).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
        cells = [contrast(profiles[s][cell == c].mean(axis=0)) for c in np.flatnonzero(np.bincount(cell) >= 20)]
        cells = [c for c in cells if c is not None]
        out[int(m)] = {
            "sampled": int(s.sum()),
            "ridge_abs_p50_p90_vox": [round(float(v), 2) for v in np.percentile(np.abs(found[ok]), (50, 90))],
            "ridge_signed_p50_vox": round(float(np.median(found[ok])), 2),
            "null_abs_p50_vox": round(float(np.nanmedian(np.abs(null))), 2),
            "profile_contrast": round(float(np.median(cells)), 3) if cells else contrast(mean),
            "pooled_contrast": contrast(mean),
            "profile_peak_vox": round(float(offsets[int(np.argmax(mean))]), 1),
            "ct_empty_share": round(float(empty[every].mean()), 4),
            "air_share": round(float(air[s].mean()), 4),
        }
    return out


def weak_cells(name, truth, least=40, top=12):
    """The (mesh, 128^3 chunk) cells of bench.ridge's sample whose mean CT profile is flattest: where a mesh leaves
    its sheet. Each with its centre (x, y, z), vertices, contrast and median distance to the ridge."""
    path = CHECKS / f"{name}-ridge.npz"
    if not path.exists():
        return []
    data = np.load(path)
    rows, found = data["truth_index"], data["ridge_sigma3_vox"]
    profiles = data["profiles"].astype(np.float32)
    keep = ~np.all(profiles[:, np.abs(data["offsets_vox"]) <= 3] == 0, axis=1)
    rows, found, profiles = rows[keep], found[keep], profiles[keep]
    points = truth["points"][rows]
    key = np.column_stack([truth["mesh"][rows], np.floor(points[:, ::-1] / 128).astype(np.int64)])
    cells, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out = []
    for c in np.flatnonzero(counts >= least):
        s = inverse == c
        mean = profiles[s].mean(axis=0)
        ok = s & np.isfinite(found)
        out.append(
            {
                "mesh": int(cells[c][0]),
                "segment": str(truth["names"][cells[c][0]]),
                "centre_xyz": [int(v) for v in np.median(points[s], axis=0)],
                "vertices": int(s.sum()),
                "contrast": contrast(mean),
                "ridge_abs_p50_vox": round(float(np.median(np.abs(found[ok]))), 2) if ok.any() else None,
            }
        )
    out = [o for o in out if o["contrast"] is not None]
    contrasts = np.array([o["contrast"] for o in out])
    summary = {
        "cells": len(out),
        "contrast_p10_p50": [round(float(v), 3) for v in np.percentile(contrasts, (10, 50))] if len(out) else None,
        "under_0.1": int((contrasts < 0.1).sum()),
    }
    return summary, sorted(out, key=lambda o: o["contrast"])[:top]


def numbers(name):
    truth = packages.load(name)
    record = truth["record"]
    crossings = dict(np.load(packages.PACKAGES / name / "crossings.npz"))
    flags, mesh, area = truth["flags"], truth["mesh"], truth["area_mm2"]
    kept = (flags & packages.DUPLICATE) == 0
    frame_of_mesh = frames_of(truth)
    holes = gaps(crossings, truth["knots"], int(truth["sense"]), frame_of_mesh)
    per_ridge = mesh_ridges(name, truth)
    path = QA / name / "vertices.npz"
    vertex_qa = np.load(path)["qa"] if path.exists() else None
    bits = {
        "pressed": packages.PRESSED,
        "touching": packages.TOUCHING,
        "conflict": packages.CONFLICT,
        "order": packages.ORDER,
        "overlap": packages.OVERLAP_NEXT | packages.OVERLAP_PREVIOUS | packages.OVERLAP_OTHER,
    }
    links = [link for link in record["chain"]["links"] if link.get("votes")]
    purity = [link["purity"] for link in links]
    out = {
        "name": name,
        "scan": record["scan"],
        "z": record["definition"]["z"],
        "package_version": record.get("version"),
        "vertices": len(flags),
        "area_cm2": round(float(area[kept].sum() / 100), 1),
        "meshes": len(truth["names"]),
        "named": int((truth["mesh_wrap"] >= 0).sum()),
        "frames": int(len(np.unique(frame_of_mesh[frame_of_mesh >= 0]))),
        "sense": int(truth["sense"]),
        "sense_votes": record["sense_votes"],
        "duplicate_share": round(float(1 - kept.mean()), 4),
        "shares": {bit: share((flags & value) > 0, kept) for bit, value in bits.items()},
        "chain": {
            "runs": record["chain"]["runs"],
            "links": len(links),
            "unlinked": [[link["from"], link["to"]] for link in record["chain"]["links"] if not link.get("votes")],
            "purity_min": min(purity) if purity else None,
            "purity_median": float(np.median(purity)) if purity else None,
            "weakest": sorted(links, key=lambda link: link["purity"])[:3],
        },
        "checks": record["checks"],
        "zones": record["zones"],
        "gaps": {
            "missing_share": round(
                sum(g["missing"] for g in holes.values()) / max(sum(g["spanned"] for g in holes.values()), 1), 4
            ),
            "by_frame": holes,
        },
        "meshes_detail": [],
    }
    if vertex_qa is not None and (vertex_qa < 254).any():
        judged = vertex_qa < 254
        out["all_vertices"] = {
            "judged": int(judged.sum()),
            "zero_ct": round(float((vertex_qa[judged] == 1).mean()), 4),
            "in_air": round(float((vertex_qa[judged] == 2).mean()), 4),
        }
    pairs = same_surface(truth)
    out["same_surface_pairs"] = coincident(truth, [p for p in pairs if p["differ_share"] > 0])
    cells = weak_cells(name, truth)
    if cells:
        out["mesh_chunk_cells"], out["weakest_cells"] = cells
    ridge_file = CHECKS / f"{name}-ridge.json"
    if ridge_file.exists():
        ridge = json.loads(ridge_file.read_text())["by_group"]["all"]
        sample = np.load(CHECKS / f"{name}-ridge.npz")
        profiles, offsets = sample["profiles"].astype(np.float32), sample["offsets_vox"]
        empty = np.all(profiles[:, np.abs(offsets) <= 3] == 0, axis=1)
        out["ridge"] = {
            "ct_empty_share": round(float(empty.mean()), 4),
            "air_share": round(float(in_air(profiles, offsets, empty)[~empty].mean()), 4),
            "sampled": ridge["vertices"],
            "sigma3": {k: v for k, v in ridge["sigma_3"].items() if "abs" in k or "within" in k},
            "mean_profile_peak_vox": ridge["mean_profile_peak_vox"],
        }
    for m, segment in enumerate(truth["names"]):
        s = mesh == m
        sk = s & kept
        w = truth["winding"][s]
        detail = {
            "mesh": m,
            "segment": str(segment),
            "wrap": int(truth["mesh_wrap"][m]),
            "frame": int(frame_of_mesh[m]),
            "vertices": int(s.sum()),
            "area_cm2": round(float(area[sk].sum() / 100), 2),
            "z": [int(truth["points"][s, 2].min()), int(truth["points"][s, 2].max())],
            "turns": round(float(np.ptp(w[np.isfinite(w)])), 2) if np.isfinite(w).any() else None,
            "duplicate_share": round(float(1 - kept[s].mean()), 4),
        }
        detail |= {bit: share((flags & value) > 0, sk) for bit, value in bits.items()}
        for p in out["same_surface_pairs"]:
            if "coincident_share" in p and m in p["mesh_index"]:
                mine = p["coincident_share"][p["mesh_index"].index(m)]
                if mine > detail.get("numbered_apart_on", {}).get("share", 0):
                    other = p["meshes"][1 - p["mesh_index"].index(m)]
                    detail["numbered_apart_on"] = {"segment": other, "share": mine}
        partners = sorted((p for p in pairs if m in p["mesh_index"]), key=lambda p: -p["contacts"])
        if partners:
            other = next(x for x in partners[0]["meshes"] if x != str(segment))
            detail["shares_surface_most_with"] = {"segment": other, "contacts": partners[0]["contacts"]}
        detail |= per_ridge.get(m, {})
        judged = s & (vertex_qa < 254) if vertex_qa is not None else None
        if vertex_qa is not None and judged.sum() >= 50:
            detail["all_vertices_judged"] = int(judged.sum())
            detail["all_vertices_zero_ct"] = round(float((vertex_qa[judged] == 1).mean()), 4)
            detail["all_vertices_in_air"] = round(float((vertex_qa[judged] == 2).mean()), 4)
        out["meshes_detail"].append(detail)
    folder = QA / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "numbers.json").write_text(json.dumps(out, indent=1))
    return out


def stretch(image):
    known = image > 0
    if not known.any():
        return np.zeros(image.shape, np.uint8)
    low, high = np.percentile(image[known], (1, 99.7))
    return (np.clip((image.astype(np.float32) - low) / max(high - low, 1), 0, 1) * 239).astype(np.uint8)


def palette_image(gray):
    image = Image.frombytes("P", (gray.shape[1], gray.shape[0]), np.ascontiguousarray(gray).tobytes())
    colours = [(int(v * 255 / 239),) * 3 for v in range(240)] + FIRST + OTHER + [FLAGGED]
    colours += [(0, 0, 0)] * (256 - len(colours))
    image.putpalette([c for rgb in colours for c in rgb])
    return image


def plane(name, z, margin=64, tiles=True):
    """PNG of the level-0 CT at the crossing plane nearest z over the truth's box, crossings drawn; and tiles."""
    truth = packages.load(name)
    entry = truth["record"]["scan"]
    crossings = dict(np.load(packages.PACKAGES / name / "crossings.npz"))
    planes = np.unique(crossings["z"])
    zp = int(planes[np.argmin(np.abs(planes - z))])
    s = crossings["z"] == zp
    xy = crossings["xy"][s].astype(np.float64)
    x0, y0 = np.floor(xy.min(axis=0) - margin).astype(int)
    x1, y1 = np.ceil(xy.max(axis=0) + margin).astype(int)
    with Volume.for_scan(entry["sample"], entry["scan"], "ct", 0) as volume:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, volume.shape[2]), min(y1, volume.shape[1])
        ct, unknown = volume.read_with_unknown((zp, zp + 1, y0, y1, x0, x1))
    gray = stretch(ct[0])
    plain = gray.copy()
    here = {k: crossings[k][s] for k in ("z", "xy", "winding", "mesh", "flags")}
    n, _ = sheet_index(here, truth["knots"], int(truth["sense"]))
    frame_of_mesh = frames_of(truth)
    framed = truth["frame"][truth["frame"] >= 0]
    main = int(np.bincount(framed).argmax()) if len(framed) else -1
    frame = frame_of_mesh[crossings["mesh"][s]]
    bad = (crossings["flags"][s] & (packages.CONFLICT | packages.ORDER)) > 0
    colour = np.where(frame == main, 240, 243) + np.where(np.isfinite(n), np.mod(np.nan_to_num(n), 3), 0).astype(int)
    colour = np.where(bad, 246, colour)
    u = np.round(xy[:, 0] - x0).astype(int)
    v = np.round(xy[:, 1] - y0).astype(int)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = u + du, v + dv
            inside = (uu >= 0) & (uu < gray.shape[1]) & (vv >= 0) & (vv < gray.shape[0])
            gray[vv[inside], uu[inside]] = colour[inside]
    folder = QA / name
    folder.mkdir(parents=True, exist_ok=True)
    image = palette_image(gray)
    image.save(folder / f"z{zp}.png", optimize=False)
    overview = palette_image(plain).convert("RGB")
    overview.thumbnail((1600, 1600), Image.LANCZOS)
    shrink = gray.shape[1] / overview.width
    draw = ImageDraw.Draw(overview)
    palette = FIRST + OTHER + [FLAGGED]
    for a, b, c in zip(u / shrink, v / shrink, colour - 240, strict=True):
        draw.rectangle((a - 1, b - 1, a + 1, b + 1), fill=palette[c])
    draw.text((8, 8), f"{name} z {zp} (1:{shrink:.1f})", fill=(255, 255, 255))
    overview.save(folder / f"z{zp}-overview.png")
    written = [folder / f"z{zp}.png", folder / f"z{zp}-overview.png"]
    centres = []
    if tiles:
        if bad.any():
            cell = np.floor(np.column_stack([u[bad], v[bad]]) / TILE).astype(int)
            keys, counts = np.unique(cell, axis=0, return_counts=True)
            for key in keys[np.argsort(-counts)][:2]:
                inside = np.all(cell == key, axis=1)
                centres.append(("flagged", np.median(np.column_stack([u[bad], v[bad]])[inside], axis=0)))
        knots = truth["knots"]
        cx = np.interp(zp, knots[:, 0], knots[:, 1]) - x0
        cy = np.interp(zp, knots[:, 0], knots[:, 2]) - y0
        radius = np.hypot(u - cx, v - cy)
        for angle in (0, 90, 180, 270):
            near = np.abs(np.angle(np.exp(1j * (np.arctan2(v - cy, u - cx) - np.radians(angle))))) < np.radians(10)
            if near.any():
                r = np.median(radius[near])
                centres.append(
                    (
                        f"angle{angle}",
                        np.array([cx + r * np.cos(np.radians(angle)), cy + r * np.sin(np.radians(angle))]),
                    )
                )
        for i, (why, (cu, cv)) in enumerate(centres):
            a, b = int(cu) - TILE // 2, int(cv) - TILE // 2
            a = min(max(a, 0), max(gray.shape[1] - TILE, 0))
            b = min(max(b, 0), max(gray.shape[0] - TILE, 0))
            tile = image.crop((a, b, a + TILE, b + TILE)).convert("RGB")
            ImageDraw.Draw(tile).text((6, 6), f"{name} z {zp} x {a + x0} y {b + y0} ({why})", fill=(255, 255, 255))
            path = folder / f"z{zp}-tile{i}-{why}.png"
            tile.save(path)
            written.append(path)
    return {
        "plane": zp,
        "box_x_y": [int(x0), int(x1), int(y0), int(y1)],
        "crossings": int(s.sum()),
        "flagged": int(bad.sum()),
        "unknown_chunks": int(unknown.sum()),
        "files": [str(p) for p in written],
    }


def judge(mesh, reference):
    """Recommendation, weight and reasons for one mesh of a package against the package's medians."""
    out, lower = [], []
    sampled = mesh.get("sampled", 0) >= RIDGE_SAMPLED and mesh.get("profile_contrast") is not None
    bad = (mesh["conflict"] or 0) + (mesh["order"] or 0)
    if sampled:
        contrast, offset = mesh["profile_contrast"], mesh["ridge_abs_p50_p90_vox"][0]
        if contrast < 0.08 or (contrast < 0.15 and contrast < 0.5 * reference["contrast"]):
            out.append(f"not on the papyrus: CT profile contrast {contrast} (package median {reference['contrast']})")
        elif contrast < 0.15 or (contrast < 0.2 and contrast < 0.75 * reference["contrast"]):
            lower.append(f"weak CT profile contrast {contrast} (package median {reference['contrast']})")
        if offset > reference["offset"] + 3:
            out.append(f"off the sheet: median {offset} vox to the CT ridge (package median {reference['offset']})")
        elif offset > reference["offset"] + 1.5:
            lower.append(f"median {offset} vox to the CT ridge (package median {reference['offset']})")
    full = "all_vertices_zero_ct" in mesh
    empty = mesh["all_vertices_zero_ct"] if full else (mesh.get("ct_empty_share") or 0)
    counted = full or mesh.get("sampled", 0) >= 50
    if counted and empty > 0.5:
        out.append(f"CT masked out: {empty:.0%} of its vertices lie in zero CT")
    elif counted and empty > 0.1:
        lower.append(f"{empty:.0%} of its vertices lie in zero CT (outside the volume's mask)")
    air = mesh["all_vertices_in_air"] if full else mesh.get("air_share")
    measured = sampled or full
    if measured and air is not None and air > 0.35:
        out.append(f"in air: {air:.0%} of its vertices have no papyrus within 3 voxels")
    elif measured and air is not None and air > 0.2:
        lower.append(f"{air:.0%} of its vertices have no papyrus within 3 voxels")
    apart = mesh.get("numbered_apart_on")
    if apart and apart["share"] >= 0.3:
        lower.append(
            f"{apart['share']:.0%} of its vertices lie on the surface of {apart['segment']}, numbered another winding"
        )
    if bad > 0.25:
        out.append(f"contradicts other meshes: conflict {mesh['conflict']}, order {mesh['order']}")
    elif bad > 0.05:
        lower.append(f"conflict {mesh['conflict']}, order {mesh['order']}")
    partner = mesh.get("shares_surface_most_with", {}).get("segment", "a larger mesh")
    if mesh["duplicate_share"] > 0.9:
        out.append(f"a copy: {mesh['duplicate_share']:.0%} of its vertices repeat {partner}")
    elif mesh["duplicate_share"] > 0.5:
        lower.append(f"{mesh['duplicate_share']:.0%} of its vertices repeat {partner}")
    if out:
        return "exclude", 0.0, out + lower
    if lower:
        return "lower weight", 0.5, lower
    return "include", 1.0, [] if measured else [f"CT unmeasured ({mesh.get('sampled', 0)} vertices sampled)"]


def label_selection(name):
    """The mesh selection recorded in network-training/labels/NAME.npz (ml.winding.labels --only/--exclude, with the module's EXCLUDE
    and ONLY), or None."""
    path = WINDING_LABELS / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as labels:
        if "meta" not in labels.files:
            return None
        return json.loads(str(labels["meta"]))


def selected(segment, selection):
    """Whether ml.winding.labels' selection keeps this mesh."""
    if any(k in segment for k in selection.get("module_exclude", []) + selection.get("exclude", [])):
        return False
    for only in (selection.get("module_only", []), selection.get("only", [])):
        if only and not any(k in segment for k in only):
            return False
    return True


def selection_shares(name, selection):
    """Per selected mesh, the duplicate, conflict and order shares with bench.package's flags taken again among the
    labels' selected meshes inside their zones and window: the package flags every vertex against every mesh, also
    those the selection leaves out (PHercParis4: the 0623 series, which the 0701 meshes repeat and part-cross)."""
    truth = packages.load(name)
    names = [str(n) for n in truth["names"]]
    keep = np.array([selected(n, selection) for n in names])
    inside = keep[truth["mesh"]] & np.isin(truth["zone"], selection.get("zones", []))
    if selection.get("z"):
        inside &= (truth["points"][:, 2] >= selection["z"][0]) & (truth["points"][:, 2] < selection["z"][1])
    rows = np.flatnonzero(inside)
    sizes = np.bincount(truth["mesh"], minlength=len(names))
    size_rank = np.argsort(np.argsort(-sizes, kind="stable"), kind="stable")
    flags = packages.vertex_flags(
        truth["points"][rows].astype(np.float64),
        truth["normals"][rows].astype(np.float64),
        truth["mesh"][rows],
        truth["winding"][rows],
        truth["frame"][rows],
        truth["mesh_wrap"],
        size_rank,
        float(truth["voxel_um"]),
        float(truth["spacing_um"]),
    )
    package_flags = truth["flags"][rows]
    drop = packages.DUPLICATE | packages.CONFLICT | packages.ORDER
    out = {}
    for m in np.flatnonzero(keep):
        s = truth["mesh"][rows] == m
        if not s.any():
            continue
        kept = s & ((flags & packages.DUPLICATE) == 0)
        out[names[m]] = {
            "vertices": int(s.sum()),
            "area_cm2": round(float(truth["area_mm2"][rows][kept].sum() / 100), 2),
            "duplicate_share": round(float(((flags[s] & packages.DUPLICATE) > 0).mean()), 4),
            "conflict": share((flags & packages.CONFLICT) > 0, kept),
            "order": share((flags & packages.ORDER) > 0, kept),
            "dropped_by_other_meshes": round(
                float(
                    (((package_flags[s] & (packages.CONFLICT | packages.ORDER)) > 0) & ((flags[s] & drop) == 0)).mean()
                ),
                4,
            ),
        }
    return out


def table(names, notes=None):
    """BENCH/qa/training-truth.json and .md from the numbers of each package; notes (markdown) go first in the .md.
    Where a package's labels record a mesh selection (ml.winding.labels meta), meshes outside it are listed as left out,
    and when the labels judged duplicates within the selection, the duplicate, conflict and order shares are taken
    again among the selected meshes inside the labels' window (selection_shares)."""
    packages_out = {}
    lines = ["# Training truth", ""]
    if notes:
        lines += [Path(notes).read_text().strip(), ""]
    lines += ["## Rules", "", "Per-mesh recommendation for training labels (bench.qa table).", ""]
    for rule, items in RULES.items():
        lines.append(f"- **{rule}**: " + "; ".join(items) + ".")
    for name in names:
        numbers_ = json.loads((QA / name / "numbers.json").read_text())
        meshes = numbers_["meshes_detail"]
        measured = [m for m in meshes if m.get("sampled", 0) >= RIDGE_SAMPLED and m.get("profile_contrast") is not None]
        reference = {
            "contrast": round(float(np.median([m["profile_contrast"] for m in measured])), 3) if measured else 0.0,
            "offset": round(float(np.median([m["ridge_abs_p50_p90_vox"][0] for m in measured])), 2)
            if measured
            else 0.0,
        }
        selection = label_selection(name)
        within = bool(selection) and selection.get("duplicates") == "judged within the mesh selection"
        again = selection_shares(name, selection) if within else {}
        rows = []
        for m in meshes:
            if selection and not selected(m["segment"], selection):
                verdict, weight, reasons = "left out", 0.0, ["outside the labels' mesh selection"]
            elif within:
                m = m | again.get(m["segment"], {})
                apart = m.get("numbered_apart_on")
                if apart and not selected(apart["segment"], selection):
                    m = {k: v for k, v in m.items() if k != "numbered_apart_on"}
                verdict, weight, reasons = judge(m, reference)
            else:
                verdict, weight, reasons = judge(m, reference)
            rows.append(
                {
                    "segment": m["segment"],
                    "wrap": m["wrap"],
                    "frame": m["frame"],
                    "area_cm2": m["area_cm2"],
                    "recommendation": verdict,
                    "weight": weight,
                    "reasons": reasons,
                    "metrics": {
                        k: m.get(k)
                        for k in (
                            "duplicate_share",
                            "conflict",
                            "order",
                            "touching",
                            "pressed",
                            "sampled",
                            "ridge_abs_p50_p90_vox",
                            "ridge_signed_p50_vox",
                            "null_abs_p50_vox",
                            "profile_contrast",
                            "ct_empty_share",
                            "air_share",
                            "all_vertices_zero_ct",
                            "all_vertices_in_air",
                            "dropped_by_other_meshes",
                        )
                    },
                }
            )
        counts = {
            v: sum(r["recommendation"] == v for r in rows) for v in ("include", "lower weight", "exclude", "left out")
        }
        area = {v: round(sum(r["area_cm2"] for r in rows if r["recommendation"] == v), 1) for v in counts}
        packages_out[name] = {
            "scan": numbers_["scan"],
            "z": numbers_["z"],
            "area_cm2": numbers_["area_cm2"],
            "reference": reference,
            "label_selection": selection,
            "counts": counts,
            "area_by_recommendation_cm2": area,
            "meshes": rows,
        }
        lines += [
            "",
            f"## {name}",
            "",
            f"{numbers_['scan']['sample']} {numbers_['scan']['scan']}, z {numbers_['z'][0]}-{numbers_['z'][1]}, "
            f"{numbers_['area_cm2']} cm2, {numbers_['meshes']} meshes; package medians: contrast "
            f"{reference['contrast']}, {reference['offset']} vox to the ridge. Include {counts['include']} "
            f"({area['include']} cm2), lower weight {counts['lower weight']} ({area['lower weight']} cm2), "
            f"exclude {counts['exclude']} ({area['exclude']} cm2)"
            + (
                f", left out by the labels' selection {counts['left out']} ({area['left out']} cm2; only "
                f"{selection.get('only')}, exclude {selection.get('module_exclude', []) + selection.get('exclude', [])}"
                f", z {selection.get('z')}"
                + (
                    "; for the selected meshes, area and the duplicate, conflict and order shares are taken again"
                    " among the selected meshes inside the window"
                    if within
                    else ""
                )
                + ")"
                if selection
                else ""
            )
            + ".",
            "",
            "| Mesh | Wrap | cm2 | Recommendation | Why |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r["wrap"], r["segment"])):
            why = "; ".join(r["reasons"]) or "-"
            lines.append(f"| {r['segment']} | {r['wrap']} | {r['area_cm2']} | {r['recommendation']} | {why} |")
    QA.mkdir(parents=True, exist_ok=True)
    record = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "rules": RULES, "packages": packages_out}
    (QA / "training-truth.json").write_text(json.dumps(record, indent=1))
    (QA / "training-truth.md").write_text("\n".join(lines) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("numbers")
    a.add_argument("names", nargs="+")
    b = sub.add_parser("planes")
    b.add_argument("name")
    b.add_argument("--z", type=int, nargs="*", help="planes (default: a quarter, half and three quarters of the box)")
    b.add_argument("--no-tiles", action="store_true")
    v = sub.add_parser("vertices")
    v.add_argument("names", nargs="+")
    c = sub.add_parser("table")
    c.add_argument("names", nargs="+")
    c.add_argument("--notes", help="markdown placed first in training-truth.md")
    args = parser.parse_args()
    if args.command == "numbers":
        for name in args.names:
            out = numbers(name)
            print(json.dumps({k: v for k, v in out.items() if k not in ("meshes_detail", "zones")}))
    elif args.command == "planes":
        z0, z1 = packages.definition(args.name)["z"]
        for z in args.z or [z0 + (z1 - z0) * q // 4 for q in (1, 2, 3)]:
            print(json.dumps(plane(args.name, z, tiles=not args.no_tiles)))
    elif args.command == "vertices":
        for name in args.names:
            print(json.dumps({"name": name} | vertex_pass(name)))
    else:
        record = table(args.names, args.notes)
        print(json.dumps({name: p["counts"] for name, p in record["packages"].items()}))


if __name__ == "__main__":
    main()
