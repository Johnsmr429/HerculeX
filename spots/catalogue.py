"""Every circle drawn with the viewer's Spot tool, located on its pieces and in the scan, as the spots table.

./run.sh spots.catalogue            rebuild the table from the live marks and print a summary

A circle is a closed outline in sheet millimetres on the placement it was drawn on (kept beside the marks under
its layout digest). Each piece of that placement is a grid of mesh nodes every stride voxels; a node's place on the
sheet follows from the piece's x_mm, y_mm, flips and node spacing. Where pieces overlap, the sheet shows the
larger piece first and its own cells before its margins, as sheet_tiles paints them, and each sheet cell under the
circle is given to the piece shown there. The nodes the circle covers give the spot's place in the scan.

Every standing circle is a target. HERCULEX_SPOTS_SINCE=YYYY-MM-DD makes only the circles drawn on or after that day
targets; the older ones stay in the table, marked as not targets.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from spots import database
from spots.sources import all_sheets, logs
from website.site_server import standing

TARGETS_SINCE = os.environ.get("HERCULEX_SPOTS_SINCE", "")
CLICK_MM, CLICK_BOX_MM = 2.0, 10.0
REACH_MM = 1.0  # pieces looked at reach this far past the outline's box
SAME_PLACE_MM = 0.3  # a node of one circle this close to a node of another counts as the same papyrus
SAME_PLACE_SHARE = 0.25


def outline_of(mark):
    """A circle's outline in sheet mm; a click becomes a square of CLICK_BOX_MM round it."""
    points = np.asarray(mark["points"], dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    if (high - low).max() < CLICK_MM:
        middle, half = (low + high) / 2, CLICK_BOX_MM / 2
        return middle + np.array([[-half, -half], [half, -half], [half, half], [-half, half]]), True
    return points, False


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def inside(xy, polygon):
    """Which points lie inside a closed polygon, by crossing count."""
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    ax, ay = polygon[:, 0][None], polygon[:, 1][None]
    bx, by = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    straddles = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ax + (y - ay) * (bx - ax) / (by - ay)
    return ((straddles & (x < cross)).sum(axis=1) % 2) == 1


def node_places(p):
    """Sheet mm of every node of a piece's grid, as (rows, columns) of x and of y."""
    rows, columns = p["shape"]
    step = p["stride"] * p["um"]
    r, c = np.arange(rows), np.arange(columns)
    r = rows - 1 - r if p["flip_rows"] else r
    c = columns - 1 - c if p["flip_cols"] else c
    y, x = np.meshgrid(p["y_mm"] + r * step, p["x_mm"] + c * step, indexing="ij")
    return x, y


def own_cells(p):
    """Which nodes of a piece are its own rather than shared with a neighbour (overlap is top, bottom, left, right)."""
    rows, columns = p["shape"]
    top, bottom, left, right = p.get("overlap", (0, 0, 0, 0))
    own = np.zeros((rows, columns), dtype=bool)
    own[top : rows - bottom, left : columns - right] = True
    return own


def piece_folder(sheet, p):
    return sheet.pieces / p["unit"] / f"piece-{p['piece']:04d}"


def located(sheet, mark, placed):
    """The cells a circle covers, the piece shown at each, and their nodes in the scan."""
    outline, clicked = outline_of(mark)
    low, high = outline.min(axis=0) - REACH_MM, outline.max(axis=0) + REACH_MM
    near = [
        p
        for p in placed
        if p["x_mm"] < high[0]
        and p["x_mm"] + p["width_mm"] > low[0]
        and p["y_mm"] < high[1]
        and p["y_mm"] + p["height_mm"] > low[1]
    ]
    step = near[0]["stride"] * near[0]["um"] if near else 16 * 0.009362
    claimed = {}
    found = {}
    for margins in (False, True):
        for k, p in sorted(enumerate(near), key=lambda kp: -kp[1]["area"]):
            folder = piece_folder(sheet, p)
            if k not in found:
                xyz = np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"])
                meta = json.loads((folder / "meta.json").read_text())
                found[k] = (xyz, meta)
            xyz, meta = found[k]
            x, y = node_places(p)
            wanted = (xyz[0] > 0) & (own_cells(p) if not margins else np.ones(x.shape, dtype=bool))
            if xyz.shape[1:] != x.shape:
                wanted &= False
            rows, columns = np.nonzero(wanted)
            if not len(rows):
                continue
            xy = np.stack([x[rows, columns], y[rows, columns]], axis=1)
            hit = inside(xy, outline)
            for r, c, (px, py) in zip(rows[hit], columns[hit], xy[hit]):
                cell = (round(py / step), round(px / step))
                if cell not in claimed:
                    claimed[cell] = (k, r, c)
    expected = cells_inside(outline, step)
    per_piece = defaultdict(list)
    for k, r, c in claimed.values():
        per_piece[k].append((r, c))
    pieces, points = [], []
    for k, cells in sorted(per_piece.items(), key=lambda kc: -len(kc[1])):
        p = near[k]
        xyz, meta = found[k]
        rc = np.array(cells)
        xyz_here = xyz[:, rc[:, 0], rc[:, 1]].T  # x, y, z in voxels
        points.append(xyz_here)
        shown = xyz.shape[1:] == tuple(p["shape"])
        pieces.append(
            {
                "unit": p["unit"],
                "piece": int(p["piece"]),
                "cells": len(cells),
                "share": round(len(cells) / max(expected, 1), 4),
                "winding": meta.get("winding"),
                "level": p.get("level"),
                "group": str(p.get("group")),
                "turn": int(p.get("turn", 1)),
                "area_mm2": round(float(meta.get("area_mm2", p["area"])), 2),
                "matches_layout": bool(
                    shown and abs(float(meta.get("width_mm", p["width_mm"])) - p["width_mm"]) < 1e-3
                ),
            }
        )
    points = np.concatenate(points) if points else np.zeros((0, 3), dtype=np.float32)
    return outline, clicked, pieces, points, expected, step


def cells_inside(outline, step):
    low, high = outline.min(axis=0), outline.max(axis=0)
    ys = np.arange(np.floor(low[1] / step), np.ceil(high[1] / step) + 1) * step
    xs = np.arange(np.floor(low[0] / step), np.ceil(high[0] / step) + 1) * step
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return int(inside(grid, outline).sum())


def row_of(sheet, mark, voxel_um=9.362):
    placed = sheet.layout(mark["layout"])
    outline, clicked, pieces, points, expected, step = located(sheet, mark, placed)
    labeller = "target-day" if mark["time"][:10] >= TARGETS_SINCE else "earlier"
    centre = points.mean(axis=0) if len(points) else np.full(3, np.nan)
    low = points.min(axis=0) if len(points) else np.full(3, np.nan)
    high = points.max(axis=0) if len(points) else np.full(3, np.nan)
    windings = sorted({p["winding"] for p in pieces if p["winding"] is not None})
    width, height = (outline.max(axis=0) - outline.min(axis=0)).tolist()
    return {
        "spot_id": f"{mark['id'][:8]}",
        "mark_id": mark["id"],
        "sheet_key": sheet.key,
        "part": sheet.part,
        "page": sheet.page,
        "scroll": sheet.scroll,
        "scan": sheet.scan,
        "route": sheet.route,
        "labeller": labeller,
        "target": labeller == "target-day",
        "marked_at": mark["time"],
        "layer": mark.get("layer", ""),
        "layout": mark["layout"],
        "clicked": clicked,
        "outline_mm": outline.round(3).tolist(),
        "x_mm": round(float(outline[:, 0].mean()), 2),
        "y_mm": round(float(outline[:, 1].mean()), 2),
        "width_mm": round(width, 2),
        "height_mm": round(height, 2),
        "area_cm2": round(polygon_area(outline) / 100, 3),
        "cells_inside": expected,
        "cells_covered": int(sum(p["cells"] for p in pieces)),
        "covered_share": round(sum(p["cells"] for p in pieces) / max(expected, 1), 4),
        "pieces": pieces,
        "n_pieces": len(pieces),
        "main_piece": f"{pieces[0]['unit']}/piece-{pieces[0]['piece']:04d}" if pieces else "",
        "main_share": pieces[0]["share"] if pieces else 0.0,
        "windings": windings,
        "layout_matches": all(p["matches_layout"] for p in pieces),
        "centre_xyz": [round(float(v), 1) for v in centre],
        "box_low_xyz": [round(float(v), 1) for v in low],
        "box_high_xyz": [round(float(v), 1) for v in high],
        "node_mm": round(step, 5),
        "voxel_um": voxel_um,
    }, points


def same_places(rows, points):
    """Circles on different sheets of one scan that cover the same papyrus, as one place id each."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    trees = [cKDTree(p[:: max(1, len(p) // 4000)]) if len(p) else None for p in points]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if (a["scroll"], a["scan"]) != (b["scroll"], b["scan"]) or trees[i] is None or trees[j] is None:
                continue
            reach = SAME_PLACE_MM * 1000 / a["voxel_um"]
            if np.linalg.norm(np.array(a["centre_xyz"]) - np.array(b["centre_xyz"])) > 80 * reach:
                continue
            near_ab = (trees[j].query(trees[i].data, distance_upper_bound=reach)[0] < np.inf).mean()
            near_ba = (trees[i].query(trees[j].data, distance_upper_bound=reach)[0] < np.inf).mean()
            if max(near_ab, near_ba) >= SAME_PLACE_SHARE:
                parent[root(i)] = root(j)
    groups = {}
    for i, r in enumerate(rows):
        g = root(i)
        groups.setdefault(
            g, f"{r['scroll']}-place-{len([k for k in groups.values() if k.startswith(r['scroll'])]) + 1:03d}"
        )
        r["place"] = groups[g]
    counts = defaultdict(int)
    for r in rows:
        counts[r["place"]] += 1
    for r in rows:
        r["place_circles"] = counts[r["place"]]


def build():
    sources = list(all_sheets())
    known = {s.key for s in sources}
    unknown = [k for k in logs() if k not in known]
    if unknown:
        raise SystemExit(f"marks logs with no sheet entry in spots/sources.py: {unknown}")
    rows, points = [], []
    by_key = defaultdict(list)
    for s in sources:
        by_key[s.key].append(s)
    for sheets in by_key.values():
        for mark in standing(sheets[0].log):
            if mark.get("kind") != "spot":
                continue
            best = None
            for s in sheets:
                row, pts = row_of(s, mark)
                if best is None or row["cells_covered"] > best[0]["cells_covered"]:
                    best = (row, pts)
            rows.append(best[0])
            points.append(best[1])
    same_places(rows, points)
    previous = {r["spot_id"]: r for r in database.rows("spots")}
    for r in rows:
        r["status"] = previous.get(r["spot_id"], {}).get("status", "catalogued")
    database.write("spots", rows)
    gone = sorted(set(previous) - {r["spot_id"] for r in rows})
    return rows, gone


def report(rows, gone):
    targets = [r for r in rows if r["target"]]
    print(len(rows), "standing circles,", len(targets), "of them targets;", len(gone), "no longer standing:", gone)
    by_sheet = defaultdict(list)
    for r in targets:
        by_sheet[(r["scroll"], r["page"], r["part"])].append(r)
    for (scroll, page, part), rs in sorted(by_sheet.items()):
        places = len({r["place"] for r in rs})
        print(
            f"  {scroll:11s} {page:42s} {part:5s} circles {len(rs):3d}  places {places:3d}  "
            f"area {sum(r['area_cm2'] for r in rs):6.1f} cm²  covered {np.mean([r['covered_share'] for r in rs]):.2f}  "
            f"layout ok {sum(r['layout_matches'] for r in rs)}/{len(rs)}"
        )
    print("places with target circles:", len({r["place"] for r in targets}))


if __name__ == "__main__":
    report(*build())
    print("table", Path(database.path_of("spots")))
