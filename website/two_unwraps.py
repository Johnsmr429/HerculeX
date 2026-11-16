"""Both unwrapping routes of one scan on one sheet."""

import copy
import json
import multiprocessing
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from sampling import column_sampling as sampler
from sampling.sheet_images import mosaics_of
from storage.data_roots import BROWSER_ASSETS, GROWN_ROUTE, LEVEL_ROUTE, SHEETS, stage_folder
from storage.tile_store import TileWriter
from website.sheet_tiles import (
    PROCESSES,
    SCAN_UNDER_INK,
    SCAN_UNDER_TAUGHT,
    TILE,
    WEAKEST_INK,
    compose,
    seen_from_written_face,
)
from website.site_pages import CARD, OVERVIEW_LEVEL, crop, shelves_of_pieces
from website.window_proposals import boxes_of

CLOSE_VOXELS = 6.0
NEW_EVERY, FIRST_EVERY = 2, 3
LEAST_CELLS = 30
AGREE_MM = 3.0


def cells_of(base, p):
    xyz, _ = sampler.load_surface(base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}")
    i, j = np.nonzero(xyz[0] > 0)
    return xyz[:, i, j].T.astype(np.float64), i, j


def on_sheet(p, i, j, flip_rows=None, flip_cols=None):
    """Sheet place in mm of a piece's cells, its own corner at (0, 0) when it has no place yet."""
    side = p["stride"] * p["um"]
    rows, columns = p["shape"]
    flip_rows, flip_cols = (
        p["flip_rows"] if flip_rows is None else flip_rows,
        p["flip_cols"] if flip_cols is None else flip_cols,
    )
    return np.where(flip_cols, columns - 1 - j, j) * side, np.where(flip_rows, rows - 1 - i, i) * side


def densest(dx, dy):
    """The offset most shared cells agree on, and how many do."""
    cell = np.floor(dx / AGREE_MM).astype(np.int64) * 100003 + np.floor(dy / AGREE_MM).astype(np.int64)
    values, counts = np.unique(cell, return_counts=True)
    chosen = cell == values[np.argmax(counts)]
    x, y = float(np.median(dx[chosen])), float(np.median(dy[chosen]))
    return x, y, int(((np.abs(dx - x) <= AGREE_MM) & (np.abs(dy - y) <= AGREE_MM)).sum())


def first_on_sheet(first, new, first_base, new_base):
    """The first route's pieces placed on the second's sheet by the cells they share, unit by unit, facing as the cells say."""
    start = time.time()
    out, shared, alone = [], 0, 0
    for unit in sorted({p["unit"] for p in first}):
        points, sheet_x, sheet_y = [], [], []
        for p in (n for n in new if n["unit"] == unit):
            xyz, i, j = cells_of(new_base, p)
            x, y = on_sheet(p, i[::NEW_EVERY], j[::NEW_EVERY])
            points.append(xyz[::NEW_EVERY])
            sheet_x.append(x + p["x_mm"])
            sheet_y.append(y + p["y_mm"])
        if not points:
            continue
        tree, sheet_x, sheet_y = cKDTree(np.concatenate(points)), np.concatenate(sheet_x), np.concatenate(sheet_y)
        for p in (f for f in first if f["unit"] == unit):
            xyz, i, j = cells_of(first_base, p)
            xyz, i, j = xyz[::FIRST_EVERY], i[::FIRST_EVERY], j[::FIRST_EVERY]
            distance, index = tree.query(xyz, distance_upper_bound=CLOSE_VOXELS, workers=4)
            hit = np.isfinite(distance)
            q = copy.deepcopy(p)
            if hit.sum() >= LEAST_CELLS:
                best = max(
                    (
                        (
                            densest(
                                sheet_x[index[hit]] - on_sheet(p, i[hit], j[hit], r, c)[0],
                                sheet_y[index[hit]] - on_sheet(p, i[hit], j[hit], r, c)[1],
                            ),
                            r,
                            c,
                        )
                        for r in (False, True)
                        for c in (False, True)
                    ),
                    key=lambda t: t[0][2],
                )
                (q["x_mm"], q["y_mm"], _), q["flip_rows"], q["flip_cols"] = best
                q["shared"] = float(hit.mean())
                shared += 1
            else:
                _, near = tree.query(xyz[len(xyz) // 2])
                x, y = on_sheet(p, i[len(i) // 2 : len(i) // 2 + 1], j[len(j) // 2 : len(j) // 2 + 1])
                q["x_mm"], q["y_mm"] = float(sheet_x[near] - x[0]), float(sheet_y[near] - y[0])
                q["shared"] = float(hit.mean())
                alone += 1
            q["group"] = f"first-{p.get('group')}"
            out.append(q)
        print(
            unit,
            len(out),
            "placed",
            shared,
            "by shared cells,",
            alone,
            "alone,",
            round(time.time() - start),
            "s",
            flush=True,
        )
    low_x, low_y = min(q["x_mm"] for q in out), min(q["y_mm"] for q in out)
    return out, low_x, low_y


def run(sample, volume, pixel_mm):
    first_base, new_base = GROWN_ROUTE / sample / volume, LEVEL_ROUTE / sample / volume
    new_faces, first_faces = (stage_folder(b, "faces") / "level-0" for b in (new_base, first_base))
    new = seen_from_written_face(
        json.loads((stage_folder(new_base, "sheet") / "layout.json").read_text())["pieces"], new_faces, "taught-side"
    )
    first, low_x, low_y = first_on_sheet(
        json.loads((stage_folder(first_base, "sheet") / "placed-taught.json").read_text())["pieces"],
        new,
        first_base,
        new_base,
    )
    shift_x, shift_y = max(0.0, -low_x), max(0.0, -low_y)
    for p in (*new, *first):
        p["x_mm"] += shift_x
        p["y_mm"] += shift_y
    out = SHEETS / f"two-unwraps-{sample}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "placed.json").write_text(json.dumps({"new": new, "first": first}))
    layers = []
    with multiprocessing.get_context("spawn").Pool(PROCESSES) as pool, TileWriter(out) as writer:
        for name, pieces, faces, layer in (
            ("first-ink", first, first_faces, "taught"),
            ("new-ink", new, new_faces, "taught"),
            ("first-scan", first, first_faces, SCAN_UNDER_TAUGHT),
            ("new-scan", new, new_faces, SCAN_UNDER_TAUGHT),
        ):
            layers.append(compose(pieces, faces, writer, name, layer, pixel_mm, pool))
    (out / "layers.json").write_text(json.dumps(layers))
    pages(out, sample, volume, new, first)


def squares(sample, volume, new, first):
    """Each mesh's proposal windows where this page lays its pieces."""
    found = {}
    for name, pieces, base in (("first", first, GROWN_ROUTE), ("new", new, LEVEL_ROUTE)):
        found[name] = boxes_of(pieces, stage_folder(base / sample / volume, "faces") / "level-0")[0]
        print(name, "mesh squares", len(found[name]), flush=True)
    return found


def pages(out, sample, volume, new, first):
    layers = json.loads((out / "layers.json").read_text())
    shelves = shelves_of_pieces([*new, *first])
    whole = {"x": 0, "y": 0, "w": max(s["w"] for s in shelves), "h": shelves[-1]["y"] + shelves[-1]["h"]}
    places = []
    for members in sorted(mosaics_of(new), key=lambda m: -sum(p["area"] for p in m))[:200]:
        x0, y0 = min(p["x_mm"] for p in members), min(p["y_mm"] for p in members)
        x1, y1 = max(p["x_mm"] + p["width_mm"] for p in members), max(p["y_mm"] + p["height_mm"] for p in members)
        places.append({"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)})
    title = f"{sample} · two unwraps"
    meta = {
        "title": title,
        "layers": layers,
        "tile": TILE,
        "area": whole,
        "back": "../index.html",
        "places": places,
        "shelves": shelves,
        "squares": squares(sample, volume, new, first),
        "scan": f"two-unwraps/{sample}",
    }
    html = (
        (BROWSER_ASSETS / "two_unwraps.html")
        .read_text()
        .replace("__TITLE__", title)
        .replace("__META__", json.dumps(meta))
        .replace("__FLOOR__", str(WEAKEST_INK))
        .replace("__SCAN__", str(SCAN_UNDER_INK))
    )
    (out / "index.html").write_text(html)
    scan = next(layer for layer in layers if layer["path"] == "new-scan")
    if places:
        w, h = CARD[0] * scan["mm"] * 2**OVERVIEW_LEVEL, CARD[1] * scan["mm"] * 2**OVERVIEW_LEVEL
        box = {
            "x": places[0]["x"] + places[0]["w"] / 2 - w / 2,
            "y": places[0]["y"] + places[0]["h"] / 2 - h / 2,
            "w": w,
            "h": h,
        }
        crop(out, scan, box, OVERVIEW_LEVEL).save(out / "card.jpg", quality=92)


if __name__ == "__main__":
    if sys.argv[1] == "pages":
        out = SHEETS / f"two-unwraps-{sys.argv[2]}"
        placed = json.loads((out / "placed.json").read_text())
        pages(out, sys.argv[2], sys.argv[3], placed["new"], placed["first"])
    else:
        run(sys.argv[1], sys.argv[2], float(sys.argv[3]))
