"""Hand marks on a sheet carried back to piece pixels and scan voxels, so they outlive any layout."""

import json

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from sampling import column_sampling as sampler
from sampling.sheet_images import MARGIN_CELLS, extended
from storage.data_roots import PRODUCTION_FACES, ROUTE_OUTPUTS, stage_folder
from website.site_server import folders_of, marks_of

Image.MAX_IMAGE_PIXELS = None
TAUGHT_LAYOUTS_KEPT_FROM = "2026-11-10T18:30:18Z"  # 321c2a7b; taught strokes before it kept placed.json's digest.


def faces_of(sample, volume):
    return stage_folder(ROUTE_OUTPUTS / sample / volume, "faces") / PRODUCTION_FACES


def scan_key(sample, volume):
    return f"level/{sample}/{volume}"


def placement_of(stroke, marks, sheet):
    """The placement a stroke was drawn on: its kept layout, else the one its layer is laid on now."""
    taught = "taught" in stroke.get("layer", "")
    kept = marks / "layouts" / f"{stroke.get('layout')}.json"
    if stroke.get("layout") and kept.exists() and (not taught or stroke["time"] >= TAUGHT_LAYOUTS_KEPT_FROM):
        return kept
    names = ("placed-taught.json", "placed.json", "layout.json") if taught else ("placed.json", "layout.json")
    return next(sheet / name for name in names if (sheet / name).exists())


def scan_path(faces, p):
    return faces / p["unit"] / f"piece-{p['piece']:04d}.png"


def corner_of(p, pixel_mm):
    """Where a piece's image starts on the sheet, in sheet pixels, as the composer lays it."""
    margin = round(MARGIN_CELLS * p["stride"] * p["um"] / pixel_mm)
    return round(p["x_mm"] / pixel_mm) - margin, round(p["y_mm"] / pixel_mm) - margin, margin


def on_piece(p, image, points, pixel_mm):
    """(rows, columns, own, seen) of sheet points on a piece's image as stored, before the layout turned it."""
    left, top, margin = corner_of(p, pixel_mm)
    u, v = (
        np.rint(points[:, 0] / pixel_mm - left).astype(np.int64),
        np.rint(points[:, 1] / pixel_mm - top).astype(np.int64),
    )
    h, w = image.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    own = inside & (u >= margin) & (u < w - margin) & (v >= margin) & (v < h - margin)
    rows, columns = (h - 1 - v if p["flip_rows"] else v), (w - 1 - u if p["flip_cols"] else u)
    seen = np.zeros(len(points), dtype=bool)
    seen[inside] = image[rows[inside], columns[inside]] > 0
    return rows, columns, own & seen, seen


def voxels_of(p, unit_folder, rows, columns):
    """Scan voxels (x, y, z) of pixels of a piece's image, on the mesh its stack is sampled on."""
    xyz, meta = sampler.load_surface(unit_folder / f"piece-{p['piece']:04d}")
    xyz = sampler.readable_mesh(xyz, meta["stride"])
    grid = extended(np.where(xyz[0] > 0, xyz, -1.0).astype(np.float32))
    at = np.stack([rows, columns]) / meta["stride"]
    return np.stack([ni.map_coordinates(plane, at, order=1, mode="nearest") for plane in grid], axis=1)


def carried_back(sample, volume):
    """Every standing stroke point by point on the piece shown at that place, as laid when it was drawn."""
    base = ROUTE_OUTPUTS / sample / volume
    faces = faces_of(sample, volume)
    marks, sheet = folders_of(scan_key(sample, volume))
    placements = {}
    found = []
    for stroke in marks_of(scan_key(sample, volume)):
        path = placement_of(stroke, marks, sheet)
        if path not in placements:
            pieces = json.loads(path.read_text())["pieces"]
            placements[path] = sorted((p for p in pieces if scan_path(faces, p).exists()), key=lambda p: -p["area"]), {}
        read, images = placements[path]
        pixel_mm = read[0]["um"]
        points = np.array(stroke["points"], dtype=np.float64)
        owner = np.full(len(points), -1)
        place = np.zeros((len(points), 2), dtype=np.int64)
        x0, y0, x1, y1 = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()
        near = [
            k
            for k, p in enumerate(read)
            if p["x_mm"] - 1 < x1
            and p["x_mm"] + p["width_mm"] + 1 > x0
            and p["y_mm"] - 1 < y1
            and p["y_mm"] + p["height_mm"] + 1 > y0
        ]
        for use_margin in (False, True):
            for k in near:
                if k not in images:
                    images[k] = np.asarray(Image.open(scan_path(faces, read[k])))
                rows, columns, own, seen = on_piece(read[k], images[k], points, pixel_mm)
                free = (owner < 0) & (seen if use_margin else own)
                owner[free] = k
                place[free] = np.stack([rows[free], columns[free]], axis=1)
        for k in sorted(set(owner[owner >= 0])):
            part = owner == k
            p = read[k]
            found.append(
                {
                    "stroke": stroke["id"],
                    "unit": p["unit"],
                    "piece": p["piece"],
                    "turn": p.get("turn", 1),
                    "width_px": stroke["width_mm"] / pixel_mm,
                    "pixels": place[part],
                    "sheet_mm": points[part],
                    "voxels": voxels_of(p, base / "pieces" / p["unit"], place[part, 0], place[part, 1]),
                    "time": stroke["time"],
                }
            )
        if (owner < 0).any():
            print("off every piece", stroke["id"][:8], int((owner < 0).sum()), "of", len(points), flush=True)
    return found


def kept(found, path):
    """Carried strokes in one compressed file, point by point."""
    units = sorted({part["unit"] for part in found})
    lengths = [len(part["pixels"]) for part in found]
    np.savez_compressed(
        path,
        units=np.array(units),
        part_of_point=np.repeat(np.arange(len(found), dtype=np.int32), lengths),
        stroke=np.array([part["stroke"] for part in found]),
        unit=np.array([units.index(part["unit"]) for part in found], dtype=np.int16),
        piece=np.array([part["piece"] for part in found], dtype=np.int32),
        turn=np.array([part["turn"] for part in found], dtype=np.int8),
        width_px=np.array([part["width_px"] for part in found], dtype=np.float32),
        time=np.array([part["time"] for part in found]),
        pixels=np.concatenate([part["pixels"] for part in found]).astype(np.int32),
        sheet_mm=np.concatenate([part["sheet_mm"] for part in found]).astype(np.float32),
        voxels=np.concatenate([part["voxels"] for part in found]).astype(np.float32),
    )


def run(sample, volume, out):
    found = carried_back(sample, volume)
    kept(found, out)
    print(
        len(found), "stroke parts on", len({(p["unit"], p["piece"]) for p in found}), "pieces, kept in", out, flush=True
    )


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sys.argv[2], sys.argv[3])
