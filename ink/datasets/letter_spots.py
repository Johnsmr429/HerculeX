"""Windows of a scroll's own pieces where its read shows a few letters, read again by a second reader as targets."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from ink.datasets.own_reads import READS, read_path
from ink.model.stock_model import NO_INK, SURE
from storage.data_roots import LETTER_SPOTS, ROUTE_OUTPUTS, route_id, stage_folder

SIDE, STEP, CELL = 768, 384, 4
SURE_SHOWN = round((SURE - NO_INK) / (1 - 2 * NO_INK) * 255)
WIDE_MM, LEAST_LENGTH_MM = 0.8, 1.0
LEAST_SHEET = 0.98
STROKES = (0.02, 0.35)
MOST_WIDE = 0.25
LEAST_READ_INK, LEAST_INSIDE = 0.005, 0.5
MOST_WINDOWS = 1500
SHOWN = "shown"
WORKERS = 6
Image.MAX_IMAGE_PIXELS = None


def folder_of(sample):
    return LETTER_SPOTS / route_id(ROUTE_OUTPUTS) / sample


def on_cells(mask):
    h, w = mask.shape[0] // CELL * CELL, mask.shape[1] // CELL * CELL
    return mask[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL).mean(axis=(1, 3)) >= 0.5


def strokes_of(sure, cell_mm):
    """Sure ink no wider than a letter's stroke and no shorter than one, and the sure ink too wide to be a stroke."""
    reach = WIDE_MM / 2 / cell_mm
    wide = ni.distance_transform_edt(~(ni.distance_transform_edt(sure) > reach)) <= reach
    wide &= sure
    marked, count = ni.label(sure & ~wide, np.ones((3, 3)))
    long = np.zeros(count + 1, dtype=bool)
    for number, box in enumerate(ni.find_objects(marked), 1):
        long[number] = max(box[0].stop - box[0].start, box[1].stop - box[1].start) * cell_mm >= LEAST_LENGTH_MM
    return long[marked], wide


def summed(mask):
    return np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def windows_of(job):
    """A piece's windows that lie on the sheet and hold strokes at a letter's share, as (strokes, unit, piece, y, x, sure, wide)."""
    faces, unit, piece, um = job
    out = faces / unit / f"piece-{piece:04d}"
    try:
        with Image.open(f"{out}-taught.png") as ink, Image.open(f"{out}.png") as scan:
            sheet = on_cells(np.asarray(scan) > 0)
            sure = on_cells(np.asarray(ink) >= SURE_SHOWN) & sheet
    except FileNotFoundError:
        return []
    side, step = SIDE // CELL, STEP // CELL
    if min(sheet.shape) < side or not sure.any():
        return []
    strokes, wide = strokes_of(sure, CELL * um / 1000)
    sums = [summed(mask) for mask in (sheet, strokes, sure, wide)]
    found = []
    for y in range(0, sheet.shape[0] - side + 1, step):
        for x in range(0, sheet.shape[1] - side + 1, step):
            on, stroke, ink, flood = (
                (s[y + side, x + side] - s[y, x + side] - s[y + side, x] + s[y, x]) / side**2 for s in sums
            )
            if on >= LEAST_SHEET and STROKES[0] <= stroke <= STROKES[1] and flood <= MOST_WIDE * ink:
                found.append((round(stroke, 4), unit, piece, y * CELL, x * CELL, round(ink, 4), round(flood, 4)))
    return found


def candidates(sample, volume):
    """Every piece's letter-like windows, the fullest of strokes first, no two of a piece overlapping."""
    base = ROUTE_OUTPUTS / sample / volume
    faces = stage_folder(base, "faces") / "level-0"
    jobs = []
    for path in sorted(faces.glob("*/piece-*-taught.png")):
        piece = path.name.split("-")[1]
        um = json.loads((base / "pieces" / path.parent.name / f"piece-{piece}" / "meta.json").read_text())["voxel_um"]
        jobs.append((faces, path.parent.name, int(piece), um))
    with ProcessPoolExecutor(WORKERS) as pool:
        found = sorted((row for rows in pool.map(windows_of, jobs, chunksize=16) for row in rows), reverse=True)
    chosen, taken = [], {}
    for stroke, unit, piece, y, x, ink, flood in found:
        near = taken.setdefault((unit, piece), [])
        if all(abs(y - a) >= SIDE or abs(x - b) >= SIDE for a, b in near):
            near.append((y, x))
            chosen.append({"unit": unit, "piece": piece, "y": y, "x": x, "strokes": stroke, "sure": ink, "wide": flood})
    folder_of(sample).mkdir(parents=True, exist_ok=True)
    (folder_of(sample) / "candidates.json").write_text(
        json.dumps({"scan": volume, "pieces": len(jobs), "windows_found": len(found), "windows": chosen}, indent=1)
    )
    print(sample, "pieces", len(jobs), "windows", len(found), "apart", len(chosen), "in pieces", len(taken), flush=True)


def build(sample, volume, weights):
    """The candidates read by a second reader, kept where it draws ink inside the first read's strokes."""
    from ink.model.turned_reader import named_reader
    from ink.reading.piece_reading import kept_face, kept_window

    folder, base = folder_of(sample), ROUTE_OUTPUTS / sample / volume
    rows = json.loads((folder / "candidates.json").read_text())["windows"][:MOST_WINDOWS]
    model, reader = named_reader(weights)
    for name in (READS, SHOWN):
        (folder / name).mkdir(exist_ok=True)
    pieces = {}
    for row in rows:
        pieces.setdefault((row["unit"], row["piece"]), []).append(row)
    kept = 0
    for (unit, piece), windows in pieces.items():
        wanted = [w for w in windows if not (folder / f"{name_of(w)}.json").exists()]
        stack_path = kept_window(base / "pieces" / unit / f"piece-{piece:04d}")
        if not wanted or not stack_path.exists():
            continue
        face = kept_face(stack_path)
        with Image.open(stage_folder(base, "faces") / "level-0" / unit / f"piece-{piece:04d}-taught.png") as image:
            shown = np.asarray(image)
        for row in wanted:
            here = (slice(row["y"], row["y"] + SIDE), slice(row["x"], row["x"] + SIDE))
            stack = np.ascontiguousarray(face[:, here[0], here[1]])
            read = reader.read(stack)
            ink, first = read >= SURE, shown[here] >= SURE_SHOWN
            numbers = {
                "read_ink": round(float(ink.mean()), 4),
                "inside": round(float((ink & first).sum() / max(ink.sum(), 1)), 4),
                "first_ink": round(float(first.mean()), 4),
            }
            numbers["kept"] = numbers["read_ink"] >= LEAST_READ_INK and numbers["inside"] >= LEAST_INSIDE
            if numbers["kept"]:
                np.save(folder / f"{name_of(row)}.npy", stack)
                np.save(read_path(folder / f"{name_of(row)}.npy"), np.round(read * 255).astype(np.uint8))
                np.save(folder / SHOWN / f"{name_of(row)}.npy", shown[here])
                kept += 1
            (folder / f"{name_of(row)}.json").write_text(json.dumps(row | numbers | {"reader": model}))
        print(unit, piece, len(wanted), "windows read,", kept, "kept so far", flush=True)


def name_of(row):
    return f"{row['unit']}-piece-{row['piece']:04d}-{row['y']:05d}-{row['x']:05d}"


if __name__ == "__main__":
    {"candidates": candidates, "build": build}[sys.argv[1]](*sys.argv[2:])
