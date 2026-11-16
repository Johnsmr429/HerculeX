"""2 x 2 cm windows of a sheet where the ink model is most sure."""

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from sampling.sheet_images import MARGIN_CELLS
from storage.data_roots import ROUTE_OUTPUTS, stage_folder
from website.sheet_tiles import piece_png, seen_from_written_face, upright

CELL_MM = 0.15
WINDOW_MM = 20.0
STEP_MM = 5.0
MIN_SHARE = 0.5
RIM_MM = 1.0
STRONG = 235
MOST_STRONG = 0.25
MOST = 40
SURE_ON_WRITTEN_FACE = 0.1
OTHER_FACE_SHARE = 0.4
MOST_SCRAMBLED = 0.01
Image.MAX_IMAGE_PIXELS = None


def cells_of(image, sheet, step):
    """An ink image on stroke cells."""
    h, w = image.shape[0] // step * step, image.shape[1] // step * step
    blocks, scan = (a[:h, :w].reshape(h // step, step, w // step, step) for a in (image, sheet))
    return (blocks >= STRONG).mean(axis=(1, 3)), (scan > 0).mean(axis=(1, 3)) > 0.5


def looks_of(folder, p):
    try:
        return json.loads((folder / p["unit"] / f"piece-{p['piece']:04d}-taught-side.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def reads_as_written(looks):
    """Whether a piece read as labelled scrolls do when they were left out of training."""
    if not looks:
        return False
    written, other = (looks["near"], looks["far"]) if looks["written"] == "near" else (looks["far"], looks["near"])
    return (
        written >= SURE_ON_WRITTEN_FACE
        and other <= OTHER_FACE_SHARE * written
        and looks.get("scrambled", 0.0) <= MOST_SCRAMBLED
    )


def windows(strong, sheet, side, step):
    """(y, x, share) of windows of a piece, in cells, the share of strong ink over the sheet inside."""
    inner = ni.distance_transform_edt(sheet) * CELL_MM > RIM_MM
    h, w = inner.shape
    if h < side or w < side:
        return []
    mass = np.pad(np.where(inner, strong, 0.0), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    cover = np.pad(inner.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rows = []
    for y in range(0, h - side + 1, step):
        for x in range(0, w - side + 1, step):
            area = cover[y + side, x + side] - cover[y, x + side] - cover[y + side, x] + cover[y, x]
            if area >= MIN_SHARE * side**2:
                rows.append(
                    (y, x, (mass[y + side, x + side] - mass[y, x + side] - mass[y + side, x] + mass[y, x]) / area)
                )
    return rows


def boxes_of(pieces, folder, most=MOST):
    """The surest windows of pieces as they lie on a sheet, apart from each other, pieces whose fibres name the face read first."""
    side, step = int(WINDOW_MM / CELL_MM), int(STEP_MM / CELL_MM)
    found = []
    taught = any(folder.glob("*/piece-*-taught.png"))
    for p in pieces:
        path = piece_png(folder, p, "taught" if taught else "ink")
        scan = piece_png(folder, p, "sheet")
        if p["width_mm"] < WINDOW_MM / 2 or p["height_mm"] < WINDOW_MM / 2 or not path.is_file() or not scan.is_file():
            continue
        looks = looks_of(folder, p) if taught else {}
        if taught and not reads_as_written(looks):
            continue
        with Image.open(path) as ink, Image.open(scan) as image:
            strong, sheet = cells_of(
                upright(np.asarray(ink), p), upright(np.asarray(image), p), round(CELL_MM / p["um"])
            )
        margin = MARGIN_CELLS * p["stride"] * p["um"]
        agreed = taught and looks.get("fibres", {}).get("face") == looks["written"]
        found += [
            (agreed, share, p["x_mm"] - margin + x * CELL_MM, p["y_mm"] - margin + y * CELL_MM)
            for y, x, share in windows(strong, sheet, side, step)
            if share <= MOST_STRONG
        ]
    boxes = []
    for agreed, share, x, y in sorted(found, reverse=True):
        if all(abs(b["x"] - x) >= WINDOW_MM or abs(b["y"] - y) >= WINDOW_MM for b in boxes):
            boxes.append(
                {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "w": WINDOW_MM,
                    "h": WINDOW_MM,
                    "share": round(float(share), 4),
                    "fibres_agree": bool(agreed),
                }
            )
        if len(boxes) == most:
            break
    return boxes, len(found)


def run(sample, volume):
    base = ROUTE_OUTPUTS / sample / volume
    folder = stage_folder(base, "faces") / "level-0"
    taught = any(folder.glob("*/piece-*-taught.png"))
    boxes, count = boxes_of(
        seen_from_written_face(
            json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"],
            folder,
            "taught-side" if taught else "side",
        ),
        folder,
    )
    (stage_folder(base, "sheet") / "proposals.json").write_text(
        json.dumps({"boxes": boxes, "windows": count}, indent=1)
    )
    print(sample, "windows", count, "top", boxes[:2], flush=True)
    return boxes


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
