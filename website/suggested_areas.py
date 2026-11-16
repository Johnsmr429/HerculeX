"""Squares worth the eye first on a sheet, in the sheet's millimetres."""

import json

from PIL import Image

from calibration.text.blind_pairs import FIELD, SAMPLE, VOLUME
from sampling.sheet_images import MARGIN_CELLS
from storage.data_roots import INK_RECORDS, ROUTE_OUTPUTS, stage_folder

PAIRS = (29, 21, 17, 14, 25)
Image.MAX_IMAGE_PIXELS = None


def square_on_sheet(p, corner, shape):
    """A square of a piece's stack on the sheet, the piece placed and turned as the viewer shows it."""
    (y, x), (height, width) = corner, shape
    margin = MARGIN_CELLS * p["stride"] * p["um"]
    y, x = height - y - FIELD if p["flip_rows"] else y, width - x - FIELD if p["flip_cols"] else x
    return {
        "x": round(p["x_mm"] - margin + x * p["um"], 2),
        "y": round(p["y_mm"] - margin + y * p["um"], 2),
        "w": round(FIELD * p["um"], 2),
        "h": round(FIELD * p["um"], 2),
    }


def squares(sample, volume):
    """A sheet's suggested squares, best first."""
    scored = INK_RECORDS / "blind-pairs.json"
    if (sample, volume) != (SAMPLE, VOLUME) or not scored.exists():
        return []
    base = ROUTE_OUTPUTS / sample / volume
    placed = {
        (p["unit"], p["piece"]): p
        for p in json.loads((stage_folder(base, "sheet") / "placed-taught.json").read_text())["pieces"]
    }
    pairs = {p["number"]: p for p in json.loads(scored.read_text())["ranked"]}
    found = []
    for rank, number in enumerate(PAIRS, 1):
        pair = pairs[number]
        p = placed.get((pair["unit"], pair["piece"]))
        if p is None:
            continue
        path = stage_folder(base, "faces") / "level-0" / p["unit"] / f"piece-{p['piece']:04d}-taught.png"
        try:
            with Image.open(path) as image:
                width, height = image.size
        except FileNotFoundError:
            continue
        found.append(
            square_on_sheet(p, pair["corner"], (height, width)) | {"label": f"{rank} · {p['unit']} piece {p['piece']}"}
        )
    return found
