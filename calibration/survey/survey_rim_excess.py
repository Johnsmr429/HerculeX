"""The sure ink excess of every surveyed piece with its sheet rims cut away."""

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from calibration.survey.scroll_survey import (
    CELL,
    REPLICATED,
    SURVEYED,
    fullest,
    pieces_of,
)
from ink.model.stock_model import SURE

RIMS_MM = (0.0, 0.5, 1.0, 2.0, 3.0)
SURE_LEVEL = int(np.ceil(SURE * 254)) + 1
RECORD = SURVEYED / "rim-excess.json"
Image.MAX_IMAGE_PIXELS = None


def piece_counts(json_path, voxel_mm):
    """Open pixels, forward and reversed sure pixels of a piece's ruler windows at each rim cut, or None for a piece with no reversed map."""
    back = json_path.with_name(f"reversed-{json_path.stem}.png")
    if not back.exists() or not json_path.with_suffix(".npz").exists():
        return None
    near, far = (np.asarray(Image.open(path)) for path in (json_path.with_suffix(".png"), back))
    sheet = near > 0
    distance = ni.distance_transform_edt(sheet) * voxel_mm
    spots, side_y, side_x = fullest(sheet)
    kept = np.load(json_path.with_suffix(".npz"))
    counts = {str(rim): [0, 0, 0] for rim in RIMS_MM}
    for k, (y, x) in enumerate(spots):
        if f"open{k}" not in kept:
            continue
        opened = kept[f"open{k}"].astype(np.float32) > 0.5
        gap = np.zeros((side_y, side_x), dtype=bool)
        gap[: opened.shape[0] * CELL, : opened.shape[1] * CELL] = opened.repeat(CELL, 0).repeat(CELL, 1)
        box = (slice(y, y + side_y), slice(x, x + side_x))
        gap &= sheet[box]
        forward, backward = near[box] >= SURE_LEVEL, far[box] >= SURE_LEVEL
        for rim in RIMS_MM:
            where = gap & (distance[box] >= rim)
            part = counts[str(rim)]
            part[0] += int(where.sum())
            part[1] += int(forward[where].sum())
            part[2] += int(backward[where].sum())
    return counts


def run(items):
    """Bands as sample/volume/band, each scroll kept as soon as it is done."""
    kept = {}
    for item in items:
        sample, volume, band = item.split("/")
        base = pieces_of(sample, volume, band)
        kept[sample] = {}
        for name, folder in (("first", SURVEYED / sample), ("replicate", REPLICATED / sample)):
            rows = {}
            for path in sorted(folder.glob("piece-*.json")):
                voxel_mm = json.loads((base / path.stem / "meta.json").read_text())["voxel_um"] / 1000
                found = piece_counts(path, voxel_mm) if json.loads(path.read_text()).get("pixels", 0) else None
                if found:
                    rows[path.stem] = found
            kept[sample][name] = rows
        RECORD.write_text(json.dumps(kept))
        print(sample, {name: len(rows) for name, rows in kept[sample].items()}, flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
