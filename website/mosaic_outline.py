"""A mosaic's border drawn on its sheet's viewer, traced from its pieces' own cells.

Every piece of the mosaic is laid where the page laid it (placed.json: its corner, its flips) at its own cell size
(stride x voxel, 0.15 mm at 9.362 um), keeping the cells the piece holds a surface for and owns (not the overlap it
shares with a neighbour, spots.catalogue.own_cells). The union, closed over the seams between pieces and with its
holes filled, is traced round each part of at least SMALLEST_CM2, simplified to half a cell and kept in the page's meta as `outlines`,
which the viewer draws in amber with the boxes. Nothing is recomposed; the marks are untouched.

./run.sh website.mosaic_outline PAGE SAMPLE VOLUME [RANK]     RANK 1 is the largest mosaic
"""

import json
import sys

import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure

from sampling.sheet_images import mosaics_of
from spots.catalogue import own_cells
from storage.data_roots import LEVEL_ROUTE, SHEETS, stage_folder
from website.refresh_viewers import refresh

SMALLEST_CM2 = 1.0  # parts of the mosaic smaller than this are not outlined
SEAM_CELLS = 3  # gaps between pieces closed before tracing


def outline(sample, volume, rank=1):
    base = LEVEL_ROUTE / sample / volume
    placed = json.loads((stage_folder(base, "sheet") / "placed.json").read_text())["pieces"]
    pieces = sorted(mosaics_of(placed), key=lambda g: -sum(p["area"] for p in g))[rank - 1]
    area = sum(p["area"] for p in pieces) / 100
    cell = pieces[0]["stride"] * pieces[0]["um"]
    x0, y0 = min(p["x_mm"] for p in pieces), min(p["y_mm"] for p in pieces)
    width = max(round((p["x_mm"] - x0) / cell) + p["shape"][1] for p in pieces)
    height = max(round((p["y_mm"] - y0) / cell) + p["shape"][0] for p in pieces)
    mask = np.zeros((height, width), dtype=bool)
    for p in pieces:
        z = tifffile.imread(base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}" / "z.tif")
        held = (z > 0) & own_cells(p)
        held = held[::-1] if p["flip_rows"] else held
        held = held[:, ::-1] if p["flip_cols"] else held
        r, c = round((p["y_mm"] - y0) / cell), round((p["x_mm"] - x0) / cell)
        mask[r : r + held.shape[0], c : c + held.shape[1]] |= held
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(mask, 2 + SEAM_CELLS), iterations=SEAM_CELLS))
    labels, count = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, count + 1)) * cell**2 / 100
    mask = np.isin(labels, 1 + np.flatnonzero(sizes >= SMALLEST_CM2))
    lines = []
    for contour in measure.find_contours(mask.astype(np.uint8), 0.5):
        contour = measure.approximate_polygon(contour, tolerance=0.5)
        if len(contour) >= 4:
            lines.append(
                [
                    [round(x0 + (col - 2 - SEAM_CELLS) * cell, 3), round(y0 + (row - 2 - SEAM_CELLS) * cell, 3)]
                    for row, col in contour
                ]
            )
    name = f"largest mosaic · {area:.0f} cm²" if rank == 1 else f"mosaic {rank} · {area:.0f} cm²"
    return name, lines, mask.sum() * cell**2 / 100


def main(page, sample, volume, rank=1):
    name, lines, drawn = outline(sample, volume, rank)
    refresh(SHEETS, extra={page: {"outlines": [{"name": name, "points": line} for line in lines]}})
    print(
        json.dumps(
            {
                "page": page,
                "outline": name,
                "traced_cm2": round(drawn, 1),
                "contours": len(lines),
                "points": sum(len(line) for line in lines),
            }
        )
    )


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 1)
