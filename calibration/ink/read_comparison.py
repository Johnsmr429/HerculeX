"""Two reads of the same pieces side by side: the face each took, sure ink on the sheet, ink too wide to be a stroke, letter-like windows and the scrambled control."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ink.datasets.letter_spots import CELL, SURE_SHOWN, on_cells, strokes_of, windows_of
from storage.data_roots import ROUTE_OUTPUTS, calibration_output, stage_folder

OUT = calibration_output("read_comparison")
WORKERS = 6
Image.MAX_IMAGE_PIXELS = None


def piece_of(job):
    """One piece of one read: its face, sure and too-wide shares of the sheet, letter-like windows, scrambled sure share."""
    faces, unit, piece, um = job
    out = faces / unit / f"piece-{piece:04d}"
    side = json.loads(Path(f"{out}-taught-side.json").read_text())
    with Image.open(f"{out}-taught.png") as ink, Image.open(f"{out}.png") as scan:
        sheet = on_cells(np.asarray(scan) > 0)
        sure = on_cells(np.asarray(ink) >= SURE_SHOWN) & sheet
    _, wide = strokes_of(sure, CELL * um / 1000) if sure.any() else (None, sure)
    return {
        "read": faces.name,
        "unit": unit,
        "piece": piece,
        "written": side.get("written"),
        "sheet": int(sheet.sum()),
        "sure": int(sure.sum()),
        "wide": int(wide.sum()),
        "letter_windows": len(windows_of(job)),
        "scrambled": side.get("scrambled"),
    }


def summary_of(rows):
    """One read's pieces summed."""
    sheet, sure = sum(r["sheet"] for r in rows), sum(r["sure"] for r in rows)
    scrambled = [r["scrambled"] for r in rows if r["scrambled"] is not None]
    return {
        "pieces": len(rows),
        "far_face": sum(r["written"] == "far" for r in rows),
        "sure_share": round(sure / max(sheet, 1), 4),
        "wide_share_of_sure": round(sum(r["wide"] for r in rows) / max(sure, 1), 4),
        "letter_windows": sum(r["letter_windows"] for r in rows),
        "pieces_with_letter_windows": sum(r["letter_windows"] > 0 for r in rows),
        "scrambled_sure": round(float(np.mean(scrambled)), 4) if scrambled else None,
    }


def run(sample, volume, first, second, units, out=None):
    """Both reads over the pieces they both hold, piece by piece, by unit and summed."""
    base = ROUTE_OUTPUTS / sample / volume
    units = set(units.split(","))
    folders = {name: stage_folder(base, "faces") / name for name in (first, second)}
    jobs = []
    for path in sorted(folders[second].glob("*/piece-*-taught-side.json")):
        unit, piece = path.parent.name, int(path.name.split("-")[1])
        if unit in units and (folders[first] / unit / path.name).exists():
            um = json.loads((base / "pieces" / unit / f"piece-{piece:04d}" / "meta.json").read_text())["voxel_um"]
            jobs += [(folders[name], unit, piece, um) for name in (first, second)]
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(piece_of, jobs, chunksize=8))
    found, by_unit = {}, {}
    for k, name in enumerate((first, second)):
        mine = rows[k::2]
        found[name] = summary_of(mine)
        by_unit[name] = {u: summary_of([r for r in mine if r["unit"] == u]) for u in sorted({r["unit"] for r in mine})}
        print(name, json.dumps(found[name]), flush=True)
    out = Path(out) if out else OUT / sample / f"{first}-against-{second}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": found, "units": by_unit, "pieces": rows}, indent=1))


if __name__ == "__main__":
    run(*sys.argv[1:])
