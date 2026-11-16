"""Row power of every surveyed piece on its forward map and, as control, its reversed map."""

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from calibration.survey.scroll_survey import (
    REPLICATED,
    ROW_SAMPLE,
    SURVEYED,
    pieces_of,
)
from calibration.text import line_spectrum as spectra

ROW_WINDOW_MM = (16.0, 10.0)
NO_RIM_MM, NO_RIM_WINDOW_MM = 1.0, (14.0, 10.0)
ROW_PERIODS = (spectra.PERIODS >= 4.0) & (spectra.PERIODS <= 7.0)
RECORD = SURVEYED / "row-power.json"
Image.MAX_IMAGE_PIXELS = None


def row_ratio(image, voxel_mm, window_mm=ROW_WINDOW_MM, rim_mm=0.0):
    """Across over along at row spacings of one map, with the windows it rests on."""
    spectra.WINDOW_MM, spectra.UM = window_mm, voxel_mm
    seen = image > 0
    if rim_mm:
        seen = ni.distance_transform_edt(seen) * voxel_mm >= rim_mm
    values, cover = spectra.cells((image.astype(np.float32) - 1) / 254, seen)
    found = spectra.spectrum(values, cover)
    if not found["across"] or not found["along"]:
        return None
    across, along = (np.mean(found[name], axis=0) for name in ("across", "along"))
    return {
        "ratio": float(across[ROW_PERIODS].mean() / along[ROW_PERIODS].mean()),
        "windows": len(found["across"]),
        "across": across.round(4).tolist(),
        "along": along.round(4).tolist(),
    }


def stage_rows(folder, base):
    """Each piece of a stage that holds a window, by name."""
    rows = {}
    for path in sorted(folder.glob("piece-*.png")):
        voxel_mm = json.loads((base / path.stem / "meta.json").read_text())["voxel_um"] / 1000
        image = np.asarray(Image.open(path))
        back = path.with_name(f"reversed-{path.name}")
        reversed_image = np.asarray(Image.open(back)) if back.exists() else None
        found = row_ratio(image, voxel_mm) or {}
        no_rim = row_ratio(image, voxel_mm, NO_RIM_WINDOW_MM, NO_RIM_MM)
        if found and reversed_image is not None:
            found["reversed"] = row_ratio(reversed_image, voxel_mm)
        if no_rim:
            found["no_rim"] = no_rim
            if reversed_image is not None:
                found["no_rim"]["reversed"] = row_ratio(reversed_image, voxel_mm, NO_RIM_WINDOW_MM, NO_RIM_MM)
        if found:
            rows[path.stem] = found
    return rows


def run(items):
    """Bands as sample/volume/band, each scroll's stages kept as soon as they are done."""
    kept = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    for item in items:
        sample, volume, band = item.split("/")
        base = pieces_of(sample, volume, band)
        kept[sample] = {
            "first": stage_rows(SURVEYED / sample, base),
            "replicate": stage_rows(REPLICATED / sample, base),
            "row_sample": stage_rows(ROW_SAMPLE / sample, base),
        }
        RECORD.write_text(json.dumps(kept, indent=0))
        ratios = [r["ratio"] for stage in kept[sample].values() for r in stage.values() if "ratio" in r]
        print(sample, len(ratios), "pieces", round(float(np.median(ratios)), 3) if ratios else None, flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
