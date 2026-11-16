"""A laid-out sheet as one tifxyz grid for VC3D."""

import json
import sys
import uuid

import numpy as np
import tifffile
from scipy import ndimage as ni

from sampling import column_sampling as sampler
from storage.data_roots import ROUTE_OUTPUTS, SHEETS, stage_folder


def run(sample, volume, name):
    base = ROUTE_OUTPUTS / sample / volume
    pieces = json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]
    side = pieces[0]["stride"] * pieces[0]["um"]
    W = int(max(p["x_mm"] + p["width_mm"] for p in pieces) / side) + 4
    H = int(max(p["y_mm"] + p["height_mm"] for p in pieces) / side) + 4
    grid = np.full((3, H, W), -1.0, dtype=np.float32)
    for p in sorted(pieces, key=lambda p: p["area"]):
        xyz, _meta = sampler.load_surface(base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}")
        if p["flip_rows"]:
            xyz = xyz[:, ::-1]
        if p["flip_cols"]:
            xyz = xyz[:, :, ::-1]
        solid = ni.binary_erosion(xyz[0] >= 0)
        y, x = int(round(p["y_mm"] / side)), int(round(p["x_mm"] / side))
        h, w = min(solid.shape[0], H - y), min(solid.shape[1], W - x)
        if h <= 0 or w <= 0:
            continue
        here = (slice(None), slice(y, y + h), slice(x, x + w))
        grid[here] = np.where(solid[None, :h, :w], xyz[:, :h, :w], grid[here])
    out = SHEETS / name / "tifxyz" / f"{sample}-{volume}-sheet"
    out.mkdir(parents=True, exist_ok=True)
    for axis, letter in enumerate("xyz"):
        tifffile.imwrite(out / f"{letter}.tif", grid[axis])
    known = grid[:, grid[0] >= 0]
    stride = pieces[0]["stride"]
    (out / "meta.json").write_text(
        json.dumps(
            {
                "sample": sample,
                "scan": volume,
                "voxel_um": pieces[0]["um"] * 1000,
                "format": "tifxyz",
                "type": "seg",
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, str(out))),
                "bbox": [known.min(axis=1).tolist(), known.max(axis=1).tolist()],
                "scale": [1 / stride, 1 / stride],
                "stride": stride,
                "area_mm2": float((grid[0] >= 0).sum()) * side**2,
                "width_mm": W * side,
                "height_mm": H * side,
            },
            indent=1,
        )
    )
    print(sample, "tifxyz", grid.shape, round(float((grid[0] >= 0).sum()) * side**2 / 100), "cm2", flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:4])
