import argparse
import json
import os
import re

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_tif(path):
    return np.asarray(Image.open(path), dtype=np.float32)


def load_atlas(folder):
    meta = json.load(open(os.path.join(folder, "meta.json")))
    shape, b = meta["shape"], meta["brick"][0]
    offsets = np.load(os.path.join(folder, "offsets.npy"))
    atlas = np.zeros(shape, dtype=np.uint8)
    raw = np.memmap(os.path.join(folder, "bricks.bin"), dtype=np.uint8, mode="r")
    for (bz, by, bx), off in np.ndenumerate(offsets):
        if off < 0:
            continue
        block = np.asarray(raw[off : off + b**3]).reshape(b, b, b)
        z0, y0, x0 = bz * b, by * b, bx * b
        z1, y1, x1 = min(z0 + b, shape[0]), min(y0 + b, shape[1]), min(x0 + b, shape[2])
        atlas[z0:z1, y0:y1, x0:x1] = block[: z1 - z0, : y1 - y0, : x1 - x0]
    level = re.search(r"L(\d+)", os.path.basename(os.path.normpath(folder)))
    return atlas, 2.0 ** int(level.group(1)) if level else 8.0


def sample(atlas, x, y, z, scale):
    p = np.stack([z, y, x], 0) / scale
    lo = np.floor(p).astype(np.int64)
    f = p - lo
    out = np.zeros(x.shape, dtype=np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                iz = np.clip(lo[0] + dz, 0, atlas.shape[0] - 1)
                iy = np.clip(lo[1] + dy, 0, atlas.shape[1] - 1)
                ix = np.clip(lo[2] + dx, 0, atlas.shape[2] - 1)
                w = (f[0] if dz else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dx else 1 - f[2])
                out += w * atlas[iz, iy, ix]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--overview-width", type=int, default=2400)
    args = ap.parse_args()
    atlas, scale = load_atlas(args.atlas)
    lay = json.load(open(args.layout))["pieces"]
    mm, stride = lay[0]["um"], lay[0]["stride"]
    x0 = {p["uuid"]: int(round(p["x_mm"] / mm / stride)) for p in lay}
    y0 = {p["uuid"]: int(round(p["y_mm"] / mm / stride)) for p in lay}
    left = min(x0.values())
    top = min(y0.values())
    w = max(x0[p["uuid"]] + p["shape"][1] for p in lay) - left
    h = max(y0[p["uuid"]] + p["shape"][0] for p in lay) - top
    canvas = np.zeros((h, w), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)
    for p in sorted(lay, key=lambda q: q["area"]):
        d = os.path.join(args.pieces, p["uuid"])
        known = np.isfinite(read_tif(os.path.join(d, "phi.tif")))
        grids = [read_tif(os.path.join(d, f"{c}.tif")) for c in "xyz"]
        if p["col_rev"]:
            known, grids = known[:, ::-1], [g[:, ::-1] for g in grids]
        if p["row_rev"]:
            known, grids = known[::-1, :], [g[::-1, :] for g in grids]
        rr, cc = np.nonzero(known)
        if len(rr) == 0:
            continue
        v = sample(atlas, grids[0][rr, cc], grids[1][rr, cc], grids[2][rr, cc], scale)
        gy, gx = rr + y0[p["uuid"]] - top, cc + x0[p["uuid"]] - left
        canvas[gy, gx] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
        filled[gy, gx] = True
    hist = np.bincount(canvas[filled], minlength=256).cumsum()
    lo, hi = np.searchsorted(hist, hist[-1] * 0.01), np.searchsorted(hist, hist[-1] * 0.995)
    lut = np.clip((np.arange(256) - lo) / max(hi - lo, 1) * 235 + 20, 20, 255).astype(np.uint8)
    img = np.where(filled, lut[canvas], 0).astype(np.uint8)
    del canvas
    if args.rotate180:
        img = img[::-1, ::-1]
    row_w = int(np.ceil(w / args.rows))
    gap = max(8, h // 12)
    folded = np.zeros((args.rows * h + (args.rows - 1) * gap, row_w), dtype=np.uint8)
    for k in range(args.rows):
        part = img[:, k * row_w : (k + 1) * row_w]
        folded[k * (h + gap) : k * (h + gap) + h, : part.shape[1]] = part
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray(folded).save(args.out + ".png")
    height = max(1, int(folded.shape[0] * args.overview_width / row_w))
    Image.fromarray(folded).resize((args.overview_width, height), Image.LANCZOS).save(
        args.out + "_overview.jpg", quality=90
    )
    cell_cm2 = (stride * mm) ** 2 / 100
    print(
        f"strip {w * stride * mm / 1000:.2f} m x {h * stride * mm:.0f} mm, {len(lay)} pieces, "
        f"{filled.sum() * cell_cm2:.0f} cm2 drawn, {args.out}.png {folded.shape[1]}x{folded.shape[0]} "
        f"at {stride * mm * 1000:.0f} um per pixel"
    )


if __name__ == "__main__":
    main()
