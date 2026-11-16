"""A published scroll page drawn as one picture: its whole unrolled sheet, shelf under shelf, as the viewer shows it
zoomed all the way out, the ink in the viewer's colours over the scan.

./run.sh website.overview PAGE OUT [--width 2400] [--floor 24] [--scan 0.6] [--tint purple|yellow] [--no-ink] [--loose]

PAGE is a folder under website/sheets (e.g. sheet-PHerc0125-113keV-level); OUT a .png or .jpg. The picture is cut from
the page's own tile pyramid at the level nearest the width asked for, so nothing is recomposed. Shelves are packed
close (their empty rows dropped) unless --loose keeps the viewer's spacing.
"""

import argparse
import json
import re

import numpy as np
from PIL import Image

from storage.data_roots import SHEETS
from storage.tile_store import TileStore

TILE = 512
TINTS = {
    "purple": (np.array([0.33, 0.10, 0.62]), np.array([0.82, 0.55, 1.0])),
    "yellow": (np.array([0.69, 0.44, 0.06]), np.array([1.0, 0.957, 0.376])),
}


def meta_of(page):
    html = (SHEETS / page / "index.html").read_text()
    return json.loads(re.search(r"const M=(.*?),TOP=", html)[1])


def cut(store, layer, level, mm, x0, y0, x1, y1):
    """The stored values of a layer over a box of the sheet (mm), at one pyramid level."""
    step = mm * 2**level
    c0, r0, c1, r1 = int(x0 / step), int(y0 / step), int(np.ceil(x1 / step)), int(np.ceil(y1 / step))
    out = np.zeros((max(r1 - r0, 1), max(c1 - c0, 1)), np.uint8)
    for r in range(r0 // TILE, (r1 - 1) // TILE + 1):
        for c in range(c0 // TILE, (c1 - 1) // TILE + 1):
            tile = store.image(layer, level, c, r)
            if tile is None:
                continue
            ty, tx = r * TILE, c * TILE
            ys, xs = slice(max(r0, ty), min(r1, ty + tile.shape[0])), slice(max(c0, tx), min(c1, tx + tile.shape[1]))
            if ys.start < ys.stop and xs.start < xs.stop:
                out[ys.start - r0 : ys.stop - r0, xs.start - c0 : xs.stop - c0] = tile[
                    ys.start - ty : ys.stop - ty, xs.start - tx : xs.stop - tx
                ]
    return out


def drawn(scan, ink, scan_layer, floor, brightness, tint):
    """The viewer's drawing: the scan stretched and dimmed, the ink above the floor in the tint's ramp (alpha 3t)."""
    v = np.clip((scan.astype(np.float32) - scan_layer["low"]) / (scan_layer["high"] - scan_layer["low"]), 0, 1)
    base = np.repeat((v * brightness)[..., None], 3, -1)
    if ink is None:
        return base
    w = floor / 255
    t = np.clip((ink.astype(np.float32) / 255 - w) / (1 - w), 0, 1)
    dark, light = TINTS[tint]
    colour = dark + (light - dark) * t[..., None]
    alpha = np.minimum(1, 3 * t)[..., None]
    return base * (1 - alpha) + colour * alpha


def overview(page, width=2400, floor=24, brightness=0.6, tint="purple", ink=True, compact=True, gap=10):
    meta = meta_of(page)
    layers = {entry["path"]: entry for entry in meta["layers"]}
    scan_layer = layers["sheet"]
    ink_layer = next((k for k in ("taught", "ink") if k in layers), None) if ink else None
    store = TileStore(SHEETS / page)
    shelves = meta["shelves"] or [{"x": 0, "y": 0, **{k: meta["area"][k] for k in ("w", "h")}}]
    mm, widest = scan_layer["mm"], max(s["w"] for s in shelves)
    level = max(0, min(scan_layer["levels"] - 1, int(np.floor(np.log2(widest / mm / width)))))
    step = mm * 2**level
    height = int(np.ceil((shelves[-1]["y"] + shelves[-1]["h"]) / step))
    canvas = np.zeros((height, int(np.ceil(widest / step)), 3), np.float32)
    bands = []
    for s in shelves:
        box = (s["x"], 0.0, s["x"] + s["w"], s["h"])
        scan = cut(store, "sheet", level, mm, *box)
        read = cut(store, ink_layer, level, mm, *box) if ink_layer else None
        image = drawn(scan, read, scan_layer, floor, brightness, tint)
        y = int(round(s["y"] / step))
        h, w = min(image.shape[0], canvas.shape[0] - y), min(image.shape[1], canvas.shape[1])
        canvas[y : y + h, :w] = image[:h, :w]
        filled = np.flatnonzero(image[:h, :w].max(axis=(1, 2)) > 0.02)
        if len(filled):
            bands.append((y + filled[0], y + filled[-1] + 1))
    if compact and bands:
        rows = [canvas[a:b] for a, b in bands]
        spacer = np.zeros((max(1, round(gap * canvas.shape[1] / width)), canvas.shape[1], 3), np.float32)
        canvas = np.concatenate([part for band in rows for part in (band, spacer)][:-1])
    picture = Image.fromarray((canvas * 255).astype(np.uint8))
    scale = width / picture.width
    return picture.resize((width, max(1, round(picture.height * scale))), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--floor", type=float, default=24)
    parser.add_argument("--scan", type=float, default=0.6)
    parser.add_argument("--tint", choices=tuple(TINTS), default="purple")
    parser.add_argument("--no-ink", action="store_true")
    parser.add_argument("--loose", action="store_true")
    args = parser.parse_args()
    picture = overview(args.page, args.width, args.floor, args.scan, args.tint, not args.no_ink, not args.loose)
    picture.save(args.out, quality=90) if args.out.endswith((".jpg", ".jpeg")) else picture.save(args.out)
    print(args.out, picture.size, flush=True)


if __name__ == "__main__":
    main()
