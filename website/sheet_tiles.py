"""One continuous sheet out of every piece's raw scan images and the layout, kept as tiles of a tile store."""

import argparse
import copy
import io
import json
import math
import multiprocessing
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from sampling.sheet_images import MARGIN_CELLS, largest_mosaic, mosaics_of
from storage.data_roots import BROWSER_ASSETS, FACES, PRODUCTION_FACES, ROUTE_OUTPUTS, SHEETS, stage_folder
from storage.tile_store import TileWriter

TILE = 512
STRIP = 8
PARTING_MM = 0.25
SAMPLE_PIECES = 64
SHOWN_PERCENTILES = (1.0, 99.0)
COARSE_QUALITY = 90
PROCESSES = int(os.environ.get("HERCULEX_COMPOSE_PROCESSES", "4"))
SCAN_UNDER_TAUGHT = "taught-scan"
WEAKEST_INK, SCAN_UNDER_INK = 160, 0.3
Image.MAX_IMAGE_PIXELS = None


def upright(image, p):
    if p["flip_rows"]:
        image = image[::-1]
    if p["flip_cols"]:
        image = image[:, ::-1]
    return image


def written_on_far_side(folder, p, record="side"):
    path = folder / p["unit"] / f"piece-{p['piece']:04d}-{record}.json"
    try:
        return json.loads(path.read_text()).get("written") == "far"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def seen_from_written_face(pieces, folder, record="side"):
    """Pieces as they read from the face that carries the writing."""
    for mosaic in mosaics_of(pieces):
        far = [p for p in mosaic if written_on_far_side(folder, p, record)]
        if sum(p["area"] for p in far) * 2 > sum(p["area"] for p in mosaic):
            left, right = min(p["x_mm"] for p in mosaic), max(p["x_mm"] + p["width_mm"] for p in mosaic)
            for p in mosaic:
                p["x_mm"], p["flip_cols"] = left + right - p["x_mm"] - p["width_mm"], not p["flip_cols"]
        else:
            for p in far:
                p["flip_cols"] = not p["flip_cols"]
    return pieces


def piece_png(folder, p, layer):
    name = f"piece-{p['piece']:04d}" + ("" if layer in ("sheet", SCAN_UNDER_TAUGHT) else f"-{layer}")
    return folder / p["unit"] / f"{name}.png"


def piece_layer(folder, p, layer):
    """A piece's (image, seen, pixel in mm) for a layer, upright."""
    png = piece_png(folder, p, layer)
    mask = piece_png(folder, p, "sheet") if layer in ("ink", "taught") else png
    if not png.is_file() or (mask != png and not mask.is_file()):
        return None
    pixels = np.asarray(Image.open(png))
    seen = pixels if mask == png else np.asarray(Image.open(mask))
    return upright(pixels, p), upright(seen > 0, p), p["um"]


def parted(covered, owner, width):
    """Clears a line along every place where two mosaics touch."""
    meets = np.zeros(owner.shape, dtype=np.uint8)
    meets[1:] |= (owner[1:] != owner[:-1]) & (owner[1:] > 0) & (owner[:-1] > 0)
    meets[:, 1:] |= (owner[:, 1:] != owner[:, :-1]) & (owner[:, 1:] > 0) & (owner[:, :-1] > 0)
    covered &= ni.maximum_filter1d(ni.maximum_filter1d(meets, width, axis=0), width, axis=1) == 0


def shown_range(pieces, folder, layer):
    """The grey levels a sheet's one linear stretch runs between."""
    if layer in ("ink", "taught"):
        return 0.0, 255.0
    values = []
    for p in sorted(pieces, key=lambda p: -p["area"]):
        found = piece_layer(folder, p, layer)
        if found is not None:
            values.append(found[0][found[1]][::97])
        if len(values) == SAMPLE_PIECES:
            break
    if not any(part.size for part in values):
        return 0.0, 255.0
    low, high = np.percentile(np.concatenate(values), SHOWN_PERCENTILES)
    return (float(low), float(high)) if low < high else (0.0, 255.0)


def encoded(pixels, level):
    """A tile as stored: the sheet's own values without loss at level 0, a preview above it."""
    out = io.BytesIO()
    if level:
        Image.fromarray(pixels).save(out, format="JPEG", quality=COARSE_QUALITY)
    else:
        Image.fromarray(pixels).save(out, format="PNG", compress_level=6)
    return out.getvalue()


def shown(pixels, layer):
    """Stored values as the viewer shows them: one linear stretch between the layer's two grey levels."""
    if not layer.get("raw"):
        return pixels
    return np.clip(
        (pixels.astype(np.float32) - layer["low"]) * (255.0 / (layer["high"] - layer["low"])), 0, 255
    ).astype(np.uint8)


def paint_strip(job):
    """(column, row, tile) of every tile in tile columns c0 onwards, raw values, 0 where there is no scan."""
    c0, pieces, folder, layer, pixel_mm, rows, columns = job
    tile_mm = TILE * pixel_mm
    x0 = c0 * tile_mm
    H, W = rows * TILE, STRIP * TILE
    raw = np.zeros((H, W), dtype=np.uint8)
    covered = np.zeros((H, W), dtype=bool)
    mosaics, held = {}, []
    for p in pieces:
        found = piece_layer(folder, p, layer)
        if found is None:
            continue
        image, seen, pixel = found
        ratio = pixel / pixel_mm
        if abs(ratio - 1) > 0.01:
            size = (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio)))
            image = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize(size, Image.LANCZOS))
            seen = (
                np.asarray(
                    Image.fromarray(np.ascontiguousarray(seen).astype(np.uint8) * 255).resize(size, Image.NEAREST)
                )
                > 0
            )
        margin = round(MARGIN_CELLS * p["stride"] * p["um"] / pixel_mm)
        px, py = round((p["x_mm"] - x0) / pixel_mm) - margin, round(p["y_mm"] / pixel_mm) - margin
        a0, b0, a1, b1 = max(py, 0), max(px, 0), min(py + image.shape[0], H), min(px + image.shape[1], W)
        if a1 <= a0 or b1 <= b0:
            continue
        part = (slice(a0 - py, a1 - py), slice(b0 - px, b1 - px))
        top, bottom, left, right = p.get("overlap", (0, 0, 0, 0))
        top, bottom = (bottom, top) if p["flip_rows"] else (top, bottom)
        left, right = (right, left) if p["flip_cols"] else (left, right)
        cells = p["stride"] * p["um"] / pixel_mm
        own = np.zeros(image.shape, dtype=bool)
        own[
            margin + round(top * cells) : image.shape[0] - margin - round(bottom * cells),
            margin + round(left * cells) : image.shape[1] - margin - round(right * cells),
        ] = True
        mosaic = mosaics.setdefault((p.get("group"), round(p.get("moved_from_joint_mm", 0.0))), len(mosaics) + 1)
        held.append(((slice(a0, a1), slice(b0, b1)), image[part], seen[part], own[part], mosaic))
    owner = np.zeros((H, W), dtype=np.int32) if len(mosaics) > 1 else None
    for margins in (False, True):
        for here, image, seen, own, mosaic in held:
            free = seen & ~covered[here] & (own | margins)
            raw[here][free] = image[free]
            covered[here] |= free
            if owner is not None:
                owner[here][free] = mosaic
    if owner is not None:
        parted(covered, owner, max(3, round(PARTING_MM / pixel_mm)))
    raw[~covered] = 0
    tiles = []
    for c in range(min(STRIP, columns - c0)):
        for r in range(rows):
            cut = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            if covered[cut].any():
                tiles.append((c0 + c, r, encoded(raw[cut], 0)))
    return tiles


def shrink_column(job):
    """(column, row, tile) of one column of a coarser level, from the tiles of the level below."""
    c, rows, level, pack, below = job
    tiles = []
    with open(pack, "rb") as source:
        for r in range(*rows):
            canvas = Image.new("L", (2 * TILE, 2 * TILE), 0)
            seen = False
            for a in (0, 1):
                for b in (0, 1):
                    found = below.get((2 * c + a, 2 * r + b))
                    if found:
                        source.seek(found[0])
                        canvas.paste(Image.open(io.BytesIO(source.read(found[1]))), (a * TILE, b * TILE))
                        seen = True
            if seen:
                tiles.append((c, r, encoded(np.asarray(canvas.resize((TILE, TILE), Image.LANCZOS)), level)))
    return tiles


def compose(pieces, folder, writer, path, layer, pixel_mm, pool):
    """Every tile of one layer at every zoom, written under a path of a sheet's tile store."""
    tile_mm = TILE * pixel_mm
    rows = int(math.ceil((max(p["y_mm"] + p["height_mm"] for p in pieces) + 2) / tile_mm))
    columns = int(math.ceil((max(p["x_mm"] + p["width_mm"] for p in pieces) + 2) / tile_mm))
    first = [max(0, int((min(p[side] for p in pieces) - 2) // tile_mm)) for side in ("x_mm", "y_mm")]
    spans = [(*first, columns, rows)]
    while max(spans[-1][2:]) > 1 and len(spans) < 12:
        c0, r0, c1, r1 = spans[-1]
        spans.append((c0 // 2, r0 // 2, -(-c1 // 2), -(-r1 // 2)))
    low, high = shown_range(pieces, folder, layer)
    grid, origin = [(c1 - c0, r1 - r0) for c0, r0, c1, r1 in spans], [span[:2] for span in spans]
    writer.layer(path, grid, origin=origin, raw=True, low=low, high=high)
    jobs = []
    order = sorted(pieces, key=lambda p: -p["area"])
    for c0 in range(0, columns, STRIP):
        x0, x1 = c0 * tile_mm, (c0 + STRIP) * tile_mm
        members = [p for p in order if p["x_mm"] < x1 and p["x_mm"] + p["width_mm"] > x0]
        if members:
            jobs.append((c0, members, folder, layer, pixel_mm, rows, columns))
    start = time.time()
    for k, tiles in enumerate(pool.imap_unordered(paint_strip, jobs)):
        for c, r, blob in tiles:
            writer.add(path, 0, c, r, blob)
        if k % 20 == 19:
            print(path, k + 1, "of", len(jobs), "strips", round(time.time() - start), "s", flush=True)
    for level in range(1, len(grid)):
        writer.flush()
        c0, r0, c1, r1 = spans[level]
        below = {
            c: {
                (a, b): writer.place(path, level - 1, a, b)
                for a in (2 * c, 2 * c + 1)
                for b in range(2 * r0, 2 * r1)
                if writer.place(path, level - 1, a, b)
            }
            for c in range(c0, c1)
        }
        jobs = [(c, (r0, r1), level, writer.path, below[c]) for c in below if below[c]]
        for tiles in pool.imap_unordered(shrink_column, jobs):
            for c, r, blob in tiles:
                writer.add(path, level, c, r, blob)
    print(path, "tiles done", len(grid), "levels", round(time.time() - start), "s", flush=True)
    return {
        "path": path,
        "mm": pixel_mm,
        "levels": len(grid),
        "low": low,
        "high": high,
        "stamp": int(time.time()),
        "raw": True,
    }


def layer_name(layer, pixel_mm):
    return {
        "ink": "ink",
        "taught": "ink",
        SCAN_UNDER_TAUGHT: "scan under the ink",
        "sheet": "brightest across the sheet",
    }.get(layer) or f"{int(layer.removeprefix('layer')) * pixel_mm * 1000:+.0f} µm"


def page(folder, title, layers, area, back, scan, boxes=(), places=(), shelves=(), suggested=(), tint=""):
    """The viewer for a sheet folder, fitted to an area in mm."""
    meta = {
        "title": title,
        "layers": layers,
        "tile": TILE,
        "area": area,
        "back": back,
        "scan": scan,
        "boxes": list(boxes),
        "places": list(places),
        "shelves": list(shelves),
        "suggested": list(suggested),
        "tint": tint,
    }
    (folder / "index.html").write_text(
        (BROWSER_ASSETS / "sheet_viewer.html")
        .read_text()
        .replace("__TITLE__", title)
        .replace("__META__", json.dumps(meta))
        .replace("__FLOOR__", str(WEAKEST_INK))
        .replace("__SCAN__", str(SCAN_UNDER_INK))
    )


def run(
    sample,
    volume,
    pixel_mm,
    name,
    layers=("ink", "layer0", "sheet"),
    largest=False,
    units=None,
    model="",
    *,
    ink_name="",
    written_face=False,
):
    """Tiles of the named layers of a sheet read into FACES, or of its largest continuous area alone, or of some units."""
    base = ROUTE_OUTPUTS / sample / volume
    folder = stage_folder(base, "faces") / FACES
    layout = json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]
    layout = [p for p in layout if not units or p["unit"] in units]
    taught = seen_from_written_face(copy.deepcopy(layout), folder, "taught-side")
    pieces = taught if written_face else seen_from_written_face(copy.deepcopy(layout), folder)
    placement = SHEETS / name if FACES != PRODUCTION_FACES or units else stage_folder(base, "sheet")
    placement.mkdir(parents=True, exist_ok=True)
    for file, group in (("placed.json", pieces), ("placed-taught.json", taught)):
        (placement / file).write_text(json.dumps({"pieces": group}))
    pieces, taught = (largest_mosaic(group) if largest else group for group in (pieces, taught))
    available = []
    for layer in layers:
        group = taught if layer.startswith("taught") else pieces
        if any(piece_png(folder, p, layer).is_file() for p in group):
            available.append((layer, group))
    if not available:
        raise ValueError(f"no source images for {name}: {', '.join(layers)}")
    out = SHEETS / name
    out.mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(PROCESSES) as pool, TileWriter(out) as writer:
        done = [
            compose(group, folder, writer, layer, layer, pixel_mm, pool)
            | {"name": ink_name if layer == "taught" and ink_name else layer_name(layer, pixel_mm)}
            | ({"model": model} if model and layer == "taught" else {})
            for layer, group in available
        ]
    kept = json.loads((out / "layers.json").read_text()) if (out / "layers.json").exists() else []
    written = {layer for layer, _ in available}
    (out / "layers.json").write_text(json.dumps(done + [layer for layer in kept if layer["path"] not in written]))
    print("composed", name, ", ".join(layer for layer, _ in available), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="website.sheet_tiles", description=__doc__)
    parser.add_argument("sample")
    parser.add_argument("volume")
    parser.add_argument("pixel_mm", type=float)
    parser.add_argument("name")
    parser.add_argument("layers", nargs="?", default="ink,layer0,sheet")
    parser.add_argument("largest", nargs="?", choices=("largest",))
    parser.add_argument("--units", help="only these units, comma separated")
    parser.add_argument("--model", default="", help="the model that read the taught ink, named on its page")
    parser.add_argument("--ink-name", default="", help="display name for the taught ink layer")
    parser.add_argument("--written-face", action="store_true", help="place all layers on the ink map's face")
    args = parser.parse_args()
    run(
        args.sample,
        args.volume,
        args.pixel_mm,
        args.name,
        tuple(args.layers.split(",")),
        bool(args.largest),
        set(args.units.split(",")) if args.units else None,
        args.model,
        ink_name=args.ink_name,
        written_face=args.written_face,
    )
