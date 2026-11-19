"""Labelled ink cases: every labelled segment in every scan that holds it, and infrared fragments."""

import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request

import imagecodecs
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ni

from scans.catalog.scan_catalog import BUCKET, load, segments
from scans.transfer.fetch_fragment import fetch_fragment
from scans.transfer.parallel_fetch import fetch_all
from scans.transfer.sharded_labels import read_cells
from storage.brick_store import BrickStore
from storage.data_roots import INFRARED_FRAGMENTS, LABELLED_CASES

HOLDOUT_EVERY = 4
MARGIN_BRICKS = 1
MAX_BRICKS = 3000
MAX_VOXEL_UM = 10.0
CELL_UM = 9.4


def voxel_of(path):
    return float(re.search(r"(?:^|/)(\d+\.?\d*)um", path).group(1))


def partition(sample, segment):
    digest = hashlib.sha256(f"{sample}/{segment}".encode()).digest()
    return "holdout" if digest[0] % HOLDOUT_EVERY == 0 else "development"


def plan_cases(rows):
    """One case per labelled segment and scan."""
    cases = []
    for row in rows:
        labels = row["products"].get("ink-labels", [])
        if not labels:
            continue
        source = min(labels, key=lambda p: voxel_of(p.split("/ink-labels/")[1]))
        for layers in row["products"].get("layers-zarr", []):
            name = layers.rstrip("/").rsplit("/", 1)[1]
            voxel = voxel_of(name)
            if voxel > MAX_VOXEL_UM:
                continue
            own = [p for p in labels if name.split("-volume-")[1][:14] in p]
            cases.append(
                {
                    "case_id": f"{row['sample']}-{row['segment']}-{name.split('-volume-')[0]}",
                    "sample": row["sample"],
                    "segment": row["segment"],
                    "scan": name.removesuffix(".zarr"),
                    "layers_path": layers,
                    "labels_path": own[0] if own else source,
                    "labels_resampled": not own,
                    "truth": "community-labels",
                    "partition": partition(row["sample"], row["segment"]),
                }
            )
    return sorted(cases, key=lambda c: c["case_id"])


def cell_level(path):
    """The pyramid level whose lateral step is nearest the surface cell, with its depth and lateral steps in um."""
    attributes = json.load(urllib.request.urlopen(f"{BUCKET}/{path.rstrip('/')}/.zattrs", timeout=60))
    steps = [d["coordinateTransformations"][0]["scale"] for d in attributes["multiscales"][0]["datasets"]]
    level = min(range(len(steps)), key=lambda k: abs(math.log(steps[k][1] / CELL_UM)))
    return level, steps[level][0], steps[level][1]


def label_level(root, shape):
    """The label pyramid level nearest the target grid."""
    best, gap = 0, None
    for level in range(6):
        try:
            meta = json.load(urllib.request.urlopen(f"{root}/{level}/zarr.json", timeout=60))
        except urllib.error.HTTPError:
            break
        distance = abs(math.log(meta["shape"][0] / shape[0]))
        if gap is None or distance < gap:
            best, gap = level, distance
    return best


def resampled_cells(cells, source, shape, size=1024):
    """Packed cells of a mask taken by nearest neighbour from its source grid onto `shape`."""
    mask = np.zeros(source, dtype=bool)
    for (y0, x0), bits in cells.items():
        part = np.unpackbits(bits, axis=1)[:, : source[1] - x0].astype(bool)
        mask[y0 : y0 + part.shape[0], x0 : x0 + part.shape[1]] = part
    y = np.minimum((np.arange(shape[0]) + 0.5) * source[0] / shape[0], source[0] - 1).astype(np.int64)
    x = np.minimum((np.arange(shape[1]) + 0.5) * source[1] / shape[1], source[1] - 1).astype(np.int64)
    out = {}
    for a in range(0, shape[0], size):
        rows = mask[y[a : a + size]]
        for b in range(0, shape[1], size):
            part = rows[:, x[b : b + size]]
            if part.any():
                out[(a, b)] = np.packbits(part, axis=1)
    return out


def read_mask(root, shape):
    height, width, cells = read_cells(root, label_level(root, shape))
    return cells if (height, width) == tuple(shape) else resampled_cells(cells, (height, width), shape)


def brick_grid(cells, grid_shape, bh, bw):
    """Which bricks hold a pixel of the mask, and the mask's centre in bricks."""
    grid = np.zeros(grid_shape, dtype=np.int64)
    for (y0, x0), bits in cells.items():
        ys, xs = np.nonzero(np.unpackbits(bits, axis=1))
        np.add.at(grid, ((y0 + ys) // bh, (x0 + xs) // bw), 1)
    return grid


def packed_box(cells, box):
    """The mask over the box as rows of packed bits."""
    y0, y1, x0, x1 = box
    out = np.zeros((y1 - y0, (x1 - x0) // 8), dtype=np.uint8)
    for (cy, cx), bits in cells.items():
        a, b = max(cy, y0), min(cy + bits.shape[0], y1)
        c, d = max(cx, x0), min(cx + bits.shape[1] * 8, x1)
        if a < b and c < d:
            out[a - y0 : b - y0, (c - x0) // 8 : -(-(d - x0) // 8)] |= bits[
                a - cy : b - cy, (c - cx) // 8 : -(-(d - cx) // 8)
            ]
    return out


def chosen_bricks(ink, supervised):
    """Bricks over the supervision, dilated, as a disc around the ink of at most MAX_BRICKS."""
    grid = ni.binary_dilation((ink + supervised) > 0, iterations=MARGIN_BRICKS)
    picks = np.argwhere(grid)
    if len(picks) > MAX_BRICKS:
        centre = (np.argwhere(ink > 0) if ink.any() else picks).mean(axis=0)
        picks = picks[np.argsort(((picks - centre) ** 2).sum(axis=1), kind="stable")[:MAX_BRICKS]]
    return picks


def build_case(case):
    folder = LABELLED_CASES / case["case_id"]
    if (folder / "case.json").exists():
        return json.loads((folder / "case.json").read_text())
    level, depth_um, lateral_um = cell_level(case["layers_path"])
    base = f"{BUCKET}/{case['layers_path'].rstrip('/')}/{level}"
    array = json.load(urllib.request.urlopen(f"{base}/.zarray", timeout=60))
    depth, height, width = array["shape"]
    bh, bw = array["chunks"][1:]
    root = f"{BUCKET}/{case['labels_path'].rstrip('/')}"
    ink = read_mask(f"{root}/inklabels.zarr", (height, width))
    supervised = read_mask(f"{root}/supervision.zarr", (height, width))
    shape = (-(-height // bh), -(-width // bw))
    picks = chosen_bricks(brick_grid(ink, shape, bh, bw), brick_grid(supervised, shape, bh, bw))
    box = [
        int(picks[:, 0].min()) * bh,
        (int(picks[:, 0].max()) + 1) * bh,
        int(picks[:, 1].min()) * bw,
        (int(picks[:, 1].max()) + 1) * bw,
    ]
    store = BrickStore(folder / "layers", (depth, height, width), tuple(array["chunks"]), source=case["layers_path"])
    todo = [(0, int(y), int(x)) for y, x in picks if store.offsets[0, y, x] == -1]
    separator, packed = array.get("dimension_separator", "."), array.get("compressor") is not None

    def deliver(index, raw):
        store.put(index, imagecodecs.blosc_decode(raw) if packed and raw is not None else raw)

    failures = fetch_all(base, [(i, separator.join(map(str, i))) for i in todo], deliver)
    store.save()
    if not (store.offsets >= 0).any():
        raise OSError(f"no bricks of {case['case_id']} exist under {base}")
    ink_bits, supervised_bits = packed_box(ink, box), packed_box(supervised, box)
    supervised_bits |= ink_bits
    np.save(folder / "ink.npy", ink_bits.ravel())
    np.save(folder / "supervised.npy", supervised_bits.ravel())
    record = case | {
        "level": level,
        "voxel_um": depth_um,
        "lateral_um": lateral_um,
        "shape": [depth, height, width],
        "box": box,
        "ink_pixels": int(np.bitwise_count(ink_bits).sum()),
        "supervised_pixels": int(np.bitwise_count(supervised_bits).sum()),
        "bricks": len(todo),
        "failures": len(failures),
    }
    (folder / "case.json").write_text(json.dumps(record, indent=1))
    return record


def load_truth(record, step=1):
    """(ink, supervised) boolean arrays over the case's box at every `step`-th pixel, labels registered to the scan."""
    folder = LABELLED_CASES / record["case_id"]
    y0, y1, x0, x1 = record["box"]
    rows, width = y1 - y0, x1 - x0

    def unpack(name):
        bits = np.load(folder / name, mmap_mode="r")
        if width % 8:
            return np.unpackbits(bits)[: rows * width].reshape(rows, width)[::step, ::step].astype(bool)
        bits = bits[: rows * width // 8].reshape(rows, width // 8)
        return np.unpackbits(bits[::step], axis=1)[:, ::step].astype(bool)

    ink, supervised = unpack("ink.npy"), unpack("supervised.npy")
    offset = folder / "registration.json"
    if offset.exists():
        um = json.loads(offset.read_text())["offset_um"]
        shift = [round(v / (record.get("lateral_um", record["voxel_um"]) * step)) for v in um]
        ink, supervised = np.roll(ink, shift, axis=(0, 1)), np.roll(supervised, shift, axis=(0, 1))
    return ink, supervised


IR_FRAGMENTS = (
    ("Frag1", "PHercParis2Fr47", "development"),
    ("Frag2", "PHercParis2Fr143", "development"),
    ("Frag3", "PHercParis1Fr34", "development"),
)
IR_WORKING = "54keV_exposed_surface"
IR_VOXEL_UM = 3.24
Image.MAX_IMAGE_PIXELS = None


def build_fragment(name, package, split):
    """An infrared-truth fragment as a case like the others."""
    raw = INFRARED_FRAGMENTS / name
    folder = LABELLED_CASES / f"{name}-{IR_VOXEL_UM}um"
    if (folder / "case.json").exists():
        return json.loads((folder / "case.json").read_text())
    fetch_fragment(name, package, IR_WORKING, raw)
    ink = np.array(Image.open(raw / "inklabels.png")) > 0
    mask = np.array(Image.open(raw / "mask.png")) > 0
    if ink.ndim == 3:
        ink, mask = ink[..., 0], mask[..., 0]
    layers = [tifffile.memmap(p, mode="r") for p in sorted((raw / "surface_volume").glob("*.tif"))]
    depth, (height, width), brick = len(layers), ink.shape, 128
    store = BrickStore(folder / "layers", (depth, height, width), (depth, brick, brick), source=f"{name}/{IR_WORKING}")
    for by in range(store.grid[1]):
        ys = slice(by * brick, (by + 1) * brick)
        if not mask[ys].any():
            continue
        strip = np.stack([(layer[ys] // 257).astype(np.uint8) for layer in layers])
        for bx in range(store.grid[2]):
            xs = slice(bx * brick, (bx + 1) * brick)
            if mask[ys, xs].any():
                tile = np.zeros((depth, brick, brick), dtype=np.uint8)
                part = strip[:, :, xs]
                tile[:, : part.shape[1], : part.shape[2]] = part
                store.put((0, by, bx), tile.tobytes())
    store.save()
    y, x = np.nonzero(mask)
    box = [
        int(y.min()) // brick * brick,
        min(-(-int(y.max() + 1) // brick) * brick, height),
        int(x.min()) // brick * brick,
        min(-(-int(x.max() + 1) // brick) * brick, width),
    ]
    np.save(folder / "ink.npy", np.packbits(ink[box[0] : box[1], box[2] : box[3]]))
    np.save(folder / "supervised.npy", np.packbits(mask[box[0] : box[1], box[2] : box[3]]))
    record = {
        "case_id": folder.name,
        "sample": package,
        "segment": name,
        "voxel_um": IR_VOXEL_UM,
        "scan": IR_WORKING,
        "truth": "infrared",
        "partition": split,
        "labels_resampled": False,
        "shape": [depth, height, width],
        "box": box,
        "ink_pixels": int(ink.sum()),
        "supervised_pixels": int(mask.sum()),
    }
    (folder / "case.json").write_text(json.dumps(record, indent=1))
    return record


def build():
    """Every labelled segment and fragment not yet held."""
    cases = plan_cases(segments(load()))
    print(len(cases), "cases", sum(c["partition"] == "holdout" for c in cases), "holdout", flush=True)
    for case in cases:
        record = build_case(case)
        print(
            record["case_id"],
            record["partition"],
            record["bricks"],
            "bricks",
            record["ink_pixels"],
            "ink px",
            record["failures"],
            "failures",
            flush=True,
        )
    for name, package, split in IR_FRAGMENTS:
        build_fragment(name, package, split)
    write_manifest()


def write_manifest():
    rows = [json.loads(p.read_text()) for p in sorted(LABELLED_CASES.glob("*/case.json"))]
    (LABELLED_CASES / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    sys.exit(build())
