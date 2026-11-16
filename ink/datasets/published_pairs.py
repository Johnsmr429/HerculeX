"""Training pairs: 9 µm surface windows with the ink map a stronger model made from a better scan."""

import io
import json
import re
import sys
import urllib.request

import numpy as np
from PIL import Image

from scans.catalog.scan_catalog import BUCKET, load, segments
from scans.transfer.parallel_fetch import fetch_all
from storage.data_roots import INK_RECORDS, PUBLISHED_PAIRS, PUBLISHED_PAIRS_WRITING

WINDOW = 2048
BRICK = 128
BUDGET = {"PHerc0500P2": 100, "PHerc0009B": 60, "PHerc0343P": 25, "PHerc0841": 14, "PHerc0139": 80, "PHerc0814": 60}
Image.MAX_IMAGE_PIXELS = None


def voxel_of(name):
    found = re.search(r"(\d+\.?\d*)um", name)
    return float(found.group(1)) if found else 0.0


def listed():
    """Every segment with layers near 9 um and a published map."""
    rows = []
    for row in segments(load()):
        layers = [
            p for p in row["products"].get("layers-zarr", []) if 7 <= voxel_of(p.rstrip("/").rsplit("/", 1)[1]) <= 10
        ]
        maps = sorted(row["products"].get("ink-detection-downsampled", []), key=lambda p: "new_canon" not in p)
        if layers and maps and row["sample"] in BUDGET:
            rows.append(
                {"sample": row["sample"], "segment": row["segment"], "layers": layers[0].rstrip("/"), "map": maps[0]}
            )
    return rows


def map_of(row):
    path = PUBLISHED_PAIRS / "maps" / f"{row['sample']}-{row['segment']}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(f"{BUCKET}/{row['map']}", timeout=120).read()
        Image.open(io.BytesIO(raw)).convert("L").save(path)
    return np.asarray(Image.open(path))


def shape_of(row):
    return json.load(urllib.request.urlopen(f"{BUCKET}/{row['layers']}/0/.zarray", timeout=60))


def planned(rows):
    """The richest windows of every scroll, as many as its budget, none overlapping."""
    plan = []
    for sample, budget in BUDGET.items():
        found = []
        for row in (r for r in rows if r["sample"] == sample):
            image = map_of(row).astype(np.float32)
            array = shape_of(row)
            _, height, _width = array["shape"]
            scale = image.shape[0] / height
            side = int(WINDOW * scale)
            strong = image >= max(float(np.percentile(image[image > 0], 90)) if (image > 0).any() else 255, 96)
            mass = np.pad(strong.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            seen = np.pad((image > 0).astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            taken = []
            step = max(1, int(BRICK * 4 * scale))
            options = []
            for a in range(0, max(image.shape[0] - side, 0) + 1, step):
                for b in range(0, max(image.shape[1] - side, 0) + 1, step):
                    a1, b1 = min(a + side, image.shape[0]), min(b + side, image.shape[1])
                    cover = seen[a1, b1] - seen[a, b1] - seen[a1, b] + seen[a, b]
                    ink = mass[a1, b1] - mass[a, b1] - mass[a1, b] + mass[a, b]
                    if cover >= 0.5 * side * side:
                        options.append((ink / cover, a, b))
            for share, a, b in sorted(options, reverse=True):
                if all(abs(a - p) >= side or abs(b - q) >= side for p, q in taken):
                    taken.append((a, b))
                    found.append((share, row, int(a / scale) // BRICK * BRICK, int(b / scale) // BRICK * BRICK))
        found.sort(key=lambda f: -f[0])
        plan += found[:budget]
        print(
            sample,
            "options",
            len(found),
            "taken",
            min(budget, len(found)),
            "share",
            round(found[0][0], 3) if found else None,
            round(found[min(budget, len(found)) - 1][0], 3) if found else None,
            flush=True,
        )
    return plan


def fetch_window(item, folder=PUBLISHED_PAIRS):
    """A window's raw layers and its map on the same pixels."""
    _share, row, y, x = item
    out = folder / row["sample"] / f"{row['segment']}-{y}-{x}.npy"
    if out.exists() or (PUBLISHED_PAIRS / row["sample"] / out.name).exists():
        return out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    array = shape_of(row)
    depth, height, width = array["shape"]
    bd, bh, bw = array["chunks"]
    cube = np.zeros((depth, WINDOW, WINDOW), dtype=np.uint8)
    base = f"{BUCKET}/{row['layers']}/0"
    separator = array.get("dimension_separator", ".")
    items = [
        ((0, y // bh + i, x // bw + j), separator.join(map(str, (0, y // bh + i, x // bw + j))))
        for i in range(WINDOW // bh)
        for j in range(WINDOW // bw)
        if (y // bh + i) * bh < height and (x // bw + j) * bw < width
    ]

    def deliver(index, raw):
        if raw is None:
            return
        block = np.frombuffer(raw if array.get("compressor") is None else decoded(raw), dtype=np.uint8).reshape(
            bd, bh, bw
        )
        a, b = index[1] * bh - y, index[2] * bw - x
        cube[:, a : a + bh, b : b + bw] = block[:depth]

    fetch_all(base, items, deliver)
    image = Image.fromarray(map_of(row))
    scale_y, scale_x = image.size[1] / height, image.size[0] / width
    label = image.transform(
        (WINDOW, WINDOW), Image.AFFINE, (scale_x, 0, x * scale_x, 0, scale_y, y * scale_y), Image.BILINEAR
    )
    np.save(out.with_suffix(".part.npy"), cube)
    out.with_suffix(".part.npy").replace(out)
    label.save(out.with_suffix(".png"))
    return out.name


def decoded(raw):
    import imagecodecs

    return imagecodecs.blosc_decode(raw)


def run():
    rows = listed()
    print(len(rows), "segments", flush=True)
    plan_path = PUBLISHED_PAIRS / "plan.json"
    if not plan_path.exists():
        PUBLISHED_PAIRS.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(planned(rows)))
    plan = json.loads(plan_path.read_text())
    print(len(plan), "windows", flush=True)
    for k, item in enumerate(plan):
        print(k + 1, "of", len(plan), item[1]["sample"], fetch_window(item), round(item[0], 3), flush=True)


def run_writing():
    rows = {(r["sample"], r["segment"]): r for r in listed()}
    plan = json.loads((INK_RECORDS / "published-windows-richest-in-text.json").read_text())
    for k, window in enumerate(plan):
        item = (window["text_cm2"], rows[window["sample"], window["segment"]], window["y"], window["x"])
        print(k + 1, "of", len(plan), window["sample"], fetch_window(item, PUBLISHED_PAIRS_WRITING), flush=True)


if __name__ == "__main__":
    run_writing() if sys.argv[1:] == ["writing"] else run()
