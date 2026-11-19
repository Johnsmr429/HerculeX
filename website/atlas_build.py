"""The atlas of known letters, infrared fragments, published maps and a Greek alphabet."""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ni

from ink.datasets.labelled_cases import load_truth
from sampling.sheet_images import across_sheet
from storage.brick_store import BrickStore
from storage.data_roots import (
    BROWSER_ASSETS,
    INFRARED_FRAGMENTS,
    LABELLED_CASES,
    OPEN_DATA_MIRROR,
    SITE_ATLAS,
    WEBSITE,
)

Image.MAX_IMAGE_PIXELS = None
WORK = Path(os.environ.get("HERCULEX_ATLAS_WORK", WEBSITE / "atlas-work"))
BODY_UM = 19.0
MIN_BODY_MM2, MAX_BODY_MM2 = 0.08, 30.0
RENDER_SECONDS = float(os.environ.get("HERCULEX_ATLAS_SECONDS", "1e9"))
REGION_MARGIN_UM = 3000.0
REGION_ROWS = 4096
CARD_MM, CARD_PX, GAP = 9.0, 420, 2
OUTLINE = (255, 196, 0)
SHOWN_PERCENTILES = (0.5, 99.5)
EXAMPLE_PX, EXAMPLES, UNCONFIRMED, STRIP_PX = 200, 14, 6, 90
EMPTY_PANELS = 34
COARSE_MM, MOST_COMPARED, COMPARE_W, COMPARE_H = 0.15, 6, 800, 480
SEGMENT_PX = 2400
LINE_BAND_MM = (3.5, 14.0)
FULL_COPIES = 24
FRAGMENT_PX, FRAGMENT_PANEL, FRAGMENT_WINDOW_MM, FRAGMENT_WINDOWS = 2400, 800, 10.0, 8
WINDOW_W_MM, WINDOW_H_MM, WINDOW_PX_W, WINDOW_PX_H = 30.0, 18.0, 1000, 600
MOST_WINDOWS, MIN_WINDOW_LETTERS = 12, 2
GRID = 64
TAU = 6.0
BLOB_THICKNESS = 0.17
SHEARS, TURNS, STRETCHES = (-0.15, 0.0, 0.15), (-8.0, 0.0, 8.0), (0.85, 1.0, 1.15)
FONT = str(Path.home() / "Library/Fonts/Calibri.ttf")
ALPHABET = (
    ("Alpha", "Α", "α", "ΑΛ"),
    ("Beta", "Β", "β", "Β"),
    ("Gamma", "Γ", "γ", "Γ"),
    ("Delta", "Δ", "δ", "Δ"),
    ("Epsilon", "Ε", "ε", "ΕЄ"),
    ("Zeta", "Ζ", "ζ", "Ζ"),
    ("Eta", "Η", "η", "Η"),
    ("Theta", "Θ", "θ", "Θ"),
    ("Iota", "Ι", "ι", "Ι"),
    ("Kappa", "Κ", "κ", "Κ"),
    ("Lambda", "Λ", "λ", "Λ"),
    ("Mu", "Μ", "μ", "Μ"),
    ("Nu", "Ν", "ν", "Ν"),
    ("Xi", "Ξ", "ξ", "Ξ"),
    ("Omicron", "Ο", "ο", "Ο"),
    ("Pi", "Π", "π", "Π"),
    ("Rho", "Ρ", "ρ", "Ρ"),
    ("Sigma", "Σ", "σ", "ϹΣ"),
    ("Tau", "Τ", "τ", "Τ"),
    ("Upsilon", "Υ", "υ", "Υ"),
    ("Phi", "Φ", "φ", "Φ"),
    ("Chi", "Χ", "χ", "Χ"),
    ("Psi", "Ψ", "ψ", "Ψ"),
    ("Omega", "Ω", "ω", "Ωω"),
)


def records():
    """Every labelled control with its scroll, segment, scan energy and cell."""
    rows = [json.loads(line) for line in (LABELLED_CASES / "manifest.jsonl").read_text().splitlines()]
    for r in rows:
        r["energy"] = int(re.search(r"(\d+)keV", r["scan"]).group(1))
        r["lateral"] = r.get("lateral_um", r["voxel_um"])
    return rows


def published_maps(record):
    """The published ink images of a control's own scan, best recipe first."""
    volume = re.search(r"volume-(\d+)", record["scan"])
    if not volume:
        return []
    found = [
        p
        for p in OPEN_DATA_MIRROR.glob(
            f"{record['sample']}/segments/{record['segment']}*/ink-detection/downsampled/*.jpg"
        )
        if volume.group(1) in p.name and "alpha" not in p.name
    ]
    return sorted(found, key=lambda p: ("new_canon" not in p.name, p.name))


def cut_bodies(record):
    """Connected labelled ink of a control as letter bodies."""
    step = max(1, round(BODY_UM / record["lateral"]))
    ink, _supervised = load_truth(record, step)
    labels, count = ni.label(ink, structure=np.ones((3, 3), dtype=int))
    cell_mm = record["lateral"] * step / 1000
    rows = []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for k, box in enumerate(ni.find_objects(labels), start=1):
        area = float(sizes[k]) * cell_mm**2
        if MIN_BODY_MM2 <= area <= MAX_BODY_MM2:
            rows.append(
                {
                    "id": k,
                    "y0": box[0].start,
                    "y1": box[0].stop,
                    "x0": box[1].start,
                    "x1": box[1].stop,
                    "mm2": round(area, 3),
                    "h_mm": round((box[0].stop - box[0].start) * cell_mm, 2),
                    "w_mm": round((box[1].stop - box[1].start) * cell_mm, 2),
                }
            )
    return labels.astype(np.uint32), rows, step


def bodies():
    """Letter bodies of every control, kept in the work folder."""
    out = WORK / "bodies"
    out.mkdir(parents=True, exist_ok=True)
    for r in records():
        path = out / f"{r['case_id']}.json"
        if path.exists():
            continue
        start = time.time()
        labels, rows, step = cut_bodies(r)
        np.savez_compressed(out / f"{r['case_id']}.npz", labels=labels)
        path.write_text(json.dumps({"step": step, "rows": rows}))
        print(r["case_id"], len(rows), round(time.time() - start, 1), flush=True)


def thinned(mask):
    """Zhang-Suen thinning of a small binary picture."""
    m = mask.astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            p = np.pad(m, 1)
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            total = sum(n)
            turns = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            first = (
                (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
                if phase == 0
                else (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            )
            gone = (m == 1) & (total >= 2) & (total <= 6) & (turns == 1) & first
            if gone.any():
                m[gone] = 0
                changed = True
        if not changed:
            return m.astype(bool)


def normal_picture(mask):
    """A letter's mask cropped and fitted into the grid, its shape kept."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((GRID, GRID), dtype=bool)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = (GRID - 8) / max(crop.shape)
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) > 127
    out = np.zeros((GRID, GRID), dtype=bool)
    y, x = (GRID - small.shape[0]) // 2, (GRID - small.shape[1]) // 2
    out[y : y + small.shape[0], x : x + small.shape[1]] = small
    return out


def curve(points, steps=12):
    """A smooth line through points, Catmull-Rom."""
    p = [points[0], *points, points[-1]]
    out = []
    for i in range(1, len(p) - 2):
        a, b, c, d = map(np.array, p[i - 1 : i + 3])
        for s in np.linspace(0, 1, steps, endpoint=False):
            out.append(
                0.5 * (2 * b + (c - a) * s + (2 * a - 5 * b + 4 * c - d) * s**2 + (3 * b - a - 3 * c + d) * s**3)
            )
    out.append(np.array(points[-1]))
    return [tuple(q) for q in out]


def ring(cx, cy, rx, ry):
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)]


def bookhand():
    """Stroke models of the bookhand's letters as polylines in a box one high."""
    lunate = curve([(0.85, 0.1), (0.35, 0.03), (0.05, 0.5), (0.35, 0.97), (0.85, 0.9)])
    return {
        "Alpha": [
            [[(0, 1), (0.5, 0), (1, 1)], [(0.22, 0.62), (0.78, 0.62)]],
            [[(0, 1), (0.5, 0), (0.9, 1)], [(0.1, 0.75), (0.45, 0.45), (0.62, 0.8), (0.05, 0.8)]],
        ],
        "Beta": [
            [
                [(0.1, 0), (0.1, 1)],
                curve([(0.1, 0.02), (0.65, 0.1), (0.6, 0.42), (0.1, 0.48)]),
                curve([(0.1, 0.48), (0.8, 0.58), (0.75, 0.95), (0.1, 1)]),
            ]
        ],
        "Gamma": [[[(0.2, 1), (0.2, 0), (0.85, 0.02)]]],
        "Delta": [[[(0.5, 0), (0, 1), (1, 1), (0.5, 0)]]],
        "Epsilon": [[lunate, [(0.05, 0.5), (0.7, 0.5)]]],
        "Zeta": [[[(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)]]],
        "Eta": [[[(0.1, 0), (0.1, 1)], [(0.9, 0), (0.9, 1)], [(0.1, 0.5), (0.9, 0.5)]]],
        "Theta": [[ring(0.5, 0.5, 0.45, 0.5), [(0.1, 0.5), (0.9, 0.5)]]],
        "Iota": [[[(0.5, 0), (0.5, 1)]]],
        "Kappa": [[[(0.15, 0), (0.15, 1)], [(0.85, 0.05), (0.15, 0.55), (0.9, 1)]]],
        "Lambda": [[[(0, 1), (0.55, 0), (1, 1)]], [[(0.05, 1), (0.42, 0.42)], [(0.2, 0), (1, 1)]]],
        "Mu": [
            [[(0, 1), (0.15, 0), (0.5, 0.75), (0.85, 0), (1, 1)]],
            [curve([(0, 1), (0.1, 0.2), (0.25, 0.1), (0.5, 0.7), (0.75, 0.1), (0.9, 0.2), (1, 1)])],
        ],
        "Nu": [[[(0.1, 1), (0.1, 0), (0.9, 1), (0.9, 0)]]],
        "Xi": [[[(0.1, 0), (0.9, 0)], [(0.2, 0.5), (0.8, 0.5)], [(0.1, 1), (0.9, 1)]]],
        "Omicron": [[ring(0.5, 0.5, 0.5, 0.5)]],
        "Pi": [[[(0, 0.03), (1, 0.03)], [(0.2, 0.03), (0.2, 1)], [(0.8, 0.03), (0.8, 1)]]],
        "Rho": [[[(0.25, 0), (0.25, 1.35)], curve([(0.25, 0.02), (0.75, 0.05), (0.8, 0.4), (0.25, 0.55)])]],
        "Sigma": [[lunate], [[(0.9, 0), (0.1, 0), (0.5, 0.5), (0.1, 1), (0.9, 1)]]],
        "Tau": [[[(0, 0.02), (1, 0.02)], [(0.5, 0.02), (0.5, 1)]]],
        "Upsilon": [
            [[(0, 0), (0.5, 0.5), (1, 0)], [(0.5, 0.5), (0.5, 1)]],
            [curve([(0, 0), (0.25, 0.4), (0.5, 0.55), (0.75, 0.4), (1, 0)]), [(0.5, 0.55), (0.5, 1)]],
        ],
        "Phi": [[ring(0.5, 0.5, 0.45, 0.3), [(0.5, -0.1), (0.5, 1.1)]]],
        "Chi": [[[(0, 0), (1, 1)], [(1, 0), (0, 1)]]],
        "Psi": [
            [[(0.5, -0.05), (0.5, 1.05)], curve([(0.02, 0.1), (0.15, 0.5), (0.5, 0.65), (0.85, 0.5), (0.98, 0.1)])]
        ],
        "Omega": [
            [curve([(0.15, 0.05), (0, 0.6), (0.22, 1), (0.5, 0.55), (0.78, 1), (1, 0.6), (0.85, 0.05)])],
            [curve([(0.1, 1), (0.3, 0.9), (0.08, 0.5), (0.5, 0), (0.92, 0.5), (0.7, 0.9), (0.9, 1)])],
        ],
    }


WIDTHS = {
    "Iota": 0.12,
    "Rho": 0.55,
    "Tau": 0.85,
    "Gamma": 0.7,
    "Epsilon": 0.8,
    "Sigma": 0.8,
    "Omicron": 1.0,
    "Phi": 0.85,
    "Theta": 0.85,
    "Mu": 1.15,
    "Omega": 1.2,
    "Psi": 1.0,
    "Upsilon": 0.95,
}
HOLES = {
    "Alpha": (0, 1),
    "Beta": (1, 2),
    "Delta": (1,),
    "Theta": (2, 1),
    "Omicron": (1,),
    "Rho": (1, 0),
    "Phi": (2, 1),
    "Omega": (0,),
}
HOLE_COST = 0.06


def stroke_picture(strokes, width, shear, turn, stretch):
    """A letter's strokes drawn one cell wide on the grid."""
    a = np.radians(turn)
    lines = []
    for stroke in strokes:
        q = np.array(stroke, dtype=np.float64)
        x, y = (q[:, 0] - 0.5) * width * stretch, q[:, 1] - 0.5
        x = x + shear * y
        lines.append(np.c_[x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)])
    every = np.concatenate(lines)
    low, high = every.min(axis=0), every.max(axis=0)
    scale = (GRID - 8) / max(high - low)
    middle = (low + high) / 2
    canvas = Image.new("1", (GRID, GRID), 0)
    draw = ImageDraw.Draw(canvas)
    for q in lines:
        draw.line([tuple(v) for v in (q - middle) * scale + GRID / 2], fill=1, width=1)
    return np.asarray(canvas, dtype=bool)


def templates():
    """Distance maps and skeletons of every letter's stroke models over small changes of form, with the letter each belongs to."""
    models = bookhand()
    letters, distance, skeleton = [], [], []
    for k, row in enumerate(ALPHABET):
        for strokes in models[row[0]]:
            for shear in SHEARS:
                for turn in TURNS:
                    for stretch in STRETCHES:
                        line = stroke_picture(strokes, WIDTHS.get(row[0], 0.9), shear, turn, stretch)
                        letters.append(k)
                        distance.append(np.minimum(ni.distance_transform_edt(~line), TAU).astype(np.float32))
                        skeleton.append(line.ravel())
    return np.array(letters), np.stack(distance), np.stack(skeleton).astype(np.float32)


def holes_of(mask):
    """Enclosed background regions of a letter's picture big enough to be real."""
    picture = normal_picture(mask)
    filled = ni.binary_fill_holes(picture)
    regions, count = ni.label(filled & ~picture)
    return int((np.bincount(regions.ravel())[1:] >= 12).sum()) if count else 0


def matched(mask, letters, distance, skeleton, skip=None):
    """Best score of a letter's mask against each alphabet letter."""
    line, own = line_of(mask)
    if line.sum() < 8:
        return np.zeros(len(ALPHABET))
    ys, xs = np.nonzero(line)
    near = distance[:, ys, xs].mean(axis=1)
    far = (skeleton @ own.ravel()) / np.maximum(skeleton.sum(axis=1), 1.0)
    score = 1.0 - 0.5 * (near + far) / TAU
    if skip is not None:
        score[skip] = 0.0
    out = np.zeros(len(ALPHABET))
    np.maximum.at(out, letters, score)
    holes = holes_of(mask)
    for k, row in enumerate(ALPHABET):
        out[k] -= HOLE_COST * min(abs(holes - a) for a in HOLES.get(row[0], (0,)))
    return out


def primary_cases(rows):
    """One control per segment for the alphabet."""
    chosen = {}
    for r in rows:
        key = (r["sample"], r["segment"])
        rank = (not published_maps(r), r["energy"] not in (77, 78), "1.129um" in r["case_id"])
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, r)
    return [v[1] for v in chosen.values()]


def body_masks():
    """Every body of the primary controls as (control, row, mask)."""
    for r in primary_cases(records()):
        data = json.loads((WORK / "bodies" / f"{r['case_id']}.json").read_text())
        labels = np.load(WORK / "bodies" / f"{r['case_id']}.npz")["labels"]
        for row in data["rows"]:
            yield r, row, labels[row["y0"] : row["y1"], row["x0"] : row["x1"]] == row["id"]


def eye_reads():
    """Letters read by eye from their label shapes, once, to teach and to check the matcher."""
    index = {row[1]: k for k, row in enumerate(ALPHABET)}
    return {
        (case, body): index[letter]
        for case, body, letter in json.loads((BROWSER_ASSETS / "atlas_eye_reads.json").read_text())
    }


def line_of(mask):
    """A letter's skeleton on the grid with its clipped distance map."""
    line = thinned(ni.binary_opening(normal_picture(mask), iterations=1))
    return line, np.minimum(ni.distance_transform_edt(~line), TAU).astype(np.float32)


def match_letters():
    """Every letter body of the primary controls matched by shape to the stroke models and to the letters read by eye, a body never matched to itself."""
    start = time.time()
    models = templates()
    reads = eye_reads()
    items = list(body_masks())
    known = [i for i, (r, row, _) in enumerate(items) if (r["case_id"], row["id"]) in reads]
    lines = {i: line_of(items[i][2]) for i in known}
    letters = np.concatenate([models[0], [reads[(items[i][0]["case_id"], items[i][1]["id"])] for i in known]])
    distance = np.concatenate([models[1], np.stack([lines[i][1] for i in known])])
    skeleton = np.concatenate([models[2], np.stack([lines[i][0].ravel() for i in known]).astype(np.float32)])
    position = {i: len(models[0]) + k for k, i in enumerate(known)}
    found = []
    for i, (r, row, mask) in enumerate(items):
        score = matched(mask, letters, distance, skeleton, position.get(i))
        order = np.argsort(-score)
        eye = reads.get((r["case_id"], row["id"]))
        found.append(
            {
                "case": r["case_id"],
                "id": row["id"],
                "best": int(order[0]),
                "score": round(float(score[order[0]]), 4),
                "second": int(order[1]),
                "margin": round(float(score[order[0]] - score[order[1]]), 4),
                "eye": eye,
                "h_mm": row["h_mm"],
                "w_mm": row["w_mm"],
                "mm2": row["mm2"],
                "blob": bool(
                    ni.distance_transform_edt(normal_picture(mask)).max() / (GRID - 8) >= BLOB_THICKNESS
                    or row["w_mm"] >= 1.7 * row["h_mm"]
                ),
            }
        )
    checked = [f for f in found if f["eye"] is not None]
    agree = float(np.mean([f["best"] == f["eye"] for f in checked]))
    (WORK / "matches.json").write_text(
        json.dumps({"bodies": found, "checked": len(checked), "agreement": round(agree, 4)})
    )
    print(
        len(found),
        "bodies,",
        len(checked),
        "read by eye, matcher agrees on",
        round(agree, 3),
        round(time.time() - start),
        "s",
        flush=True,
    )


def regions(record):
    """Boxes of a control's supervised areas, in cells of the control's grid, each with a margin and cut to a size a small machine reads."""
    step = 16
    _, supervised = load_truth(record, step)
    grown = ni.binary_dilation(supervised, iterations=max(1, int(REGION_MARGIN_UM / (record["lateral"] * step))))
    labels, _ = ni.label(grown, structure=np.ones((3, 3), dtype=int))
    out = []
    for box in ni.find_objects(labels):
        y0, y1, x0, x1 = box[0].start * step, box[0].stop * step, box[1].start * step, box[1].stop * step
        for a in range(y0, y1, REGION_ROWS):
            out.append((a, min(a + REGION_ROWS, y1), x0, x1))
    return out


def render_case(record):
    """The raw scan over a control's labelled areas, at its own voxels."""
    store = BrickStore(LABELLED_CASES / record["case_id"] / "layers")
    out = WORK / "faces" / record["case_id"]
    out.mkdir(parents=True, exist_ok=True)
    y_box, x_box = record["box"][0], record["box"][2]
    tiles = []
    for k, (a, b, c, d) in enumerate(regions(record)):
        box = (y_box + a, y_box + b, x_box + c, x_box + d)
        sheet, _ = across_sheet(store, record["voxel_um"], box)
        Image.fromarray(sheet).save(out / f"tile-{k}.png")
        low, high = np.percentile(sheet[sheet > 0][::97], SHOWN_PERCENTILES) if sheet.any() else (0.0, 255.0)
        tiles.append(
            {
                "file": f"tile-{k}.png",
                "y": a,
                "x": c,
                "h": sheet.shape[0],
                "w": sheet.shape[1],
                "low": float(low),
                "high": float(max(high, low + 1)),
            }
        )
    (out / "tiles.json").write_text(json.dumps(tiles))
    return tiles


def renders():
    """Every control's render, kept."""
    rows = sorted(records(), key=lambda r: ("1.129um" in r["case_id"], r["supervised_pixels"] * r["lateral"] ** 2))
    began = time.time()
    for r in rows:
        if (WORK / "faces" / r["case_id"] / "tiles.json").exists():
            continue
        if time.time() - began > RENDER_SECONDS:
            print("time up", flush=True)
            return
        start = time.time()
        tiles = render_case(r)
        print(r["case_id"], len(tiles), "tiles", round(time.time() - start), "s", flush=True)


class CaseView:
    """One control's pictures on its own box."""

    def __init__(self, record, others=()):
        self.record, self.cell_mm = record, record["lateral"] / 1000
        tiles = WORK / "faces" / record["case_id"] / "tiles.json"
        self.tiles = json.loads(tiles.read_text()) if tiles.exists() else []
        self.factor = 1
        self.images = {}
        data = json.loads((WORK / "bodies" / f"{record['case_id']}.json").read_text())
        self.rows, self.step = data["rows"], data["step"]
        self.labels = np.load(WORK / "bodies" / f"{record['case_id']}.npz")["labels"]
        found = published_maps(record)
        self.published = np.asarray(Image.open(found[0]).convert("L")) if found else None
        self.ratio = record["shape"][1] / self.published.shape[0] if found else None
        self.borrowed = None
        if not found:
            sibling = next(
                (
                    r
                    for r in others
                    if r["sample"] == record["sample"]
                    and r["segment"] == record["segment"]
                    and r["energy"] in (77, 78)
                    and "1.129um" not in r["case_id"]
                    and published_maps(r)
                ),
                None,
            )
            if sibling:
                other = CaseView(sibling)
                dy, dx, _ = shift_mm(self, other)
                self.published, self.ratio = other.published, other.ratio
                self.borrowed = {
                    "cell_mm": other.cell_mm,
                    "y": sibling["box"][0],
                    "x": sibling["box"][2],
                    "dy": dy,
                    "dx": dx,
                    "energy": sibling["energy"],
                }

    def tile(self, tile):
        if tile["file"] not in self.images:
            raw = np.asarray(Image.open(WORK / "faces" / self.record["case_id"] / tile["file"]))
            self.images[tile["file"]] = np.clip(
                (raw.astype(np.float32) - tile["low"]) * (255.0 / (tile["high"] - tile["low"])), 0, 255
            ).astype(np.uint8)
        return self.images[tile["file"]]

    def render(self, y0, y1, x0, x1):
        """The raw scan over box pixels, black where there is none."""
        out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for t in self.tiles:
            a, b, c, d = max(y0, t["y"]), min(y1, t["y"] + t["h"]), max(x0, t["x"]), min(x1, t["x"] + t["w"])
            if a < b and c < d:
                out[a - y0 : b - y0, c - x0 : d - x0] = self.tile(t)[a - t["y"] : b - t["y"], c - t["x"] : d - t["x"]]
        return out

    def label_crop(self, y0, y1, x0, x1):
        s = self.step
        a, b, c, d = max(y0 // s, 0), max(y1 // s, 0), max(x0 // s, 0), max(x1 // s, 0)
        crop = self.labels[a:b, c:d] > 0
        return np.pad(
            crop,
            (
                (max(-y0 // s, 0), max((y1 - y0) // s - crop.shape[0] - max(-y0 // s, 0), 0)),
                (max(-x0 // s, 0), max((x1 - x0) // s - crop.shape[1] - max(-x0 // s, 0), 0)),
            ),
        )

    def published_crop(self, y0, y1, x0, x1):
        if self.published is None:
            return None
        k = self.ratio
        y, x = self.record["box"][0], self.record["box"][2]
        if self.borrowed:
            s = self.borrowed
            y0, y1 = ((v * self.cell_mm + s["dy"]) / s["cell_mm"] for v in (y0, y1))
            x0, x1 = ((v * self.cell_mm + s["dx"]) / s["cell_mm"] for v in (x0, x1))
            y, x = s["y"], s["x"]
        a, b, c, d = int((y + y0) / k), int((y + y1) / k), int((x + x0) / k), int((x + x1) / k)
        h, w = self.published.shape
        out = np.zeros((b - a, d - c), dtype=np.uint8)
        p, q, r, u = max(a, 0), min(b, h), max(c, 0), min(d, w)
        if p < q and r < u:
            out[p - a : q - a, r - c : u - c] = self.published[p:q, r:u]
        return out


def fitted(a, side_h, side_w, method=Image.LANCZOS):
    return Image.fromarray(a).resize((side_w, side_h), method)


def scale_bar(panel, mm_per_px, mm=1.0):
    """A bar of a stated length on a panel, readable on light or dark."""
    draw = ImageDraw.Draw(panel)
    length = int(round(mm / mm_per_px))
    y = panel.height - 12
    draw.rectangle((7, y - 1, 9 + length, y + 4), fill=(0, 0, 0))
    draw.rectangle((8, y, 8 + length, y + 3), fill=(255, 255, 255))
    draw.text(
        (10, y - 15),
        f"{mm:g} mm",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        font=ImageFont.truetype(FONT, 12),
    )


def outlined(panel, mask):
    """A panel with the edge of labelled ink drawn on it as a thin coloured line."""
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(panel.size, Image.NEAREST)) > 0
    edge_line = m & ~ni.binary_erosion(m, iterations=1)
    a = np.asarray(panel.convert("RGB")).copy()
    a[edge_line] = OUTLINE
    return Image.fromarray(a)


def panels(view, y0, y1, x0, x1, height, width):
    """The raw scan, the same with the label's edge, and the published map of one crop, side by side at one scale."""
    first = fitted(view.render(y0, y1, x0, x1), height, width)
    second = outlined(first, view.label_crop(y0, y1, x0, x1))
    first = first.convert("RGB")
    p = view.published_crop(y0, y1, x0, x1)
    third = fitted(p, height, width).convert("RGB") if p is not None and p.size else Image.new("RGB", (width, height))
    scale_bar(first, (x1 - x0) * view.cell_mm / width, 1.0 if (x1 - x0) * view.cell_mm < 15 else 5.0)
    canvas = Image.new("RGB", (3 * width + 2 * GAP, height))
    for k, panel in enumerate((first, second, third)):
        canvas.paste(panel, (k * (width + GAP), 0))
    return canvas


def letter_cards():
    """One card per letter body of every control, the label's middle in the middle, at one scale unless the body is bigger."""
    out = SITE_ATLAS / "letters"
    out.mkdir(exist_ok=True)
    rows = []
    matches = {(m["case"], m["id"]): m for m in json.loads((WORK / "matches.json").read_text())["bodies"]}
    every = records()
    for r in every:
        view = CaseView(r, every)
        s = view.step
        folder = out / r["case_id"]
        folder.mkdir(exist_ok=True)
        for b in view.rows:
            cy, cx = (b["y0"] + b["y1"]) * s // 2, (b["x0"] + b["x1"]) * s // 2
            side_mm = max(CARD_MM, 1.35 * max(b["h_mm"], b["w_mm"]))
            half = int(side_mm / view.cell_mm / 2) // view.factor * view.factor
            name = f"{b['id']:05d}.jpg"
            if not (folder / name).exists():
                panels(view, cy - half, cy + half, cx - half, cx + half, CARD_PX, CARD_PX).save(
                    folder / name, quality=90
                )
            m = matches.get((r["case_id"], b["id"]))
            rows.append(
                {
                    "case": r["case_id"],
                    "sample": r["sample"],
                    "segment": r["segment"],
                    "energy": r["energy"],
                    "um": round(r["voxel_um"], 2),
                    "id": b["id"],
                    "mm2": b["mm2"],
                    "h": b["h_mm"],
                    "w": b["w_mm"],
                    "y_mm": round(cy * view.cell_mm, 2),
                    "x_mm": round(cx * view.cell_mm, 2),
                    "file": f"letters/{r['case_id']}/{name}",
                    "side": round(2 * half * view.cell_mm, 1),
                    "letter": ALPHABET[m["best"]][1] if m else None,
                    "score": m["score"] if m else None,
                    "eye": ALPHABET[m["eye"]][1] if m and m["eye"] is not None else None,
                    "published": view.published is not None,
                }
            )
        print(r["case_id"], len(view.rows), flush=True)
    (WORK / "cards.json").write_text(json.dumps(rows, ensure_ascii=False))


def segment_maps():
    """Published ink maps and surface renders of every scroll, reduced for browsing, with their scale."""
    out = SITE_ATLAS / "segments"
    out.mkdir(exist_ok=True)
    rows = []
    for path in sorted(OPEN_DATA_MIRROR.glob("*/segments/*/ink-detection/downsampled/*.jpg")) + sorted(
        OPEN_DATA_MIRROR.glob("*/segments/*/surface-renders/downsampled/*.jpg")
    ):
        sample, segment = path.parts[-6], path.parts[-4]
        kind = "render" if "surface-renders" in path.parts else "ink"
        voxel = re.search(r"(\d+\.\d+)um", path.name)
        down = re.search(r"-ds(\d+)", path.name)
        scan = re.search(r"(\d+\.\d+um[^-]*(?:-[\d.]+m)?-\d+keV)", path.name)
        recipe = re.search(r"\d{14}-(?:L\d-)?(?:\d{14}-)?([a-z][^/]*?)-(?:tile|alpha|l\d)", path.name)
        name = f"{sample}-{segment[:32]}-{path.stem[-46:]}.jpg".replace("/", "-")
        target = out / name
        if not target.exists():
            image = Image.open(path).convert("L")
            size = image.size
            image.thumbnail((SEGMENT_PX, SEGMENT_PX))
            ink_share = float((np.asarray(image) > 128).mean())
            image.save(target, quality=74)
            (WORK / "segment-sizes").mkdir(exist_ok=True)
            (WORK / "segment-sizes" / f"{name}.json").write_text(
                json.dumps({"size": size, "small": image.size, "bright": round(ink_share, 4)})
            )
        info = json.loads((WORK / "segment-sizes" / f"{name}.json").read_text())
        um = float(voxel.group(1)) * int(down.group(1)) if voxel and down else None
        rows.append(
            {
                "sample": sample,
                "segment": segment,
                "kind": kind,
                "file": f"segments/{name}",
                "scan": scan.group(1) if scan else "",
                "recipe": recipe.group(1) if recipe else "",
                "mm_w": round(um * info["size"][0] / 1000) if um else None,
                "mm_h": round(um * info["size"][1] / 1000) if um else None,
                "mm_per_px": round(um * info["size"][0] / info["small"][0] / 1000, 4) if um else None,
                "w": info["small"][0],
                "h": info["small"][1],
                "bright": info["bright"],
            }
        )
        print(len(rows), name[:70], flush=True)
    (WORK / "segments.json").write_text(json.dumps(rows))


def row_strength(ink, mm_per_px):
    """How strongly a map's bright share repeats at a line spacing, along either axis."""
    best = 0.0
    for axis in (0, 1):
        profile = (ink > 128).mean(axis=axis).astype(np.float64)
        profile = profile - ni.gaussian_filter1d(profile, LINE_BAND_MM[1] / mm_per_px)
        power = np.abs(np.fft.rfft(profile * np.hanning(len(profile)))) ** 2
        periods = len(profile) * mm_per_px / np.maximum(np.arange(len(power)), 1e-9)
        band = (periods >= LINE_BAND_MM[0]) & (periods <= LINE_BAND_MM[1])
        if band.sum() >= 4:
            best = max(best, float(power[band].max() / max(np.median(power[band]), 1e-12)))
    return best


def segment_scores():
    """Published ink maps ranked by how much they look like lines of writing, so the reader meets the legible ones first."""
    rows = json.loads((WORK / "segments.json").read_text())
    for r in rows:
        if r["kind"] == "ink" and r["mm_per_px"]:
            r["rows"] = round(row_strength(np.asarray(Image.open(SITE_ATLAS / r["file"])), r["mm_per_px"]), 1)
    sources = {
        f"{p.parts[-6]}-{p.parts[-4][:32]}-{p.stem[-46:]}.jpg".replace("/", "-"): p
        for p in OPEN_DATA_MIRROR.glob("*/segments/*/ink-detection/downsampled/*.jpg")
    }
    for r in sorted((r for r in rows if r["kind"] == "ink"), key=lambda r: -r.get("rows", 0))[:FULL_COPIES]:
        name = Path(r["file"]).name
        r["full"] = f"segments/full-{name}"
        if not (SITE_ATLAS / r["full"]).exists():
            shutil.copyfile(sources[name], SITE_ATLAS / r["full"])
    (WORK / "segments.json").write_text(json.dumps(rows))
    print(sorted(((r.get("rows", 0), r["sample"], r["segment"][:20]) for r in rows), reverse=True)[:8], flush=True)


def fragments():
    """Infrared photographs beside CT for the three fragments that have them."""
    out = SITE_ATLAS / "fragments"
    out.mkdir(exist_ok=True)
    rows = []
    for r in (r for r in records() if r["truth"] == "infrared"):
        name = r["segment"]
        view = CaseView(r)
        ir = np.asarray(Image.open(INFRARED_FRAGMENTS / name / "ir.png").convert("L"))
        mask = np.asarray(Image.open(INFRARED_FRAGMENTS / name / "mask.png").convert("L")) > 0
        y0, y1, x0, x1 = r["box"]
        h, w = y1 - y0, x1 - x0
        whole = {
            "name": name,
            "sample": r["sample"],
            "mm_h": round(h * r["lateral"] / 1000, 1),
            "mm_w": round(w * r["lateral"] / 1000, 1),
            "windows": [],
        }
        scale = max(h, w) / FRAGMENT_PX
        size = (int(w / scale), int(h / scale))
        face = view.render(0, h, 0, w)
        ir_shown = np.where(mask[y0:y1, x0:x1], ir[y0:y1, x0:x1], 0).astype(np.uint8)
        for key, picture in (("ir", ir_shown), ("face", face)):
            panel = Image.fromarray(picture).resize(size, Image.LANCZOS).convert("RGB")
            scale_bar(panel, scale * r["lateral"] / 1000, 5.0)
            panel.save(out / f"{name}-{key}.jpg", quality=92)
            whole[key] = f"fragments/{name}-{key}.jpg"
        labels = outlined(Image.fromarray(ir_shown).resize(size, Image.LANCZOS), view.label_crop(0, h, 0, w))
        labels.save(out / f"{name}-labels.jpg", quality=92)
        whole["labels"] = f"fragments/{name}-labels.jpg"
        side = int(FRAGMENT_WINDOW_MM * 1000 / r["lateral"])
        ink_cells = view.label_crop(0, h, 0, w)
        density = ni.uniform_filter(ink_cells.astype(np.float32), max(1, side // view.step))
        for k in range(FRAGMENT_WINDOWS):
            y, x = np.unravel_index(np.argmax(density), density.shape)
            if density[y, x] <= 0.01:
                break
            a = max(int(y) * view.step - side // 2, 0) // 3 * 3
            c = max(int(x) * view.step - side // 2, 0) // 3 * 3
            b, d = a + side // 3 * 3, c + side // 3 * 3
            crop = ir_shown[a:b, c:d]
            if crop.shape[0] < side // 2 or crop.shape[1] < side // 2:
                density[
                    max(int(y) - side // view.step, 0) : int(y) + side // view.step,
                    max(int(x) - side // view.step, 0) : int(x) + side // view.step,
                ] = 0
                continue
            first = Image.fromarray(crop).resize((FRAGMENT_PANEL, FRAGMENT_PANEL), Image.LANCZOS)
            raw = fitted(view.render(a, b, c, d), FRAGMENT_PANEL, FRAGMENT_PANEL)
            edged = outlined(first, view.label_crop(a, b, c, d))
            first = first.convert("RGB")
            scale_bar(first, FRAGMENT_WINDOW_MM / FRAGMENT_PANEL, 1.0)
            canvas = Image.new("RGB", (3 * FRAGMENT_PANEL + 2 * GAP, FRAGMENT_PANEL))
            for n, panel in enumerate((first, raw.convert("RGB"), edged)):
                canvas.paste(panel, (n * (FRAGMENT_PANEL + GAP), 0))
            file = f"{name}-window-{k}.jpg"
            canvas.save(out / file, quality=92)
            whole["windows"].append(f"fragments/{file}")
            density[
                max(int(y) - side // view.step, 0) : int(y) + side // view.step,
                max(int(x) - side // view.step, 0) : int(x) + side // view.step,
            ] = 0
        rows.append(whole)
        print(name, len(whole["windows"]), flush=True)
    rows.sort(key=lambda f: -f["mm_h"] * f["mm_w"])
    (WORK / "fragments.json").write_text(json.dumps(rows))


def letter_windows():
    """Windows that hold several letters in rows, every control, chosen greedily to hold the most letter bodies."""
    out = SITE_ATLAS / "windows"
    out.mkdir(exist_ok=True)
    rows = []
    every = records()
    for r in every:
        view = CaseView(r, every)
        s, factor = view.step, view.factor
        centres = np.array([((b["y0"] + b["y1"]) * s / 2, (b["x0"] + b["x1"]) * s / 2) for b in view.rows])
        if len(centres) == 0:
            continue
        hw, hh = int(WINDOW_W_MM / view.cell_mm / 2), int(WINDOW_H_MM / view.cell_mm / 2)
        left = np.ones(len(centres), dtype=bool)
        k = 0
        while left.any() and k < MOST_WINDOWS:
            inside = (np.abs(centres[:, None, 0] - centres[None, left, 0]) < hh) & (
                np.abs(centres[:, None, 1] - centres[None, left, 1]) < hw
            )
            count = inside.sum(axis=1)
            best = int(np.argmax(count))
            held_here = inside[best]
            taken = centres[left][held_here]
            cy, cx = taken.mean(axis=0).astype(int)
            y0, x0 = (cy - hh) // factor * factor, (cx - hw) // factor * factor
            y1, x1 = y0 + 2 * hh // factor * factor, x0 + 2 * hw // factor * factor
            left[np.flatnonzero(left)[held_here]] = False
            if len(taken) < MIN_WINDOW_LETTERS:
                continue
            name = f"{r['case_id']}-{k:02d}.jpg"
            if not (out / name).exists():
                panels(view, y0, y1, x0, x1, WINDOW_PX_H, WINDOW_PX_W).save(out / name, quality=90)
            rows.append(
                {
                    "case": r["case_id"],
                    "sample": r["sample"],
                    "segment": r["segment"],
                    "energy": r["energy"],
                    "um": round(r["voxel_um"], 2),
                    "file": f"windows/{name}",
                    "letters": len(taken),
                    "published": view.published is not None,
                    "box": [int(y0), int(y1), int(x0), int(x1)],
                }
            )
            k += 1
        print(r["case_id"], k, flush=True)
    (WORK / "windows.json").write_text(json.dumps(rows))


NOTES = {
    "Alpha": "pointed, the cross-bar often slanted or looped",
    "Epsilon": "lunate: a rounded Є, not the angular Ε",
    "Eta": "an H",
    "Theta": "an oval with a bar, often touching both sides",
    "Iota": "a plain stroke; the commonest confusion",
    "Mu": "splayed legs, often curved",
    "Nu": "an N, the diagonal sometimes shallow, easily taken for Η",
    "Xi": "three bars, sometimes joined",
    "Pi": "top bar overhangs both legs",
    "Rho": "tall stem running below the line, small bowl",
    "Sigma": "lunate: a C, never the Σ of mathematics",
    "Upsilon": "a Y, the stem on the line",
    "Phi": "small oval, stem far above and below the line",
    "Psi": "stem above and below the line",
    "Omega": "rounded, ω-shaped and wide; the capital Ω is a later form",
}


def model_picture(strokes, width):
    """A letter's stroke model drawn large, white on dark."""
    every = np.concatenate([np.array(s, dtype=np.float64) for s in strokes])
    side, pad = 120, 18
    scale = (side - 2 * pad) / max(np.ptp(every[:, 0]) * width, np.ptp(every[:, 1]))
    middle = np.array(
        [(every[:, 0].min() + every[:, 0].max()) / 2 * width, (every[:, 1].min() + every[:, 1].max()) / 2]
    )
    canvas = Image.new("L", (side * 2, side * 2), 16)
    draw = ImageDraw.Draw(canvas)
    for stroke in strokes:
        q = (np.array(stroke, dtype=np.float64) * [width, 1] - middle) * scale + side / 2
        draw.line([tuple(v * 2) for v in q], fill=235, width=16, joint="curve")
        for v in (q[0], q[-1]):
            draw.ellipse((v[0] * 2 - 8, v[1] * 2 - 8, v[0] * 2 + 8, v[1] * 2 + 8), fill=235)
    return canvas.resize((side, side), Image.LANCZOS)


def example_panels(view, row, ir):
    """A letter tight in its crop."""
    s = view.step
    cy, cx = (row["y0"] + row["y1"]) * s // 2, (row["x0"] + row["x1"]) * s // 2
    factor = view.factor
    half = int(1.3 * max(row["h_mm"], row["w_mm"]) / view.cell_mm / 2) // factor * factor
    y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
    a, b, c, d = y0 // s, y1 // s, x0 // s, x1 // s
    labels = view.labels[max(a, 0) : b, max(c, 0) : d]
    shape = np.zeros((b - a, d - c), dtype=bool)
    shape[max(-a, 0) : max(-a, 0) + labels.shape[0], max(-c, 0) : max(-c, 0) + labels.shape[1]] = labels == row["id"]
    first = fitted((shape * 255).astype(np.uint8), EXAMPLE_PX, EXAMPLE_PX, Image.BILINEAR).convert("RGB")
    source = ir[max(y0, 0) : y1, max(x0, 0) : x1] if ir is not None else view.published_crop(y0, y1, x0, x1)
    second = (
        fitted(source, EXAMPLE_PX, EXAMPLE_PX).convert("RGB")
        if source is not None and source.size
        else Image.new("RGB", (EXAMPLE_PX, EXAMPLE_PX), (24, 24, 26))
    )
    third = fitted(view.render(y0, y1, x0, x1), EXAMPLE_PX, EXAMPLE_PX).convert("RGB")
    canvas = Image.new("RGB", (3 * EXAMPLE_PX + 2 * GAP, EXAMPLE_PX))
    for k, panel in enumerate((first, second, third)):
        canvas.paste(panel, (k * (EXAMPLE_PX + GAP), 0))
    return canvas, "infrared photograph" if ir is not None else (
        "published ML map" if source is not None else "no published map"
    )


def spread_over_scrolls(items, most):
    """Up to `most` items, best first within each scroll, taking the scrolls in turn."""
    groups = {}
    for item in items:
        groups.setdefault(item["sample"], []).append(item)
    out = []
    while len(out) < most and any(groups.values()):
        for rows in groups.values():
            if rows and len(out) < most:
                out.append(rows.pop(0))
    return out


def alphabet_examples():
    """The best real letters for every alphabet letter."""
    out = SITE_ATLAS / "alphabet"
    out.mkdir(exist_ok=True)
    found = json.loads((WORK / "matches.json").read_text())
    by_case = {r["case_id"]: r for r in records()}
    rows = []
    for m in found["bodies"]:
        rows.append(
            m
            | {
                "sample": by_case[m["case"]]["sample"],
                "segment": by_case[m["case"]]["segment"],
                "energy": by_case[m["case"]]["energy"],
            }
        )
    table = []
    wanted = {}
    models = bookhand()
    for k, row in enumerate(ALPHABET):
        sure = sorted((m for m in rows if m["eye"] == k and m["best"] == k), key=lambda m: -m["score"])
        eye_only = sorted((m for m in rows if m["eye"] == k and m["best"] != k), key=lambda m: -m["score"])
        unsure = sorted(
            (m for m in rows if m["eye"] is None and m["best"] == k and m["margin"] > 0.02 and not m["blob"]),
            key=lambda m: -m["score"],
        )
        chosen = spread_over_scrolls(sure, EXAMPLES) or spread_over_scrolls(eye_only, EXAMPLES)
        extra = spread_over_scrolls(unsure, UNCONFIRMED) if len(chosen) < EXAMPLES // 2 else []
        for m in chosen:
            m["state"] = "confirmed"
        for m in extra:
            m["state"] = "unconfirmed"
        for variant, strokes in enumerate(models[row[0]]):
            model_picture(strokes, WIDTHS.get(row[0], 0.9)).save(out / f"model-{row[0]}-{variant}.png")
        table.append(
            {
                "name": row[0],
                "capital": row[1],
                "small": row[2],
                "note": NOTES.get(row[0], ""),
                "models": [f"alphabet/model-{row[0]}-{v}.png" for v in range(len(models[row[0]]))],
                "matched": sum(m["best"] == k for m in rows),
                "eye": sum(m["eye"] == k for m in rows),
                "examples": chosen + extra,
            }
        )
        for m in chosen + extra:
            wanted.setdefault(m["case"], []).append(m)
    for case, items in wanted.items():
        r = by_case[case]
        view = CaseView(r)
        ir = None
        if r["truth"] == "infrared":
            y0, y1, x0, x1 = r["box"]
            ir = np.asarray(Image.open(INFRARED_FRAGMENTS / r["segment"] / "ir.png").convert("L"))[y0:y1, x0:x1]
        rows_by_id = {b["id"]: b for b in view.rows}
        for m in items:
            picture, source = example_panels(view, rows_by_id[m["id"]], ir)
            if np.asarray(picture)[:, EXAMPLE_PX + GAP :].mean() < EMPTY_PANELS:
                continue
            m["file"] = f"alphabet/{case}-{m['id']:05d}.jpg"
            m["source"] = source
            picture.save(SITE_ATLAS / m["file"], quality=92)
        print(case, len(items), flush=True)
    strip = Image.new("RGB", (12 * (STRIP_PX + GAP), 2 * (STRIP_PX + 20)))
    masks = {(r["case_id"], row["id"]): mask for r, row, mask in body_masks()}
    draw = ImageDraw.Draw(strip)
    for k, entry in enumerate(table):
        best = next((m for m in entry["examples"] if m.get("state") == "confirmed"), None)
        picture = (
            Image.fromarray((normal_picture(masks[(best["case"], best["id"])]) * 255).astype(np.uint8))
            if best
            else model_picture(models[entry["name"]][0], WIDTHS.get(entry["name"], 0.9))
        )
        x, y = (k % 12) * (STRIP_PX + GAP), (k // 12) * (STRIP_PX + 20)
        strip.paste(picture.resize((STRIP_PX, STRIP_PX), Image.LANCZOS).convert("RGB"), (x, y))
        draw.text(
            (x + STRIP_PX // 2 - 6, y + STRIP_PX + 2),
            entry["capital"],
            fill=(200, 198, 190),
            font=ImageFont.truetype(FONT, 14),
        )
    strip.save(out / "strip.png")
    (WORK / "alphabet.json").write_text(
        json.dumps(
            {"table": table, "checked": found["checked"], "agreement": found["agreement"], "bodies": len(rows)},
            ensure_ascii=False,
        )
    )


def coarse_labels(view):
    """A control's labels on one coarse grid in millimetres, for registering scans of a segment."""
    k = max(1, round(COARSE_MM / (view.step * view.cell_mm)))
    h, w = view.labels.shape[0] // k * k, view.labels.shape[1] // k * k
    return (view.labels[:h, :w] > 0).reshape(h // k, k, w // k, k).mean(axis=(1, 3)).astype(
        np.float32
    ), k * view.step * view.cell_mm


def shift_mm(first, second):
    """(dy, dx) in millimetres taking the first control's places to the second's, from the correlation of their labels."""
    a, mm_a = coarse_labels(first)
    b, mm_b = coarse_labels(second)
    b = ni.zoom(b, mm_b / mm_a, order=1)
    shape = [a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]]
    c = np.fft.irfft2(np.fft.rfft2(b, shape) * np.conj(np.fft.rfft2(a, shape)), shape)
    y, x = np.unravel_index(np.argmax(c), c.shape)
    y = y - shape[0] if y > shape[0] // 2 else y
    x = x - shape[1] if x > shape[1] // 2 else x
    return float(y) * mm_a, float(x) * mm_a, float(c.max() / max((a**2).sum() ** 0.5 * (b**2).sum() ** 0.5, 1e-6))


def compare():
    """The windows of each segment's 78 keV scan again in its other scans."""
    out = SITE_ATLAS / "compare"
    out.mkdir(exist_ok=True)
    windows = loaded("windows.json", [])
    by_segment = {}
    for r in records():
        by_segment.setdefault((r["sample"], r["segment"]), []).append(r)
    rows = []
    for (sample, segment), scans in by_segment.items():
        energies = {r["energy"] for r in scans}
        if len(scans) < 2 or not energies & {77, 78} or not energies & {111, 113, 116}:
            continue
        scans.sort(key=lambda r: (-r["energy"], r["voxel_um"]))
        reference = next(r for r in scans if r["energy"] in (77, 78) and "1.129um" not in r["case_id"])
        views = {
            r["case_id"]: CaseView(r) for r in scans if "4.317um" not in r["case_id"] and "2.215um" not in r["case_id"]
        }
        first = views[reference["case_id"]]
        shifts = {name: shift_mm(first, v) for name, v in views.items()}
        chosen = [w for w in windows if w["case"] == reference["case_id"]][:MOST_COMPARED]
        for k, w in enumerate(chosen):
            y0, y1, x0, x1 = w["box"]
            columns = []
            for name, v in views.items():
                dy, dx, _ = shifts[name]
                f = v.factor
                a = int((y0 * first.cell_mm + dy) / v.cell_mm) // f * f
                c = int((x0 * first.cell_mm + dx) / v.cell_mm) // f * f
                b = a + int((y1 - y0) * first.cell_mm / v.cell_mm) // f * f
                d = c + int((x1 - x0) * first.cell_mm / v.cell_mm) // f * f
                columns.append(outlined(fitted(v.render(a, b, c, d), COMPARE_H, COMPARE_W), v.label_crop(a, b, c, d)))
            scale_bar(columns[0], (x1 - x0) * first.cell_mm / COMPARE_W, 5.0)
            canvas = Image.new("RGB", (len(columns) * (COMPARE_W + GAP) - GAP, COMPARE_H))
            for i, column in enumerate(columns):
                canvas.paste(column, (i * (COMPARE_W + GAP), 0))
            name = f"{sample}-{segment}-{k:02d}.jpg"
            canvas.save(out / name, quality=90)
            rows.append(
                {
                    "sample": sample,
                    "segment": segment,
                    "file": f"compare/{name}",
                    "scans": " · ".join(f"{v.record['energy']} keV" for v in views.values()),
                }
            )
        print(sample, segment, len(chosen), {k[-22:]: round(v[2], 2) for k, v in shifts.items()}, flush=True)
    (WORK / "compare.json").write_text(json.dumps(rows))


NAV = (
    ("../sheets/index.html", "HerculeX"),
    ("index.html", "Alphabet"),
    ("lines.html", "Lines"),
    ("published.html", "Published"),
)
SCRIPTS = '<script src="data.js"></script><script src="atlas.js"></script>'


def gallery(name, grid, heading=""):
    """A section of pictures with its own filters and pages, filled by `gallery` in atlas.js."""
    title = f"<h2>{heading}</h2>" if heading else ""
    return (
        f'<section id="{name}">{title}<div class="bar"><button class="back">←</button>'
        f'<button class="next">→</button><span class="status"></span></div><div class="{grid} gallery"></div></section>'
    )


def line_spacing(cards):
    """Median distance from a letter to the next line below it, over letters of one segment and scan."""
    gaps = []
    groups = {}
    for c in cards:
        groups.setdefault(c["case"], []).append((c["y_mm"], c["x_mm"]))
    for points in groups.values():
        p = np.array(points)
        for y, x in p:
            below = p[(np.abs(p[:, 1] - x) < 4.0) & (p[:, 0] > y + 2.5) & (p[:, 0] < y + 14.0), 0]
            if len(below):
                gaps.append(below.min() - y)
    return float(np.median(gaps)) if gaps else None


def page(name, title, body, scripts=""):
    """One page of the site."""
    links = "".join(f'<a href="{file}"{" class=here" if file == name else ""}>{text}</a>' for file, text in NAV)
    (SITE_ATLAS / name).write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><link rel="stylesheet" href="atlas.css"></head><body><nav>{links}</nav><main>{body}</main>{scripts}</body></html>"""
    )


def loaded(name, fallback):
    path = WORK / name
    return json.loads(path.read_text()) if path.exists() else fallback


def pages():
    """The site, from what the other stages left in the work folder."""
    for name in ("atlas.css", "atlas.js"):
        shutil.copyfile(BROWSER_ASSETS / name, SITE_ATLAS / name)
    cards, windows, segments = loaded("cards.json", []), loaded("windows.json", []), loaded("segments.json", [])
    frags, alphabet, compare = (
        loaded("fragments.json", []),
        loaded("alphabet.json", {"table": []}),
        loaded("compare.json", []),
    )
    segments.sort(key=lambda s: (s["kind"] != "ink", -s.get("rows", 0)))
    cards.sort(key=lambda c: (c["energy"] not in (77, 78), c["um"] < 2.3, c["sample"], c["segment"], c["id"]))
    windows.sort(key=lambda w: (w["energy"] not in (77, 78), w["um"] < 2.3, w["sample"], w["segment"], -w["letters"]))
    (SITE_ATLAS / "data.js").write_text(
        "window.ATLAS="
        + json.dumps(
            {"cards": cards, "windows": windows, "segments": segments, "compare": compare},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + ";"
    )
    groups = {}
    for c in cards:
        groups.setdefault((c["sample"], c["energy"]), []).append(c)
    table = "".join(
        f"<tr><td>{s}</td><td class=n>{e} keV</td><td class=n>{len(v)}</td><td class=n>{np.median([c['h'] for c in v]):.1f}</td><td class=n>{np.median([c['w'] for c in v]):.1f}</td><td class=n>{(line_spacing(v) or 0):.1f}</td></tr>"
        for (s, e), v in sorted(groups.items())
    )
    best = next(iter(sorted(segments, key=lambda s: -s.get("rows", 0))), None)
    fronts = [(f"{f['ir']}", "lines.html#infrared") for f in frags[:1]] + (
        [(best["file"], "published.html")] if best else []
    )
    tiles = "".join(f'<a href="{link}"><img src="{image}" alt=""></a>' for image, link in fronts)
    rows = []
    for row in alphabet["table"]:
        models = "".join(f'<img src="{m}" alt="">' for m in row["models"])
        examples = "".join(
            f"""<div class="example {m.get("state", "")}"><a href="{m["file"]}"><img src="{m["file"]}" alt=""></a></div>"""
            for m in row["examples"]
            if "file" in m
        )
        rows.append(
            f"""<tr><td class="glyph">{row["capital"]}<small>{row["small"]}</small></td><td class="models">{models}</td><td class=n>{row["matched"]}</td><td><div class="examples">{examples}</div></td></tr>"""
        )
    page(
        "index.html",
        "Alphabet",
        f"""<a href="#alphabet"><img class="strip" src="alphabet/strip.png" alt=""></a><div class="grid two">{tiles}</div><table><tr><th>Scroll</th><th class=n>Energy</th><th class=n>Letters</th><th class=n>Height mm</th><th class=n>Width mm</th><th class=n>Line mm</th></tr>{table}</table><table class="alphabet" id="alphabet">{"".join(rows)}</table>{gallery("letters", "grid", "Letters")}""",
        SCRIPTS
        + """<script>gallery("letters",ATLAS.cards,[{label:"Scroll",get:i=>i.sample},{label:"Energy",get:i=>i.energy+" keV"},{label:"Letter",get:i=>i.letter},{label:"Voxel",get:i=>i.um+" µm"}],i=>`<figure><a href="${i.file}"><img src="${i.file}" alt=""></a><figcaption>${i.sample} · ${i.energy} keV · ${i.um} µm${i.letter?" · "+i.letter:""}</figcaption></figure>`)</script>""",
    )
    sections = []
    for f in frags:
        whole = "".join(f'<a href="{f[k]}"><img src="{f[k]}" alt=""></a>' for k in ("ir", "face", "labels") if k in f)
        close = "".join(f'<a href="{w}"><img src="{w}" alt=""></a>' for w in f["windows"])
        sections.append(
            f'<h2>Infrared · {f["name"]} · {f["sample"]}</h2><div class="grid three">{whole}</div><div class="grid wide">{close}</div>'
        )
    page(
        "lines.html",
        "Lines",
        f"""<section id="infrared">{"".join(sections)}</section>{gallery("scans", "grid wide", "Scans")}{gallery("lines", "grid wide", "Lines")}""",
        SCRIPTS
        + """<script>gallery("scans",ATLAS.compare,[{label:"Scroll",get:i=>i.sample},{label:"Segment",get:i=>i.segment}],i=>`<figure><a href="${i.file}"><img src="${i.file}" alt=""></a><figcaption>${i.sample} · ${i.scans}</figcaption></figure>`)
gallery("lines",ATLAS.windows,[{label:"Scroll",get:i=>i.sample},{label:"Energy",get:i=>i.energy+" keV"},{label:"Segment",get:i=>i.segment}],i=>`<figure><a href="${i.file}"><img src="${i.file}" alt=""></a><figcaption>${i.sample} · ${i.energy} keV · ${i.um} µm</figcaption></figure>`)</script>""",
    )
    page(
        "published.html",
        "Published",
        gallery("published", "grid"),
        SCRIPTS
        + """<script>gallery("published",ATLAS.segments,[{label:"Scroll",get:i=>i.sample},{label:"Kind",get:i=>i.kind},{label:"Scan",get:i=>i.scan},{label:"Recipe",get:i=>i.recipe}],i=>`<figure><a href="${i.full||i.file}"><img src="${i.file}" alt=""></a><figcaption>${i.sample} · ${i.scan}</figcaption></figure>`)</script>""",
    )
    for old in SITE_ATLAS.glob("*.html"):
        if old.name not in {file for file, _ in NAV}:
            old.unlink()


def everything():
    """All stages in order."""
    for stage in (
        bodies,
        match_letters,
        renders,
        segment_maps,
        segment_scores,
        letter_cards,
        letter_windows,
        fragments,
        compare,
        alphabet_examples,
        pages,
    ):
        stage()


if __name__ == "__main__":
    globals()[sys.argv[1]]()
