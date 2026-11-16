"""Letters named by eye on the labelled cases, each as a window of its published ink map, for the letter classifier.

Every letter of website/assets/atlas_eye_reads.json (a body of a case's labelled ink, named once by eye from its label
shape when the atlas was built) is cut as a square SIDE reference heights on a side round the body's centre, at PX
pixels. The reference height is the median height of the case's named letters, so a window holds a letter at the
scale of its own row. Each window carries:

    ink        the published ML ink map of the case's scan (Vesuvius Challenge open data), ink bright; for the three
               Paris fragments, which have none, the infrared photograph turned so ink is bright
    own, any   the letter's own labelled ink and all labelled ink, on the same grid
    known      the supervised area on the same grid
    auc        the map at the letter's own ink against supervised blank more than BLANK_CELLS label cells from any
               labelled ink, inside the window

./run.sh ink.datasets.eye_letters build        writes datasets/eye_letters/letters.npz and letters.json
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from storage.data_roots import BROWSER_ASSETS, INFRARED_FRAGMENTS, INK_TRAINING_DATA, LIBRARY

# the atlas build's work folder: letter bodies cut from each case's labels
os.environ.setdefault("HERCULEX_ATLAS_WORK", str(LIBRARY / "atlas" / "work"))

from ink.datasets.labelled_cases import load_truth
from website import atlas_build

OUT = INK_TRAINING_DATA / "eye_letters"
SIDE, PX = 3.0, 144
BLANK_CELLS = 2
GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"


def auc(pos, neg):
    """Area under the ROC curve of pos against neg, ties counted half."""
    if not len(pos) or not len(neg):
        return None
    values = np.concatenate([pos, neg]).astype(np.float64)
    order = values.argsort(kind="mergesort")
    r = np.empty(len(values))
    r[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=r)
    r = (sums / counts)[inverse]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cells(labels, y0, y1, x0, x1):
    """A crop of a label-cell map, zeros outside it."""
    out = np.zeros((y1 - y0, x1 - x0), dtype=labels.dtype)
    a, b, c, d = max(y0, 0), min(y1, labels.shape[0]), max(x0, 0), min(x1, labels.shape[1])
    if a < b and c < d:
        out[a - y0 : b - y0, c - x0 : d - x0] = labels[a:b, c:d]
    return out


def resized(a, px, method):
    return np.asarray(Image.fromarray(a).resize((px, px), method))


def build():
    reads = json.loads((BROWSER_ASSETS / "atlas_eye_reads.json").read_text())
    every = atlas_build.records()
    by_case = {r["case_id"]: r for r in every}
    named = {}
    for case, body, letter in reads:
        named.setdefault(case, []).append((body, letter))
    ink, own, anyink, known, rows = [], [], [], [], []
    for case, items in sorted(named.items()):
        r = by_case[case]
        view = atlas_build.CaseView(r, every)
        s = view.step
        bodies = {b["id"]: b for b in view.rows}
        _, supervised = load_truth(r, s)
        reference_mm = float(np.median([bodies[b]["h_mm"] for b, _ in items]))
        half = int(round(SIDE * reference_mm / view.cell_mm / 2 / s)) * s
        ir = None
        if r["truth"] == "infrared":
            y0, y1, x0, x1 = r["box"]
            ir = 255 - np.asarray(Image.open(INFRARED_FRAGMENTS / r["segment"] / "ir.png").convert("L"))[y0:y1, x0:x1]
        for body, letter in items:
            b = bodies[body]
            cy, cx = (b["y0"] + b["y1"]) * s // 2 // s * s, (b["x0"] + b["x1"]) * s // 2 // s * s
            y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
            source = cells(ir, y0, y1, x0, x1) if ir is not None else view.published_crop(y0, y1, x0, x1)
            if source is None or not source.size:
                continue
            labels = cells(view.labels, y0 // s, y1 // s, x0 // s, x1 // s)
            sup = cells(supervised, y0 // s, y1 // s, x0 // s, x1 // s)
            mine, inked = labels == body, labels > 0
            on_cells = resized(source, labels.shape[0], Image.BILINEAR).astype(np.float32)
            blank = sup & ~ni.binary_dilation(inked, iterations=BLANK_CELLS)
            score = auc(on_cells[mine], on_cells[blank])
            ink.append(resized(source, PX, Image.LANCZOS))
            own.append(resized(mine.astype(np.uint8) * 255, PX, Image.NEAREST) > 127)
            anyink.append(resized(inked.astype(np.uint8) * 255, PX, Image.NEAREST) > 127)
            known.append(resized(sup.astype(np.uint8) * 255, PX, Image.NEAREST) > 127)
            rows.append(
                {
                    "case": case,
                    "sample": r["sample"],
                    "segment": r["segment"],
                    "partition": r.get("partition"),
                    "source": "infrared" if ir is not None else "published",
                    "body": body,
                    "letter": letter,
                    "class": GREEK.index(letter),
                    "h_mm": b["h_mm"],
                    "w_mm": b["w_mm"],
                    "reference_mm": round(reference_mm, 3),
                    "window_mm": round(2 * half * view.cell_mm, 3),
                    "auc": round(score, 4) if score is not None else None,
                    "blank_cells": int(blank.sum()),
                    "map_cover": round(float((source > 0).mean()), 3),
                }
            )
        print(f"{case:55s} {len(items):3d} reference {reference_mm:.2f} mm", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "letters.npz", ink=np.stack(ink), own=np.stack(own), any=np.stack(anyink), known=np.stack(known)
    )
    (OUT / "letters.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    published = [x for x in rows if x["source"] == "published"]
    good = [x for x in published if (x["auc"] or 0) >= 0.9]
    print(len(rows), "letters;", len(published), "on published maps,", len(good), "of them at AUC >= 0.9", flush=True)


if __name__ == "__main__":
    {"build": build}[sys.argv[1]](*sys.argv[2:])
