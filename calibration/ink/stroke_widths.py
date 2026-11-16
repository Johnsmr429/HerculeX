"""Stroke widths of a model's read of known text and its scores against finer truths: fine-scan maps on published surfaces and our own pieces, infrared photographs on the fragments."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ni
from skimage.morphology import skeletonize

from calibration.ink.ink_yield import rank_auc
from calibration.ink.own_mesh_windows import face_read
from ink.datasets.labelled_cases import load_truth
from ink.datasets.own_mesh_labels import kept_windows
from ink.datasets.stroke_core import stroke_core
from ink.model.stock_model import SURE
from ink.model.turned_reader import named_reader
from ink.training.model_training import SURE_BLANK, SURE_INK
from storage.data_roots import (
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    PUBLISHED_PAIRS,
    calibration_output,
)

OUT = calibration_output("stroke_widths")
ROUTES = ("level_route", "grown_route")
EVERY = 3
FEWEST_SKELETON_PIXELS = 50
FRAGMENT_STEP = 3
HALF = 0.5
Image.MAX_IMAGE_PIXELS = None


def fine_map(case):
    """The published fine-scan map on the pixels of a case's box, moved as its labels are."""
    image = Image.open(PUBLISHED_PAIRS / "maps" / f"{case['sample']}-{case['segment']}.png")
    _, height, width = case["shape"]
    y0, y1, x0, x1 = case["box"]
    sy, sx = image.size[1] / height, image.size[0] / width
    part = np.asarray(
        image.transform((x1 - x0, y1 - y0), Image.AFFINE, (sx, 0, x0 * sx, 0, sy, y0 * sy), Image.BILINEAR)
    )
    offset = LABELLED_CASES / case["case_id"] / "registration.json"
    if offset.exists():
        um = json.loads(offset.read_text())["offset_um"]
        part = np.roll(part, [round(v / case.get("lateral_um", case["voxel_um"])) for v in um], axis=(0, 1))
    return part


def stroke_width(mask, mm):
    """Median stroke width in mm, twice the distance to the edge along the skeleton."""
    mask = ni.binary_opening(mask, iterations=1)
    widths = 2 * ni.distance_transform_edt(mask)[skeletonize(mask)] * mm
    return float(np.median(widths)) if len(widths) >= FEWEST_SKELETON_PIXELS else float("nan")


def auc(read, ink, blank):
    return float(rank_auc(read[ink][::EVERY], read[blank][::EVERY])) if ink.any() and blank.any() else float("nan")


def truth_of(label, known, fine, mm):
    """What a window's two truths say of themselves."""
    thin = known & (fine >= SURE_INK)
    return {
        "known": int(known.sum()),
        "label_width": stroke_width(known & label, mm),
        "fine_width": stroke_width(thin, mm),
        "fine_share": float(thin.sum() / max(known.sum(), 1)),
        "fine_inside_label": float((thin & label).sum() / max(thin.sum(), 1)),
    }


def scores(read, label, known, fine, mm):
    """A read against the hand label, against the fine map, and inside the hand label alone, with its stroke width at the fine map's share of ink."""
    thin, blank = known & (fine >= SURE_INK), known & (fine <= SURE_BLANK) & (fine > 0)
    if not thin.any():
        return None
    cut = np.quantile(read[known][::EVERY], 1 - thin.sum() / known.sum())
    ridge = stroke_core(label, mm)
    return {
        "auc_label": auc(read, known & label, known & ~label),
        "auc_fine": auc(read, thin, blank),
        "auc_inside_label": auc(read, thin & label, blank & label),
        "auc_core": auc(read, known & (ridge >= HALF), known & label & (ridge == 0)),
        "auc_centre": auc(read, known & (ridge >= HALF), known & ~label),
        "width": stroke_width(known & (read >= cut), mm),
        "sure_share": float((read[known] >= SURE).mean()),
        "sure_width": stroke_width(known & (read >= SURE), mm),
    }


def published(sample):
    """Name, stack, hand label, known pixels, fine map and pixel size of a scroll's labelled published surfaces."""
    for path in sorted(LABELLED_CASES.glob(f"{sample}-*/case.json")):
        case = json.loads(path.read_text())
        stack_path = LABELLED_CASE_STACKS / f"{case['case_id']}.npy"
        if not 7 <= case["voxel_um"] <= 10 or not stack_path.exists():
            continue
        if not (PUBLISHED_PAIRS / "maps" / f"{case['sample']}-{case['segment']}.png").exists():
            continue
        stack = np.load(stack_path, mmap_mode="r")
        label, known = load_truth(case)
        fine = fine_map(case)
        height, width = min(stack.shape[1], label.shape[0]), min(stack.shape[2], label.shape[1])
        stack = stack[:, :height, :width]
        on_sheet = known[:height, :width] & (stack[stack.shape[0] // 2] > 0)
        yield case["case_id"], stack, label[:height, :width], on_sheet, fine[:height, :width], case["voxel_um"] / 1000


def own(sample, route):
    """The same for our own pieces, the fine map carried through each labelled pixel's published pixel."""
    for path, row, labels in kept_windows(sample, route):
        case = json.loads((LABELLED_CASES / row["case"] / "case.json").read_text())
        ys, xs, rows, columns = labels["published"]
        fine = np.zeros(labels["known"].shape, dtype=np.uint8)
        fine[ys, xs] = fine_map(case)[rows, columns]
        yield path.stem, np.load(path, mmap_mode="r"), labels["ink"], labels["known"], fine, case["voxel_um"] / 1000


def fragments(turned):
    """The same for the infrared fragments at a third of their scan's sampling, the photograph's ink standing as the fine map."""
    for path in sorted(LABELLED_CASES.glob("Frag*/case.json")):
        case = json.loads(path.read_text())
        stack = np.load(LABELLED_CASE_STACKS / f"{case['case_id']}.npy", mmap_mode="r")
        label, known = load_truth(case, FRAGMENT_STEP)
        height, width = min(stack.shape[1], label.shape[0]), min(stack.shape[2], label.shape[1])
        stack = stack[::-1, :height, :width] if turned else stack[:, :height, :width]
        label, known = label[:height, :width], known[:height, :width] & (stack[stack.shape[0] // 2] > 0)
        yield case["case_id"], stack, label, known, np.where(label, 255, 1), case["voxel_um"] * FRAGMENT_STEP / 1000


def weighted(rows):
    """Every score's mean over windows, weighted by known pixels."""
    weights = np.array([row["known"] for row in rows], dtype=float)
    keys = [key for key in rows[0] if key not in ("name", "known")]
    table = np.array([[row[key] for key in keys] for row in rows], dtype=float)
    seen = ~np.isnan(table)
    means = (np.where(seen, table, 0) * weights[:, None]).sum(0) / np.maximum((seen * weights[:, None]).sum(0), 1)
    return {key: round(float(value), 4) for key, value in zip(keys, means)} | {"windows": len(rows)}


def by_segment(rows):
    """The same means for each labelled segment apart."""
    segments = sorted({row["name"].split("-")[1] for row in rows if "-" in row["name"]})
    return {s: weighted([row for row in rows if row["name"].split("-")[1:2] == [s]]) for s in segments}


def run(sample, out, paths):
    sets = {"published": lambda: published(sample)} | {route: lambda r=route: own(sample, r) for route in ROUTES}
    if sample == "fragments":
        sets = {"fragments": lambda: fragments(False), "fragments turned": lambda: fragments(True)}
    kept = {"sample": sample}
    for path in paths:
        model, reader = named_reader(path)
        for name, windows in sets.items():
            rows = []
            for window, stack, label, known, fine, mm in windows():
                found = scores(face_read(reader, stack), label, known, fine, mm)
                if found:
                    rows.append({"name": window} | truth_of(label, known, fine, mm) | found)
            if rows:
                kept[f"{model} {name}"] = {"rows": rows, "by_segment": by_segment(rows)} | weighted(rows)
                print(model, name, weighted(rows), flush=True)
        del reader
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
