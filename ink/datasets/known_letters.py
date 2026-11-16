"""Known letters of the hand-labelled cases, how much of each our own pieces reach, and how a model reads each on them."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from calibration.ink.agreed_truth import agreed, offset_field, oracle_read, own_windows, scores, shifted_read
from ink.datasets.own_mesh_labels import KINDS, ROUTE_FOLDERS, fine_cases_of, kept_windows, truth_of
from ink.model.input_stack import PAST_FACE, face_layer_of
from ink.model.turned_reader import named_reader
from ink.reading.passage_reads import ink_panel
from sampling.piece_images import FACE_LAYER
from storage.data_roots import LABELLED_CASES, PUBLISHED_PAIRS, SHEETS, calibration_output
from website import panel_pages

OUT = calibration_output("known_letters")
JOIN_UM = 125.0
LEAST_MM2 = 0.5
HEIGHT_MM = (1.0, 6.0)
MOST_WIDTH_MM = 12.0
REACHED = 0.8
MEAN_SHIFTS = (-4, -2, 0, 2, 4)
NEAR_MM = 0.5
GAP_PIXELS = 2
HOLDOUT_SCROLLS = ("PHerc0841", "PHerc0814")
READ_NOT_TAUGHT = ("PHerc0139-20260115000000",)
SURFACE_LAYERS = (FACE_LAYER - PAST_FACE - 1, FACE_LAYER - PAST_FACE + 2)


def cases_near_9um(kind="labels"):
    """Every hand-labelled case of a scan near 9 um, or every published window with fine-map ink."""
    if kind == "fine":
        plan = json.loads((PUBLISHED_PAIRS / "plan.json").read_text())
        scans = sorted(
            {
                (meta["sample"], Path(meta["layers"]).name.removesuffix(".zarr").split("volume-")[1])
                for _, meta, _, _ in plan
            }
        )
        for sample, volume in scans:
            yield from fine_cases_of(sample, volume)
        return
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        if 7 <= case["voxel_um"] <= 10:
            yield case


def letters_of(case):
    """A case's label map of letter-sized blobs, strokes joined across gaps of JOIN_UM, and the blobs' sizes."""
    ink, _ = truth_of(case)
    um = case["voxel_um"]
    joined = ni.binary_closing(ink, structure=np.ones((3, 3)), iterations=round(JOIN_UM / um))
    blobs, count = ni.label(joined)
    kept = np.zeros(count + 1, dtype=bool)
    sizes = []
    for k, box in enumerate(ni.find_objects(blobs), start=1):
        mm2 = float(((blobs[box] == k) & ink[box]).sum()) * (um / 1000) ** 2
        height, width = ((s.stop - s.start) * um / 1000 for s in box)
        if mm2 >= LEAST_MM2 and HEIGHT_MM[0] <= height <= HEIGHT_MM[1] and width <= MOST_WIDTH_MM:
            kept[k] = True
            sizes.append(
                {"letter": k, "height_mm": round(height, 2), "width_mm": round(width, 2), "ink_mm2": round(mm2, 2)}
            )
    return np.where(kept[blobs] & ink, blobs, 0), sizes


def reached(case, letters, route, kind="labels"):
    """For each letter, the share of its labelled ink that any kept window of our pieces holds."""
    seen = np.zeros(letters.shape, dtype=bool)
    for _, row, labels in kept_windows(case["sample"], ROUTE_FOLDERS[route], KINDS[kind]):
        if row["case"] == case["case_id"]:
            _, _, at_row, at_column = labels["published"]
            seen[at_row, at_column] = True
    ids = np.unique(letters[letters > 0])
    inked = ni.sum_labels(np.ones(letters.shape), letters, ids)
    held = ni.sum_labels(seen, letters, ids)
    return {int(k): round(float(h / max(n, 1)), 3) for k, h, n in zip(ids, held, inked)}


def census(route="level", kind="labels"):
    """Every known letter near 9 um and how much of it our pieces of a route reach."""
    rows = []
    for case in cases_near_9um(kind):
        letters, sizes = letters_of(case)
        share = reached(case, letters, route, kind)
        for size in sizes:
            rows.append(
                {"case": case["case_id"], "sample": case["sample"], **size, "reached": share.get(size["letter"], 0.0)}
            )
        held = sum(share.get(s["letter"], 0.0) >= REACHED for s in sizes)
        print(f"{case['case_id']:52s} letters {len(sizes):3d} reached {held:3d}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"census-{kind}-{route}.json").write_text(json.dumps(rows, indent=1))
    print(route, "letters", len(rows), "reached", sum(r["reached"] >= REACHED for r in rows), flush=True)
    return rows


def scored(route, weights, page=None, kind="labels", root=None):
    """From the windows under root when given, else the kind's own. Each letter as far as our windows of a route reach it, read by one model at the piece's face and as the mean over MEAN_SHIFTS, beside the published read on the same pixels, scored on agreed truth near it; with a page of each letter in the published frame."""
    model, reader = named_reader(weights)
    windows_root = Path(root) if root else KINDS[kind]
    out = SHEETS / page if page else None
    if out:
        shutil.rmtree(out, ignore_errors=True)
    rows = []
    for case in cases_near_9um(kind):
        if not any(kept_windows(case["sample"], ROUTE_FOLDERS[route], windows_root)):
            continue
        letters, sizes = letters_of(case)
        stack, ink, blank, halo = agreed(case)
        published = shifted_read(reader, stack, 0)
        sheet = np.asarray(stack[stack.shape[0] // 2]) > 0
        reach = round(NEAR_MM * 1000 / case["voxel_um"])
        found = ni.find_objects(letters)
        boxes = {
            s["letter"]: (
                max(found[s["letter"] - 1][0].start - reach, 0),
                min(found[s["letter"] - 1][0].stop + reach, letters.shape[0]),
                max(found[s["letter"] - 1][1].start - reach, 0),
                min(found[s["letter"] - 1][1].stop + reach, letters.shape[1]),
            )
            for s in sizes
        }
        label, _ = truth_of(case)
        ours = {k: np.full((4, b - a, d - c), np.nan, dtype=np.float32) for k, (a, b, c, d) in boxes.items()}
        published_ct = surface_of(stack, face_layer_of(stack))
        for _, case_id, own, (ys, xs, at_row, at_column), face in own_windows(
            case["sample"], ROUTE_FOLDERS[route], windows_root
        ):
            if case_id != case["case_id"]:
                continue
            inside = (at_row < ink.shape[0]) & (at_column < ink.shape[1])
            ys, xs, at_row, at_column = ys[inside], xs[inside], at_row[inside], at_column[inside]
            near = [
                k
                for k, (a, b, c, d) in boxes.items()
                if ((at_row >= a) & (at_row < b) & (at_column >= c) & (at_column < d)).any()
            ]
            if not near:
                continue
            reads = [shifted_read(reader, own, shift, face) for shift in MEAN_SHIFTS]
            offsets = offset_field(own, stack, ys, xs, at_row, at_column)
            oracle = oracle_read(reader, own, stack, offsets) if offsets is not None else np.full(own.shape[1:], np.nan)
            ct = surface_of(own, face)
            values = np.stack(
                [reads[MEAN_SHIFTS.index(0)][ys, xs], np.mean(reads, axis=0)[ys, xs], oracle[ys, xs], ct[ys, xs]]
            )
            for k in near:
                a, b, c, d = boxes[k]
                take = (at_row >= a) & (at_row < b) & (at_column >= c) & (at_column < d)
                ours[k][:, at_row[take] - a, at_column[take] - c] = values[:, take]
        for size in sizes:
            k = size["letter"]
            a, b, c, d = boxes[k]
            ours[k], covered = gaps_filled(ours[k])
            mine = letters[a:b, c:d] == k
            row = {
                "mark": mark_of(case, k),
                "case": case["case_id"],
                "holdout": case["sample"] in HOLDOUT_SCROLLS,
                **size,
                "reached": round(float((covered & mine).sum() / max(mine.sum(), 1)), 3),
            }
            masks = (ink[a:b, c:d] & mine & covered, blank[a:b, c:d] & covered, halo[a:b, c:d] & covered)
            shown_here = covered & sheet[a:b, c:d]
            for key, values in (
                ("published", published[a:b, c:d]),
                ("one", ours[k][0]),
                ("mean", ours[k][1]),
                ("oracle", ours[k][2]),
            ):
                if np.isnan(values[shown_here]).any():
                    continue
                found = scores(np.nan_to_num(values), *masks)
                if found:
                    row[key] = {"auc": found["auc"], "ink_sure": found["ink_sure"], "halo_sure": found["halo_sure"]}
                    if key != "published" and shown_here.sum() > 100:
                        row[key]["shape"] = round(
                            float(np.corrcoef(values[shown_here], published[a:b, c:d][shown_here])[0, 1]), 3
                        )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if out and covered.any():
                folder = out / mark_of(case, k)
                folder.mkdir(parents=True, exist_ok=True)
                on = sheet[a:b, c:d]
                panels = {
                    f"published read auc {row.get('published', {}).get('auc', 0):.3f}": ink_panel(
                        published[a:b, c:d], on
                    ),
                    f"our read at the face auc {row.get('one', {}).get('auc', 0):.3f} shape {row.get('one', {}).get('shape', 0):.2f}": ink_panel(
                        np.nan_to_num(ours[k][0]), covered
                    ),
                    f"our five-depth mean auc {row.get('mean', {}).get('auc', 0):.3f} shape {row.get('mean', {}).get('shape', 0):.2f}": ink_panel(
                        np.nan_to_num(ours[k][1]), covered
                    ),
                    f"our stack at the published depth shape {row.get('oracle', {}).get('shape', 0):.2f}": ink_panel(
                        np.nan_to_num(ours[k][2]), covered
                    ),
                    "published CT at the surface": ct_panel(published_ct[a:b, c:d], on),
                    "our CT at the surface": ct_panel(np.nan_to_num(ours[k][3]), covered),
                    "hand label": ink_panel(np.where(label[a:b, c:d], 1.0, 0.0), on),
                }
                for n, (title, gray) in enumerate(panels.items()):
                    Image.fromarray(gray).save(folder / f"{n}-{title.replace(' ', '-')}.png", compress_level=3)
    OUT.mkdir(parents=True, exist_ok=True)
    study = f"-{windows_root.parent.name}" if root else ""
    (OUT / f"scored-{kind}-{route}-{model}{study}.json").write_text(json.dumps(rows, indent=1))
    if out:
        panel_pages.write(out, f"Known letters on our {route} mesh, read by {model}")
        print("page", out / "index.html", flush=True)
    return rows


def mark_of(case, letter):
    """A letter's name: its scroll and segment, its window for a fine-map case, and its number."""
    where = case["case_id"] if case.get("truth") == "fine-map" else f"{case['sample']}-{case['segment']}"
    return f"{where}-letter-{letter:03d}"


def reviewed():
    """Every verdict given so far, by mark."""
    path = OUT / "review.json"
    return json.loads(path.read_text()) if path.exists() else {}


def review(mark, verdict, why):
    """One mark's verdict by eye, accept or reject, with the reason."""
    if verdict not in ("accept", "reject"):
        raise SystemExit("verdict is accept or reject")
    found = reviewed()
    found[mark] = {"verdict": verdict, "why": why, "holdout": mark.split("-")[0] in HOLDOUT_SCROLLS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review.json").write_text(json.dumps(found, indent=1, sort_keys=True))


def surface_of(stack, face):
    """The mean of the three layers about where a read window puts the sheet's surface."""
    first = face + PAST_FACE
    return np.asarray(stack[first + SURFACE_LAYERS[0] : first + SURFACE_LAYERS[1]], dtype=np.float32).mean(axis=0)


def ct_panel(image, sheet):
    """Raw values stretched between their 1st and 99th percentiles on the sheet, black off it."""
    values = image[sheet > 0]
    low, high = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
    return np.where(sheet > 0, np.clip((image - low) / max(high - low, 1e-6), 0, 1) * 255, 0).astype(np.uint8)


def gaps_filled(canvas):
    """Pixels the mapping onto the published frame skipped, within GAP_PIXELS of read ones, given their nearest read value."""
    missing = np.isnan(canvas[0])
    if missing.all():
        return canvas, ~missing
    _, (rows, columns) = ni.distance_transform_edt(missing, return_indices=True)
    covered = ni.binary_closing(~missing, iterations=GAP_PIXELS) | ~missing
    filled = canvas[:, rows, columns]
    filled[:, ~covered] = np.nan
    return filled, covered


if __name__ == "__main__":
    {"census": census, "scored": scored, "review": review}[sys.argv[1]](*sys.argv[2:])
