"""A reader scored only where the hand label and the fine-scan map agree, on published surfaces and our own pieces of a left-out scroll, beside the published read carried onto our pieces."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ni

from calibration.ink.ink_yield import rank_auc
from calibration.ink.stroke_widths import fine_map
from ink.datasets.labelled_cases import load_truth
from ink.datasets.own_mesh_labels import MOST_LAYER_OFFSET, kept_windows, published_stack, truth_of
from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of
from ink.model.stock_model import SURE
from ink.model.turned_reader import named_reader
from ink.training.model_training import SURE_BLANK, SURE_INK
from storage.data_roots import LABELLED_CASE_STACKS, LABELLED_CASES, OWN_MESH_LABELS, calibration_output

OUT = calibration_output("agreed_truth")
ROUTES = ("level_route", "grown_route")
RIM_MM, HALO_MM = 0.1, 0.3
EVERY = 2
BLOCK, LEAST_BLOCK_PIXELS, LEAST_BLOCK_CORRELATION = 48, 400, 0.15
LEAST_INK = 2000
MOVES = range(-2, 3)
ORACLE = True


def agreed(case):
    """Agreed ink, agreed blank, and agreed blank within a halo of the writing, over a case's box, with the case's stack; a published window's fine map is its own agreement."""
    if case.get("truth") == "fine-map":
        stack = published_stack(case)
        ink, known = truth_of(case)
        known &= np.asarray(stack[stack.shape[0] // 2]) > 0
        away = ni.distance_transform_edt(~ink) * case["voxel_um"] / 1000
        blank = known & ~ink
        return stack, ink & known, blank, blank & (away <= HALO_MM)
    stack = np.load(LABELLED_CASE_STACKS / f"{case['case_id']}.npy", mmap_mode="r")
    label, known = load_truth(case)
    fine = fine_map(case)
    height, width = (min(a.shape[i] for a in (stack[0], label, fine)) for i in (0, 1))
    stack, label, known, fine = (
        stack[:, :height, :width],
        label[:height, :width],
        known[:height, :width],
        fine[:height, :width],
    )
    known &= np.asarray(stack[stack.shape[0] // 2]) > 0
    away = ni.distance_transform_edt(~label) * case["voxel_um"] / 1000
    ink = known & label & (fine >= SURE_INK)
    blank = known & (away > RIM_MM) & (fine <= SURE_BLANK) & (fine > 0)
    return stack, ink, blank, blank & (away <= HALO_MM)


def scores(read, ink, blank, halo):
    """AUC of agreed ink against all agreed blank and against blank near writing, and the shares each calls sure ink."""
    if ink.sum() < 100 or blank.sum() < 100:
        return None
    found = {
        "ink": int(ink.sum()),
        "blank": int(blank.sum()),
        "auc": rank_auc(read[ink][::EVERY], read[blank][::EVERY]),
        "auc_halo": rank_auc(read[ink][::EVERY], read[halo][::EVERY]) if halo.sum() >= 100 else float("nan"),
        "ink_sure": float((read[ink] >= SURE).mean()),
        "blank_sure": float((read[blank] >= SURE).mean()),
        "halo_sure": float((read[halo] >= SURE).mean()) if halo.any() else float("nan"),
    }
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in found.items()}


def shifted_read(reader, stack, shift, face=None):
    """A read of the production window, its face the piece's when known, moved by whole layers and widened evenly on both sides for a reader of more layers."""
    depth = getattr(reader, "depth", DEPTH)
    face = face_layer_of(stack) if face is None else face
    first = min(max(face + PAST_FACE + shift - (depth - DEPTH) // 2, 0), stack.shape[0] - depth)
    return reader.read(np.ascontiguousarray(stack[first : first + depth]))


def offset_field(own, published, ys, xs, rows, columns):
    """Layer offset of our stack against the published one in blocks of the window, the published stack as the only reference."""
    key = (ys // BLOCK) * 100000 + xs // BLOCK
    a_all, b_all = own[:, ys, xs].astype(np.float32), published[:, rows, columns].astype(np.float32)
    field = np.full(((own.shape[1] + BLOCK - 1) // BLOCK, (own.shape[2] + BLOCK - 1) // BLOCK), np.nan, np.float32)
    for k in np.unique(key):
        here = key == k
        if here.sum() < LEAST_BLOCK_PIXELS:
            continue
        a, b = (v - v.mean(1, keepdims=True) for v in (a_all[:, here], b_all[:, here]))
        a, b = a / np.maximum(a.std(1, keepdims=True), 1e-6), b / np.maximum(b.std(1, keepdims=True), 1e-6)
        table = a @ b.T / a.shape[1]
        best = max((float(np.mean(np.diagonal(table, o))), o) for o in range(-12, 13))
        if best[0] >= LEAST_BLOCK_CORRELATION:
            field[k // 100000, k % 100000] = best[1]
    seen = ~np.isnan(field)
    if not seen.any():
        return None
    nearest = ni.distance_transform_edt(~seen, return_distances=False, return_indices=True)
    field = ni.gaussian_filter(field[nearest[0], nearest[1]], 1.0)
    grid = np.meshgrid(*(np.arange(n) / BLOCK - 0.5 for n in own.shape[1:]), indexing="ij")
    return ni.map_coordinates(field, grid, order=1, mode="nearest")


def oracle_read(reader, own, published, offsets):
    """A read of our stack moved layer by layer onto the published surface's depth, an upper bound for any local depth fix."""
    first = face_layer_of(published) + PAST_FACE
    layers = first + np.arange(DEPTH, dtype=np.float32)[:, None, None] - offsets[None]
    low = np.clip(np.floor(layers).astype(np.int64), 0, own.shape[0] - 2)
    part = np.clip(layers - low, 0, 1)
    cube = np.asarray(own, dtype=np.float32)
    moved = np.take_along_axis(cube, low, 0) * (1 - part) + np.take_along_axis(cube, low + 1, 0) * part
    return reader.read(np.ascontiguousarray(np.where(cube[own.shape[0] // 2] > 0, moved, 0).round().astype(np.uint8)))


def own_windows(sample, route, root=OWN_MESH_LABELS):
    """Our own pieces of a scroll whose raw values place them on the labelled sheet, with their published pixels."""
    for path, row, labels in kept_windows(sample, route, root):
        if abs(row["offset"]) > MOST_LAYER_OFFSET:
            continue
        stack = np.load(path, mmap_mode="r")
        yield path.stem, row["case"], stack, labels["published"], row["face"] if "face" in row else face_layer_of(stack)


def weighted(rows):
    """Every score's mean over windows with enough agreed ink, weighted by their agreed ink, the scarcer class."""
    rows = [row for row in rows if row["ink"] >= LEAST_INK]
    if not rows:
        return {"windows": 0}
    weights = np.array([row["ink"] for row in rows], dtype=float)
    keys = [key for key in rows[0] if key not in ("name", "ink", "blank", "shift")]
    table = np.array([[row[key] for key in keys] for row in rows], dtype=float)
    seen = ~np.isnan(table)
    means = (np.where(seen, table, 0) * weights[:, None]).sum(0) / np.maximum((seen * weights[:, None]).sum(0), 1)
    return {key: round(float(value), 4) for key, value in zip(keys, means)} | {"windows": len(rows)}


def run(sample, out, shifts, paths, root=OWN_MESH_LABELS):
    """Each model on the scroll's published surfaces and on our own pieces of both routes at each layer shift."""
    cases = {
        json.loads(p.read_text())["case_id"]: json.loads(p.read_text())
        for p in sorted(LABELLED_CASES.glob(f"{sample}-*/case.json"))
        if 7 <= json.loads(p.read_text())["voxel_um"] <= 10
        and (LABELLED_CASE_STACKS / f"{json.loads(p.read_text())['case_id']}.npy").exists()
    }
    truths = {name: agreed(case) for name, case in cases.items()}
    shifts = [int(s) for s in shifts.split(",")]
    kept = {"sample": sample, "rim_mm": RIM_MM, "halo_mm": HALO_MM, "shifts": shifts}
    for path in paths:
        model, reader = named_reader(path)
        rows = {}
        published_reads = {name: shifted_read(reader, truth[0], 0) for name, truth in truths.items()}
        for name, (_, ink, blank, halo) in truths.items():
            found = scores(published_reads[name], ink, blank, halo)
            if found:
                rows.setdefault("published", []).append({"name": name, "shift": 0} | found)
        for route in ROUTES:
            for name, case_id, stack, (ys, xs, at_row, at_column), face in own_windows(sample, route, root):
                _, ink, blank, halo = truths[case_id]
                inside = (at_row < ink.shape[0]) & (at_column < ink.shape[1])
                ys, xs, at_row, at_column = ys[inside], xs[inside], at_row[inside], at_column[inside]
                masks = [m[at_row, at_column] for m in (ink, blank, halo)]
                carried = scores(published_reads[case_id][at_row, at_column], *masks)
                if carried:
                    rows.setdefault(f"{route} published read carried", []).append({"name": name, "shift": 0} | carried)
                reads = []
                for shift in shifts:
                    reads.append(shifted_read(reader, stack, shift, face))
                    found = scores(reads[-1][ys, xs], *masks)
                    if found:
                        rows.setdefault(f"{route} shift {shift:+d}", []).append({"name": name, "shift": shift} | found)
                joined = {"most over shifts": lambda r: r.max(0), "mean over shifts": lambda r: r.mean(0)}
                if shifts == list(range(-max(shifts), max(shifts) + 1)) and len(shifts) > 1:
                    reach = max(shifts)
                    joined |= {
                        f"mean over shifts within {k}": lambda r, k=k, reach=reach: r[reach - k : reach + k + 1].mean(0)
                        for k in range(1, reach)
                    }
                    weights = np.exp(-0.5 * (np.array(shifts) / max(reach / 2, 1)) ** 2)[:, None, None]
                    joined["weighted mean over shifts"] = lambda r, w=weights: (r * w).sum(0) / w.sum()
                    joined["mean over every other shift"] = lambda r: r[::2].mean(0)
                for kind, join in joined.items() if len(reads) > 1 else ():
                    found = scores(join(np.stack(reads))[ys, xs], *masks)
                    if found:
                        rows.setdefault(f"{route} {kind}", []).append({"name": name, "shift": 0} | found)
                if ORACLE and getattr(reader, "depth", DEPTH) == DEPTH:
                    offsets = offset_field(stack, truths[case_id][0], ys, xs, at_row, at_column)
                    found = offsets is not None and scores(
                        oracle_read(reader, stack, truths[case_id][0], offsets)[ys, xs], *masks
                    )
                    if found:
                        spread = float(np.std(offsets[ys, xs]))
                        found |= {"offset_sd": round(spread, 3), "face": int(face)}
                        rows.setdefault(f"{route} oracle depth", []).append({"name": name, "shift": 0} | found)
                    if offsets is not None and len(reads) > 1:
                        wanted = face_layer_of(truths[case_id][0]) - float(np.median(offsets[ys, xs])) - face
                        pick = int(np.clip(round(wanted), shifts[0], shifts[-1])) - shifts[0]
                        found = scores(reads[pick][ys, xs], *masks)
                        if found:
                            rows.setdefault(f"{route} window oracle", []).append({"name": name, "shift": pick} | found)
        for group, group_rows in rows.items():
            kept[f"{model} {group}"] = {"rows": group_rows} | weighted(group_rows)
            print(model, group, weighted(group_rows), flush=True)
        del reader
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(kept, indent=1))


def placement(sample, out, path, *folders):
    """Where carried labels sit best: one read of each own window scored with our pixels moved against the labels by whole pixels, per folder of windows."""
    model, reader = named_reader(path)
    cases = {
        json.loads(p.read_text())["case_id"]: json.loads(p.read_text())
        for p in sorted(LABELLED_CASES.glob(f"{sample}-*/case.json"))
        if 7 <= json.loads(p.read_text())["voxel_um"] <= 10
    }
    truths = {name: agreed(case) for name, case in cases.items()}
    kept = {"sample": sample, "model": model}
    for folder in folders:
        for route in ROUTES:
            rows = {}
            for name, case_id, stack, (ys, xs, at_row, at_column), face in own_windows(folder, route):
                _, ink, blank, halo = truths[case_id]
                inside = (at_row < ink.shape[0]) & (at_column < ink.shape[1])
                ys, xs, at_row, at_column = ys[inside], xs[inside], at_row[inside], at_column[inside]
                read = shifted_read(reader, stack, 0, face)
                for dy in MOVES:
                    for dx in MOVES:
                        on = (ys + dy >= 0) & (ys + dy < read.shape[0]) & (xs + dx >= 0) & (xs + dx < read.shape[1])
                        masks = [m[at_row[on], at_column[on]] for m in (ink, blank, halo)]
                        found = scores(read[ys[on] + dy, xs[on] + dx], *masks)
                        if found:
                            rows.setdefault((dy, dx), []).append({"name": name, "shift": 0} | found)
            table = {f"{dy:+d},{dx:+d}": weighted(r)["auc"] for (dy, dx), r in rows.items() if weighted(r)["windows"]}
            kept[f"{folder} {route}"] = table
            print(folder, route, "best", max(table, key=table.get), json.dumps(table), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(kept, indent=1))


def summarized(path):
    """A kept output's means taken again from its rows by the current rule."""
    kept = json.loads(Path(path).read_text())
    for key, value in kept.items():
        if isinstance(value, dict) and "rows" in value:
            kept[key] = {"rows": value["rows"]} | weighted(value["rows"])
    Path(path).write_text(json.dumps(kept, indent=1))
    return kept


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        for path in sys.argv[2:]:
            summarized(path)
    elif sys.argv[1] == "placement":
        placement(*sys.argv[2:])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
