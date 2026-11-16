"""Node labels scored against wrap truth."""

import numpy as np
from scipy import ndimage as ni

from calibration.unwrap.sheet_wrap_truth import OUT
from unwrapping.detection.kernel_bindings import CELL

USABLE_MM2 = 100.0
WINDOW_MM2 = 400.0
BLOB_MM2 = 16.0


def load(name):
    data = np.load(OUT / f"{name}.npz", allow_pickle=False)
    return {k: data[k] for k in data.files}


def node_area_mm2(nodes, voxel_um, level):
    """Sheet area a node stands for."""
    norm = np.sqrt(sum(nodes[c].astype(np.float32) ** 2 for c in ("nx", "ny", "nz")))
    spread = sum(np.abs(nodes[c].astype(np.float32)) for c in ("nx", "ny", "nz")) / np.maximum(norm, 1e-6)
    side = CELL * 2**level * voxel_um / 1000
    return (side**2 / np.maximum(spread, 1.0)).astype(np.float32)


def modal_labels(piece, delta, group, weights=None):
    """True where a vertex's winding offset is its (piece, group)'s commonest."""
    pair = piece.astype(np.int64) * (int(group.max()) + 1) + group
    order = np.lexsort((delta, pair))
    p, d = pair[order], delta[order]
    run = np.flatnonzero(np.diff(p) != 0) + 1
    starts = np.concatenate([[0], run])
    ends = np.append(run, len(p))
    change = np.flatnonzero((np.diff(p) != 0) | (np.diff(d) != 0)) + 1
    sub_start = np.concatenate([[0], change])
    sub_len = np.diff(np.append(sub_start, len(p)))
    owner = np.searchsorted(starts, sub_start, side="right") - 1
    best = np.zeros(len(starts), dtype=np.int64)
    np.maximum.at(best, owner, sub_len)
    is_best = sub_len == best[owner]
    first_best = np.full(len(starts), -1, dtype=np.int64)
    idx = np.flatnonzero(is_best)[::-1]
    first_best[owner[idx]] = idx
    mode = d[sub_start[first_best]]
    out = np.empty(len(p), dtype=bool)
    out[order] = d == np.repeat(mode, ends - starts)
    return out


def classify(truth, piece_of_node, winding, usable):
    """Per truth vertex."""
    cand = truth["candidates"]
    piece_c = np.where(cand >= 0, piece_of_node[np.maximum(cand, 0)], -1)
    ok = piece_c >= 0
    ok &= usable[np.maximum(piece_c, 0)]
    first = np.argmax(ok, axis=1)
    rows = np.arange(len(cand))
    covered = ok[rows, first]
    node = cand[rows, first]
    piece = np.where(covered, piece_c[rows, first], -1)
    delta = np.where(covered, np.round(truth["Wg"] - winding[np.maximum(node, 0)]), 0).astype(np.int64)
    label = np.full(len(cand), 2, dtype=np.int8)
    label[covered & truth["unresolved"]] = 3
    judged = covered & ~truth["unresolved"]
    if judged.any():
        right = modal_labels(piece[judged], delta[judged], truth["group"][judged].astype(np.int64))
        label[np.flatnonzero(judged)] = np.where(right, 0, 1)
    return label, piece


def wrong_in_blobs(truth, label):
    """Wrong vertices lying in contiguous wrong regions of at least BLOB_MM2 on their mesh's grid."""
    total, least = 0, BLOB_MM2 / float(truth["vertex_mm2"])
    for mesh in np.unique(truth["mesh"]):
        mine = truth["mesh"] == mesh
        if not (label[mine] == 1).any():
            continue
        rows, columns = truth["row"][mine], truth["col"][mine]
        grid = np.zeros((rows.max() + 2, columns.max() + 2), dtype=np.int8)
        grid[rows, columns] = label[mine] == 1
        parts, count = ni.label(grid, structure=np.ones((3, 3)))
        if count:
            sizes = np.bincount(parts[parts > 0])
            total += int(sizes[sizes >= least].sum())
    return total


def score(truth, piece_of_node, winding, piece_area_mm2):
    """Headline numbers of one labelling."""
    cell = float(truth["vertex_mm2"]) / 100
    rows = {}
    for name, floor in (("all", 0.0), ("usable", USABLE_MM2), ("window", WINDOW_MM2)):
        label, _piece = classify(truth, piece_of_node, winding, piece_area_mm2 >= floor)
        n = len(label)
        right, wrong = int((label == 0).sum()), int((label == 1).sum())
        rows[name] = {
            "correct": round(right / n, 4),
            "wrong": round(wrong / n, 4),
            "missing": round(float((label == 2).mean()), 4),
            "unjudged": round(float((label == 3).mean()), 4),
            "wrong_rate": round(wrong / max(right + wrong, 1), 4),
            "correct_cm2": round(right * cell, 1),
            "wrong_cm2": round(wrong * cell, 1),
            "claimed_cm2": round(float(piece_area_mm2[piece_area_mm2 >= floor].sum()) / 100, 1),
            "pieces": int((piece_area_mm2 >= floor).sum()),
        }
        if "row" in truth:
            rows[name]["blob_wrong_rate"] = round(wrong_in_blobs(truth, label) / max(right + wrong, 1), 4)
    return rows
