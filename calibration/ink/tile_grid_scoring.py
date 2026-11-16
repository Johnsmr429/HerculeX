"""A model's AUC over shifted tile grids, since one grid is one draw."""

import json
import sys
from pathlib import Path

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.invention_guard import window
from calibration.ink.own_mesh_windows import SET_ASIDE, WINDOWS
from calibration.ink.stock_ink_scoring import cases
from calibration.ink.taught_model_controls import crop_of
from ink.model.neural_engine import EngineReader
from ink.training.model_training import PAST_FACE, face_layer_of
from sampling import column_sampling as sampler
from sampling.fibre_faces import recto_face

OFFSETS = ((0, 0), (128, 128), (256, 256), (0, 256), (256, 0))


def read_at(reader, stack, offset):
    """A read with its tile grid moved by a mirrored strip along the stack's top and left."""
    top, left = offset
    moved = np.pad(stack, ((0, 0), (top, 0), (left, 0)), mode="symmetric") if top or left else stack
    return reader.read(np.ascontiguousarray(moved), fixed=True)[top:, left:]


def own_mesh(sample):
    """Name, stack, scorer and weight of a scroll's own-mesh windows, weighed by labelled points."""
    for path in sorted((WINDOWS / sampler.SAMPLING / sample).glob("*.npz")):
        if path.stem not in SET_ASIDE:
            data = np.load(path)
            stack, rows, columns, ink = data["stack"], data["rows"], data["columns"], data["ink"]
            turned = stack[::-1] if recto_face(stack)["face"] == "far" else stack
            first = min(max(face_layer_of(turned) + PAST_FACE, 0), turned.shape[0] - 17)
            yield (
                path.stem,
                np.ascontiguousarray(turned[first : first + 17]),
                lambda read, r=rows, c=columns, k=ink: rank_auc(read[r, c][k], read[r, c][~k]),
                len(ink),
            )


def published(sample):
    """The same for a scroll's published crops, each weighing the same."""
    for case in cases():
        if case["sample"] == sample:
            cube, label, valid = crop_of(case)
            yield (
                case["case_id"],
                window(cube),
                lambda read, a=label, v=valid: rank_auc(read[v & a][::3], read[v & ~a][::3]),
                1,
            )


def scored(reader, windows):
    """Each window's AUC at every offset, its statistics fixed from the stack as it stands."""
    rows = []
    for name, stack, score, weight in windows:
        reader.fix_to(stack)
        rows.append(
            {
                "name": name,
                "weight": weight,
                "auc": [round(float(score(read_at(reader, stack, offset))), 4) for offset in OFFSETS],
            }
        )
        print(rows[-1], flush=True)
    return rows


def summary(rows):
    """Expected AUC, the set's AUC at each grid and its spread, and the mean spread of one window."""
    weights = np.array([row["weight"] for row in rows], dtype=float)
    auc = np.array([row["auc"] for row in rows])
    by_grid = weights @ auc / weights.sum()
    return {
        "expected": round(float(by_grid.mean()), 4),
        "by_grid": by_grid.round(4).tolist(),
        "set_spread": round(float(by_grid.std()), 4),
        "window_spread": round(float(auc.std(axis=1).mean()), 4),
    }


def run(weights, out, samples):
    reader = EngineReader(weights)
    kept = {"weights": str(weights), "samples": samples, "offsets": OFFSETS}
    for name, windows in (("own_mesh", own_mesh), ("published", published)):
        rows = [row for sample in samples for row in scored(reader, windows(sample))]
        kept[name] = {"rows": rows} | (summary(rows) if rows else {})
        print(name, {k: v for k, v in kept[name].items() if k != "rows"}, flush=True)
    Path(out).write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
