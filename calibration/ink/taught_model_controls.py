"""Whether a model taught on one scroll's strokes knows more or less about ink."""

import json
import sys
from pathlib import Path

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.stock_ink_scoring import cases
from ink.datasets.labelled_cases import load_truth
from ink.model.stock_model import InkModel
from sampling.piece_images import ink_window
from storage.brick_store import BrickStore
from storage.data_roots import LABELLED_CASES, calibration_output


def crop_of(case):
    record = json.loads((calibration_output("stock_ink_scoring") / case["case_id"] / "record.json").read_text())
    y0, y1, x0, x1 = record["box"]
    ink, known = load_truth(case)
    top, left = case["box"][0], case["box"][2]
    store = BrickStore(LABELLED_CASES / case["case_id"] / "layers")
    cube = store.read((0, store.shape[0]), (y0, y1), (x0, x1))
    return (
        cube,
        ink[y0 - top : y1 - top, x0 - left : x1 - left],
        known[y0 - top : y1 - top, x0 - left : x1 - left] & (cube.max(axis=0) > 0),
    )


def run(paths):
    models = {"stock": InkModel(names=("hybrid_3d2d-seed42/step-075000.pth",))} | {
        Path(p).stem: InkModel(names=("hybrid_3d2d-seed42/step-075000.pth",), weights=[p]) for p in paths
    }
    rows = {name: [] for name in models}
    for case in cases():
        cube, label, valid = crop_of(case)
        layers = np.ascontiguousarray(ink_window(cube, 17))
        scores = {}
        for name, model in models.items():
            read = model.read(layers)
            scores[name] = rank_auc(read[valid & label], read[valid & ~label])
            rows[name].append(scores[name])
        print(case["sample"], {k: round(v, 3) for k, v in scores.items()}, flush=True)
    print({k: round(float(np.mean(v)), 4) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
