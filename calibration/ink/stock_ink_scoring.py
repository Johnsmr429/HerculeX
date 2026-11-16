"""The production two-seed reader on labelled crops of scrolls absent from its training set."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calibration.ink.ink_yield import rank_auc, recall_at
from ink.datasets.labelled_cases import load_truth
from ink.model import stock_model
from runner import stage_queue
from sampling.piece_images import ink_window
from storage.brick_store import BrickStore
from storage.data_roots import INK_RECORDS, LABELLED_CASES, calibration_output

TRAINED = {"PHerc0139", "PHerc1667", "PHercParis4", "PHerc0814"}


def window(ink, known, side):
    """Most labelled ink among fixed 128-pixel crop positions, chosen before prediction."""
    h, w = ink.shape
    sh, sw = min(side, h), min(side, w)
    mass = np.pad((ink & known).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1, 0, 0)
    for y in sorted({*range(0, h - sh + 1, 128), h - sh}):
        for x in sorted({*range(0, w - sw + 1, 128), w - sw}):
            count = mass[y + sh, x + sw] - mass[y, x + sw] - mass[y + sh, x] + mass[y, x]
            if count > best[0]:
                best = (count, y, x)
    _, y, x = best
    return y, y + sh, x, x + sw


def cases():
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        if case["sample"] not in TRAINED and case["sample"].startswith("PHerc") and 7 <= case["voxel_um"] <= 10:
            yield case


def run(side):
    model = None
    rows = []
    for case in cases():
        folder = LABELLED_CASES / case["case_id"]
        sources = [
            Path(__file__),
            Path(stock_model.__file__),
            Path(ink_window.__code__.co_filename),
            folder / "case.json",
            folder / "ink.npy",
            folder / "supervised.npy",
        ]
        if (folder / "registration.json").exists():
            sources.append(folder / "registration.json")
        method = stage_queue.method_of(*sources) + str(side)
        out = calibration_output("stock_ink_scoring") / case["case_id"]
        if stage_queue.done(out, method):
            rows.append(stage_queue.record_of(out))
            continue
        ink, known = load_truth(case)
        top, bottom, left, right = window(ink, known, side)
        y0, _, x0, _ = case["box"]
        store = BrickStore(folder / "layers")
        cube = store.read((0, store.shape[0]), (y0 + top, y0 + bottom), (x0 + left, x0 + right))
        label = ink[top:bottom, left:right]
        valid = known[top:bottom, left:right] & (cube.max(axis=0) > 0)
        if min((valid & label).sum(), (valid & ~label).sum()) < 2000:
            raise ValueError(f"Too little labelled ink or bare surface: {case['case_id']}")
        if model is None:
            model = stock_model.InkModel()
            model.torch.set_num_threads(2)
        start = time.monotonic()
        prediction = model.read(ink_window(cube, model.depth))
        a, b = prediction[valid & label], prediction[valid & ~label]
        record = {
            "case": case["case_id"],
            "sample": case["sample"],
            "method": method,
            "voxel_um": case["voxel_um"],
            "box": [y0 + top, y0 + bottom, x0 + left, x0 + right],
            "ink_pixels": len(a),
            "bare_pixels": len(b),
            "auc": rank_auc(a, b),
            "recall_at_1pct_fpr": recall_at(a, b, 0.01),
            "recall_at_5pct_fpr": recall_at(a, b, 0.05),
            "labels_resampled": case["labels_resampled"],
            "registration": json.loads((folder / "registration.json").read_text())
            if (folder / "registration.json").exists()
            else None,
            "seconds": round(time.monotonic() - start, 2),
            "checkpoints": stock_model.CHECKPOINTS,
            "source_commit": stock_model.COMMIT,
            "device": str(model.device),
            "torch": model.torch.__version__,
            "depth_policy": "production ink_window; forward inward-normal stack; no label-selected depth or direction",
        }
        with stage_queue.building(out) as part:
            np.save(part / "probability.npy", prediction)
            stage_queue.finish(part, record)
        rows.append(record)
        print(case["case_id"], round(record["auc"], 4), record["seconds"], "s", flush=True)
    per_scroll = {
        sample: float(np.mean([r["auc"] for r in rows if r["sample"] == sample]))
        for sample in sorted({r["sample"] for r in rows})
    }
    result = {
        "cases": rows,
        "case_mean_auc": float(np.mean([r["auc"] for r in rows])),
        "per_scroll_auc": per_scroll,
        "scroll_mean_auc": float(np.mean(list(per_scroll.values()))),
        "training_scrolls": sorted(TRAINED),
        "limitations": "Label-rich crops of three unseen scrolls; resampled community annotations. Not whole-scroll coverage or a PHerc0211 accuracy estimate.",
    }
    (INK_RECORDS / "stock-ink-unseen.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=1536)
    run(parser.parse_args().side)
