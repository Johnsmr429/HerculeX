"""The training recipe with one labelled scroll left out and scored on it."""

import json
import sys
import time

import numpy as np
from PIL import Image

from calibration.ink.ink_yield import rank_auc
from calibration.ink.invention_guard import scrambled, share, window
from calibration.ink.own_mesh_windows import scored as own_mesh_scored
from calibration.ink.stock_ink_scoring import cases
from calibration.ink.taught_model_controls import crop_of
from calibration.ink.tile_reader_study import auc_of
from ink.model.tile_reader import TileReader
from ink.training.model_training import taught
from storage.data_roots import INK_RECORDS, PUBLISHED_PAIRS

RECORD = INK_RECORDS / "ink-teaching-runs.jsonl"
WHOLE_WINDOWS = 4
LEFT_OUT = ("PHerc0841", "PHerc0500P2", "PHerc0009B")


def whole_windows(reader, left_out):
    """AUC against the published map of the same surface over whole windows, not only the pixels near labelled letters."""
    scores = []
    for path in sorted((PUBLISHED_PAIRS / left_out).glob("*.npy"))[:WHOLE_WINDOWS]:
        cube = np.load(path, mmap_mode="r")
        image = np.asarray(Image.open(path.with_suffix(".png")))
        read = reader.read(np.ascontiguousarray(window(cube)))
        ink, blank = image >= 160, (image <= 64) & (image > 0) & (read > 0)
        if ink.sum() > 1000:
            scores.append(rank_auc(read[ink][::7], read[blank][::37]))
    return float(np.mean(scores)) if scores else None


def scored(path, left_out):
    """AUC on each labelled crop of the left-out scroll as the sheet is read, and of the mean of three depth windows."""
    reader = TileReader(path)
    rng = np.random.default_rng(5)
    rows = []
    for case in cases():
        if case["sample"] != left_out:
            continue
        cube, label, valid = crop_of(case)
        maps = {o: reader.read(window(cube, o)) for o in (0, 4, 2)}
        fake = reader.read(window(scrambled(cube, rng)), fixed=True)
        far = reader.read(window(cube[::-1]))
        sheet = cube.max(axis=0) > 0
        rows.append(
            {
                "auc2": auc_of(maps[2], label, valid),
                "auc": auc_of(np.mean(list(maps.values()), axis=0), label, valid),
                "far": auc_of(far, label, valid),
                "sure": share(maps[2], sheet),
                "sure_far": share(far, sheet),
                "sure_scrambled": share(fake, sheet),
                "sure_blank": share(maps[2], valid & ~label),
            }
        )
    return (
        {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        | {"crops": rows, "whole": whole_windows(reader, left_out)}
        | own_mesh_scored(path, left_out)
    )


def run(steps):
    for left_out in LEFT_OUT:
        start = time.time()
        path = taught(f"without-{left_out}", left_out, left_out, 42, steps)
        score = scored(path, left_out)
        with RECORD.open("a") as log:
            log.write(
                json.dumps({"name": path.stem, "steps": steps, "minutes": round((time.time() - start) / 60, 1)} | score)
                + "\n"
            )
        print(
            left_out,
            "AUC",
            round(score["auc2"], 4),
            "whole",
            score["whole"] and round(score["whole"], 3),
            "far",
            round(score["far"], 3),
            "sure",
            round(score["sure"], 3),
            round(score["sure_far"], 3),
            flush=True,
        )


if __name__ == "__main__":
    run(int(sys.argv[1]))
