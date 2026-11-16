"""The tile reader against the patch reader it replaced, on left-out crops."""

import json

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.invention_guard import scrambled, share, window
from calibration.ink.stock_ink_scoring import cases
from calibration.ink.taught_model_controls import crop_of
from ink.model.stock_model import CHECKPOINTS, InkModel
from ink.model.tile_reader import TileReader
from ink.training.model_training import PATCH
from storage.data_roots import INK_RECORDS, PRODUCTION_WEIGHTS, TARGET_STACKS, TRAINED_WEIGHTS

LEFT_OUT = {"PHerc0841": "F-without-0841", "PHerc0500P2": "F-without-0500P2", "PHerc0009B": "F-without-0009B"}
TARGET_PIECES = (
    ("rows-0022-0034", 39),
    ("rows-0022-0034", 45),
    ("rows-0022-0034", 47),
    ("rows-0022-0034", 5),
    ("rows-0033-0045", 1),
)
SIDE = 1100


def auc_of(read, label, valid):
    return rank_auc(read[valid & label][::3], read[valid & ~label][::3])


def scored(sample, weights, rows, rng):
    tiles, patches = TileReader(weights), InkModel(names=tuple(CHECKPOINTS)[:1], weights=[weights])
    for case in cases():
        if case["sample"] != sample:
            continue
        cube, label, valid = crop_of(case)
        near, far, fake = window(cube), window(cube[::-1]), window(scrambled(cube, rng))
        sheet = near[8] > 0
        new, new_fake = tiles.read(near), tiles.read(fake, fixed=True)
        new_far = tiles.read(far)
        old, old_far, old_fake = (patches.read(s, patch=PATCH, step=PATCH // 2) for s in (near, far, fake))
        rows.append(
            {
                "sample": sample,
                "case": case["case_id"],
                "auc": auc_of(new, label, valid),
                "patch_auc": auc_of(old, label, valid),
                "sure": share(new, sheet),
                "far": share(new_far, sheet),
                "scrambled": share(new_fake, sheet),
                "blank": share(new, valid & ~label),
                "patch_sure": share(old, sheet),
                "patch_far": share(old_far, sheet),
                "patch_scrambled": share(old_fake, sheet),
                "patch_blank": share(old, valid & ~label),
            }
        )
        print(rows[-1], flush=True)


def target(rows, rng):
    tiles = TileReader(PRODUCTION_WEIGHTS / "stem-seed42.pth")
    for unit, piece in TARGET_PIECES:
        stack = np.load(TARGET_STACKS / "PHerc0211" / unit / f"piece-{piece:04d}.npy", mmap_mode="r")
        y, x = max((stack.shape[1] - SIDE) // 2, 0), max((stack.shape[2] - SIDE) // 2, 0)
        cube = np.asarray(stack[:, y : y + SIDE, x : x + SIDE])
        near, far, fake = window(cube), window(cube[::-1]), window(scrambled(cube, rng))
        sheet = near[8] > 0
        new, new_fake = tiles.read(near), tiles.read(fake, fixed=True)
        rows.append(
            {
                "unit": unit,
                "piece": piece,
                "sure": share(new, sheet),
                "far": share(tiles.read(far), sheet),
                "scrambled": share(new_fake, sheet),
            }
        )
        print(rows[-1], flush=True)


def run():
    rng = np.random.default_rng(5)
    crops, pieces = [], []
    for sample, name in LEFT_OUT.items():
        scored(sample, TRAINED_WEIGHTS / f"{name}.pth", crops, rng)
    target(pieces, rng)
    means = {k: float(np.mean([r[k] for r in crops])) for k in crops[0] if k not in ("sample", "case")}
    (INK_RECORDS / "tile-reader-study.json").write_text(
        json.dumps({"crops": crops, "target": pieces, "means": means}, indent=1)
    )
    print(means, flush=True)


if __name__ == "__main__":
    run()
