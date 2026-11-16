"""Whether a taught model invents ink on structureless papyrus and blank faces."""

import json
import sys

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.stock_ink_scoring import cases
from calibration.ink.taught_model_controls import crop_of
from ink.model.input_stack import face_layer_of, scrambled
from ink.model.stock_model import SURE, InkModel
from storage.data_roots import TARGET_STACKS

SIDE = 1024
OFFSET = 2


def window(cube, offset=OFFSET):
    return np.ascontiguousarray(cube[min(face_layer_of(cube) + offset, cube.shape[0] - 17) :][:17])


def share(read, where):
    return round(float((read[where] >= SURE).mean()), 4)


def run(weights, patch, pieces):
    model = InkModel(names=("hybrid_3d2d-seed42/step-075000.pth",), weights=weights)
    rng = np.random.default_rng(5)
    step = patch // 2
    for case in cases():
        cube, label, valid = crop_of(case)
        cube = cube[:, :SIDE, :SIDE]
        label, valid = label[:SIDE, :SIDE], valid[:SIDE, :SIDE]
        real = model.read(window(cube), patch=patch, step=step)
        far = model.read(window(cube[::-1]), patch=patch, step=step)
        fake = model.read(window(scrambled(cube, rng)), patch=patch, step=step)
        print(
            case["sample"],
            "sure real",
            share(real, real > 0),
            "far",
            share(far, far > 0),
            "scrambled",
            share(fake, fake > 0),
            "blank",
            share(real, valid & ~label),
            "AUC",
            round(rank_auc(real[valid & label], real[valid & ~label]), 3),
            flush=True,
        )
    for unit, piece in pieces:
        stack = np.load(TARGET_STACKS / "PHerc0211" / unit / f"piece-{piece:04d}.npy", mmap_mode="r")
        h, w = stack.shape[1:]
        y, x = max((h - SIDE) // 2, 0), max((w - SIDE) // 2, 0)
        cube = np.asarray(stack[:, y : y + SIDE, x : x + SIDE])
        real = model.read(window(cube), patch=patch, step=step)
        far = model.read(window(cube[::-1]), patch=patch, step=step)
        fake = model.read(window(scrambled(cube, rng)), patch=patch, step=step)
        print(
            "PHerc0211",
            unit,
            piece,
            "sure real",
            share(real, real > 0),
            "far",
            share(far, far > 0),
            "scrambled",
            share(fake, fake > 0),
            flush=True,
        )


if __name__ == "__main__":
    run(json.loads(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3]))
