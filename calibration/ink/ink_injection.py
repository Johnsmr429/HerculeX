"""Labelled letters injected into the target's own stacks and recovered, layer by layer."""

import json
import sys

import numpy as np
from scipy import ndimage as ni

from calibration.ink.ink_yield import rank_auc
from ink.datasets.labelled_cases import richest_window
from ink.model.stock_model import InkModel
from ink.training.model_training import PATCH, Labelled, cases, face_layer_of
from sampling.piece_images import FACE_LAYER, ink_window
from storage.data_roots import TARGET_STACKS

EDGE_PX = 2.0
HIGH_PASS_PX = 60


def transplanted(target, source, mask, shift=(0, 0)):
    """A target stack with what source papyrus adds over itself under a mask."""
    face, source_face = face_layer_of(target) + FACE_LAYER, face_layer_of(source) + FACE_LAYER
    soft = np.roll(ni.gaussian_filter(mask.astype(np.float32), EDGE_PX), shift, axis=(0, 1))
    out = target.astype(np.float32)
    for layer in range(target.shape[0]):
        from_layer = layer - face + source_face
        if 0 <= from_layer < source.shape[0]:
            part = source[from_layer].astype(np.float32)
            out[layer] += np.roll(
                (part - ni.gaussian_filter(part, HIGH_PASS_PX)) * soft, (-shift[0], -shift[1]), axis=(0, 1)
            )
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def run(weights, sample, unit, piece, y0, x0, side):
    model = InkModel(weights=weights)
    labelled = Labelled(next(c for c in cases() if c["sample"] == sample))
    top, bottom, left, right = richest_window(labelled.ink, labelled.known, side)
    source = np.asarray(labelled.stack[:, top:bottom, left:right])
    mask = (labelled.ink & labelled.known)[top:bottom, left:right]
    target = np.asarray(
        np.load(TARGET_STACKS / "PHerc0211" / unit / f"piece-{piece:04d}.npy", mmap_mode="r")[
            :, y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]
        ]
    )
    sheet = target[target.shape[0] // 2] > 0
    bare = (ni.distance_transform_edt(~mask) > 25) & sheet
    shift = min(
        ((dy, dx) for dy in range(-400, 401, 50) for dx in range(-400, 401, 50) if abs(dy) + abs(dx) > 100),
        key=lambda d: (np.roll(mask, d, axis=(0, 1)) & mask).sum(),
    )
    for name, stack in (
        ("none", target),
        ("ink moved in", transplanted(target, source, mask)),
        ("bare moved in", transplanted(target, source, mask, shift)),
    ):
        read = model.read(np.ascontiguousarray(ink_window(stack, model.depth)), patch=PATCH, step=PATCH // 2)
        print(sample, name, "AUC", round(rank_auc(read[mask & sheet], read[bare]), 3), flush=True)


if __name__ == "__main__":
    run(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], *map(int, sys.argv[4:8]))
