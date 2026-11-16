"""A reader's mean over nearby depths of the windows taught on, kept as soft targets for a single-read student."""

import json
import sys
from pathlib import Path

import numpy as np

from calibration.ink.agreed_truth import shifted_read
from ink.model.tile_reader import TileReader
from storage.data_roots import (
    AGREED_TARGETS,
    DEPTH_MEAN_TARGETS,
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    OWN_MESH_LABELS,
    PUBLISHED_PAIRS,
    calibration_output,
)

SHIFTS = tuple(range(-4, 5))
LEFT_OUT = ("PHerc0841",)
PUBLISHED_A_SCROLL = 12
OWN_PIECES = calibration_output("surface_ink") / "stacks"


def path_of(kind, name):
    return DEPTH_MEAN_TARGETS / kind / f"{name}.npy"


def depth_mean(reader, stack):
    """The mean of a reader's maps over the production window moved by each shift, as bytes."""
    return np.round(np.mean([shifted_read(reader, stack, s) for s in SHIFTS], axis=0) * 255).astype(np.uint8)


def stacks():
    """(kind, name, stack) of every window of the taught scrolls: our own whole pieces and windows, labelled cases and the published windows richest in agreed ink."""
    for path in sorted(OWN_PIECES.glob("PHerc*.npy")):
        if path.stem.split("-")[0] not in LEFT_OUT:
            yield "own_pieces", path.stem, np.load(path, mmap_mode="r")
    for route in ("grown_route", "level_route"):
        for path in sorted((OWN_MESH_LABELS / route).glob("PHerc*/*.npy")):
            if path.parent.name.split("-")[0] not in LEFT_OUT:
                yield f"own_mesh_labels/{route}/{path.parent.name}", path.stem, np.load(path, mmap_mode="r")
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        stack = LABELLED_CASE_STACKS / f"{case['case_id']}.npy"
        if 7 <= case["voxel_um"] <= 10 and case["sample"] not in LEFT_OUT and stack.exists():
            yield "labelled_cases", case["case_id"], np.load(stack, mmap_mode="r")
    for folder in sorted(p for p in PUBLISHED_PAIRS.glob("PHerc*") if p.name not in LEFT_OUT):
        agreed = AGREED_TARGETS / "published_pairs" / folder.name
        chosen = sorted(
            (p for p in folder.glob("*.npy") if not p.name.endswith(".part.npy") and (agreed / p.name).exists()),
            key=lambda p: -int((np.load(agreed / p.name, mmap_mode="r") == 2).sum()),
        )[:PUBLISHED_A_SCROLL]
        for path in chosen:
            yield f"published_pairs/{folder.name}", path.stem, np.load(path, mmap_mode="r")


def build(weights):
    """Depth-mean maps of every taught window by one reader."""
    reader = TileReader(Path(weights))
    for kind, name, stack in stacks():
        out = path_of(kind, name)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out.with_suffix(".part.npy"), depth_mean(reader, stack))
        out.with_suffix(".part.npy").replace(out)
        print(kind, name, stack.shape, flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
