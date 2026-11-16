"""Every sheet control unwrapped and scored: contiguous correct area against wrong-wrap rate."""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from calibration.unwrap import sheet_wrap_score
from calibration.unwrap.sheet_wrap_truth import OUT
from runner.stage_queue import record_of
from storage.data_roots import SHEET_CONTROLS, SURFACE_RECORDS
from unwrapping.level import level_pieces, piece_growth

WORKERS = int(os.environ.get("HERCULEX_STUDY_WORKERS", "3"))


def controls():
    """Every control with truth, smallest first."""
    found = [p for p in SHEET_CONTROLS.iterdir() if p.is_dir() and (OUT / f"{p.name}.npz").exists()]
    return sorted(found, key=lambda p: record_of(p)["nodes"])


def truth_of(control):
    tier_a = OUT / f"{control.name}-tier-a.npz"
    return sheet_wrap_score.load(f"{control.name}-tier-a" if tier_a.exists() else control.name)


def score(control):
    record = record_of(control)
    start = time.time()
    nodes, theta, _first, _grid, pairs, field, _ = level_pieces.prepare([control])
    root, potential, _, _, _, _ = level_pieces.grow_pieces(nodes, theta, pairs, field)
    seconds = time.time() - start
    _ids, piece = np.unique(root, return_inverse=True)
    area = np.bincount(piece, weights=sheet_wrap_score.node_area_mm2(nodes, record["scan_voxel_um"], record["level"]))
    rows = sheet_wrap_score.score(truth_of(control), piece.astype(np.int32), potential + theta / (2 * np.pi), area)
    return {
        "control": control.name,
        "sample": record["sample"],
        "frame_costs": field.get("frame_costs"),
        "seconds": round(seconds, 1),
        "largest_cm2": round(float(area.max()) / 100, 1),
    } | rows


def set_trial(settings):
    """Trial values of the level route's constants (level_pieces' own, else piece_growth's), in each worker."""
    for name, value in settings:
        setattr(level_pieces if hasattr(level_pieces, name) else piece_growth, name, value)


def study(names=()):
    """Controls scored side by side, largest first so none is left running alone."""
    settings = [(name, float(value)) for name, value in (n.split("=") for n in names if "=" in n)]
    names = [n for n in names if "=" not in n]
    chosen = [c for c in controls() if not names or c.name in names or record_of(c)["sample"] in names]
    with ProcessPoolExecutor(
        WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=set_trial, initargs=(settings,)
    ) as pool:
        running = {control: pool.submit(score, control) for control in reversed(chosen)}
        rows = []
        for control in chosen:
            rows.append(running[control].result())
            print(json.dumps(rows[-1]), flush=True)
    if not names:
        (SURFACE_RECORDS / "wrap-study-patches.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    study(sys.argv[1:])
