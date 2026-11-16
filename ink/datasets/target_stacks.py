"""Layer stacks of the target's marked pieces, kept whole for training."""

import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

from sampling import column_sampling as sampler
from sampling.piece_images import axis_points, surface_of
from storage.brick_store import BrickStore
from storage.data_roots import ROUTE_OUTPUTS, TARGET_STACKS, ct_bricks

_store = None


def start_worker(store_folder):
    global _store
    _store = BrickStore(store_folder)


def write_stack(job):
    folder, out, axis = job
    start = time.time()
    xyz, meta = sampler.load_surface(folder)
    margined, normal = surface_of(xyz, meta, axis)
    cube = sampler.flatten(margined, meta["stride"], normal, _store).cube
    np.save(out.with_suffix(".part.npy"), cube)
    out.with_suffix(".part.npy").replace(out)
    return folder.name, cube.shape, round(time.time() - start, 1)


def run(sample, volume, wanted, bricks, processes=3):
    """A stack for each of some (unit, piece), from one store of bricks already here."""
    base = ROUTE_OUTPUTS / sample / volume
    axis = axis_points(sample, volume)
    jobs = []
    for unit, k in wanted:
        out = TARGET_STACKS / sample / unit / f"piece-{k:04d}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            jobs.append((base / "pieces" / unit / f"piece-{k:04d}", out, axis))
    with multiprocessing.get_context("spawn").Pool(processes, start_worker, (str(ct_bricks(sample, bricks)),)) as pool:
        for k, row in enumerate(pool.imap_unordered(write_stack, jobs)):
            print(k + 1, "of", len(jobs), row, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), sys.argv[4])
