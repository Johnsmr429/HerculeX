"""Readable area of a scan in continuous areas of 4 cm² or more."""

import json
import multiprocessing
import sys

import numpy as np

from sampling import column_sampling as sampler
from storage.data_roots import ROUTE_OUTPUTS, stage_folder

PROCESSES = 4
SIZES_CM2 = (4, 1)


def mesh_areas(folder):
    """A piece's area in mm2 as a mesh, and with its gaps bridged."""
    xyz, meta = sampler.load_surface(folder)
    cell = (meta["stride"] * meta["voxel_um"] / 1000) ** 2
    return (xyz[0] > 0).sum() * cell, (sampler.bridged_mesh(xyz, meta["stride"])[0] > 0).sum() * cell


def areas_by_mosaic(sample, volume):
    """Rows of (locked, mesh, bridged) in cm2, one for each continuous area of a scan's layout."""
    base = ROUTE_OUTPUTS / sample / volume
    placed = json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]
    folders = [base / "pieces" / p["unit"] / f"piece-{p['piece']:04d}" for p in placed]
    with multiprocessing.get_context("spawn").Pool(PROCESSES) as pool:
        meshes = pool.map(mesh_areas, folders, chunksize=32)
    rows = {}
    for p, (mesh, bridged) in zip(placed, meshes):
        row = rows.setdefault((p["group"], round(p["moved_from_joint_mm"])), [0.0, 0.0, 0.0])
        row[0] += p["area"]
        row[1] += mesh
        row[2] += bridged
    return np.array(list(rows.values())) / 100


def run(sample, volume):
    areas = areas_by_mosaic(sample, volume)
    for name, values in zip(("locked", "mesh", "bridged"), areas.T):
        sizes = {f"in areas of {size} cm2 or more": round(float(values[values >= size].sum())) for size in SIZES_CM2}
        print(
            name,
            "total",
            round(float(values.sum())),
            sizes,
            "areas",
            int((values >= SIZES_CM2[0]).sum()),
            "largest",
            round(float(values.max()), 1),
            flush=True,
        )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
