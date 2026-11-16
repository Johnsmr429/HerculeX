"""Every scroll's coarsest useful level, about 75 µm per voxel."""

import math
import sys
from pathlib import Path

from scans.catalog.scan_catalog import load, survey_volumes
from scans.transfer.fetch_volume import fetch_level


def primary_volumes(samples):
    """Per sample the volume closest to 9 µm, the resolution of the prize scans."""
    best = {}
    for row in survey_volumes(samples):
        path = row["products"].get("ome-zarr")
        if path is None or row["voxel_um"] is None:
            continue
        rank = (abs(math.log(row["voxel_um"] / 9.0)), row["volume"])
        if row["sample"] not in best or rank < best[row["sample"]][0]:
            best[row["sample"]] = (rank, row, path)
    return {name: (row, path) for name, (_, row, path) in best.items()}


def fetch_atlas(out, target_um=75.0):
    for name, (row, path) in sorted(primary_volumes(load()).items()):
        level = max(0, round(math.log2(target_um / row["voxel_um"])))
        folder = Path(out) / f"{name}-{row['voxel_um']}um" / f"L{level}"
        print(name, row["type"], row["voxel_um"], "level", level, flush=True)
        try:
            fetch_level(path, level, folder, root=row["roots"]["ome-zarr"])
        except (OSError, ValueError) as error:
            print("failed", name, error, flush=True)


if __name__ == "__main__":
    fetch_atlas(sys.argv[1], *(float(v) for v in sys.argv[2:3]))
