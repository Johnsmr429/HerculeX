"""The one scan each scroll is surveyed on, in the order of the prizes."""

import json
import math
import urllib.request

from scans.catalog.scan_catalog import (
    FIRST_LETTERS,
    GRAND_PRIZE,
    load,
    survey_volumes,
)

SHEET_UM = 18.7
FINE_UM = (2.0, 2.6)
ENERGY_STEP = 10
NOT_SCROLLS = {"fragment"}


def height_of(row):
    url = f"{row['roots']['ome-zarr']}/{row['products']['ome-zarr'].strip('/')}/0/.zarray"
    return json.load(urllib.request.urlopen(url, timeout=60))["shape"][0] * row["voxel_um"]


def sheet_level(voxel_um):
    return max(0, round(math.log2(SHEET_UM / voxel_um)))


def prize_order(name):
    short = name.removeprefix("PHerc")
    return (short not in GRAND_PRIZE, short not in FIRST_LETTERS, name)


def chosen_scans(samples=()):
    """A fine low-energy scan where one exists, else the finest scan."""
    volumes = {r["volume"]: r for r in survey_volumes(load())}
    scans = {}
    for row in volumes.values():
        if (
            (samples and row["sample"] not in samples)
            or row["products"].get("ome-zarr") is None
            or row["voxel_um"] is None
            or row["type"] in NOT_SCROLLS
            or row["voxel_um"] > 12
        ):
            continue
        scans.setdefault(row["sample"], []).append(row)
    chosen = {}
    for name, rows in scans.items():
        fine = [r for r in rows if FINE_UM[0] <= r["voxel_um"] <= FINE_UM[1]]
        if fine:
            energy = min(round((r["energy_kev"] or 999) / ENERGY_STEP) for r in fine)
            fine = [r for r in fine if round((r["energy_kev"] or 999) / ENERGY_STEP) == energy]
            chosen[name] = max(fine, key=lambda r: (height_of(r), r["volume"]))
        else:
            chosen[name] = min(rows, key=lambda r: (r["voxel_um"], r["volume"]))
    return [chosen[name] for name in sorted(chosen, key=prize_order)]
