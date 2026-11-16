"""Contiguous area of every survey band under the level route, against published meshes."""

import json
import sys
import time

import numpy as np

from calibration.unwrap import sheet_wrap_score
from calibration.unwrap.sheet_controls import load_mesh, meshes
from runner.stage_queue import done, record_of
from storage.data_roots import NODE_ROUTE, SURFACE_RECORDS, stage_entries
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level import level_pieces

OUT = SURFACE_RECORDS / "wrap-survey.jsonl"
BIGGEST_NODES = 32_000_000


def bands():
    """Finished bands of the survey, smallest first."""
    found = [p for p in stage_entries(NODE_ROUTE, "nodes", "band-*") if done(p)]
    return sorted(found, key=lambda p: record_of(p)["nodes"])


def published_cm2(sample, volume, voxel_um, z_low, z_high):
    """Area of published meshes inside a band, duplicates counted, from their vertices."""
    total = 0.0
    for segment, folder in meshes(sample, volume):
        if "spiral-output" in segment:
            continue
        scale = json.loads((folder / "meta.json").read_text())["scale"][0]
        z = load_mesh(folder)[2]
        total += float(((z >= z_low) & (z < z_high)).sum()) * (voxel_um / scale / 10000) ** 2
    return total


def row(band):
    record = record_of(band)
    start = time.time()
    nodes, theta, _first, _grid, pairs, field, _ = level_pieces.prepare([band])
    root, _potential, _, _, _, _ = level_pieces.grow_pieces(nodes, theta, pairs, field)
    _ids, piece = np.unique(root, return_inverse=True)
    area = np.bincount(piece, weights=sheet_wrap_score.node_area_mm2(nodes, record["scan_voxel_um"], record["level"]))
    scale = 2 ** record["level"]
    lo, hi = record["cell_planes"]
    return {
        "sample": record["sample"],
        "volume": record["volume"],
        "band": band.name,
        "voxel_um": record["scan_voxel_um"],
        "nodes": len(nodes),
        "seconds": round(time.time() - start, 1),
        "turned": round(field["turned"], 4),
        "frame_costs": field.get("frame_costs"),
        "area_cm2": round(float(area.sum()) / 100, 1),
        "in_1cm2_pieces": round(float(area[area >= 100].sum()) / 100, 1),
        "pieces_1cm2": int((area >= 100).sum()),
        "in_4cm2_pieces": round(float(area[area >= 400].sum()) / 100, 1),
        "pieces_4cm2": int((area >= 400).sum()),
        "largest_cm2": round(float(area.max()) / 100, 1),
        "published_cm2": round(
            published_cm2(
                record["sample"], record["volume"], record["scan_voxel_um"], lo * CELL * scale, hi * CELL * scale
            ),
            1,
        ),
    }


def survey(part=0, parts=1):
    seen = (
        {json.loads(line)["band"] + json.loads(line)["volume"] for line in OUT.read_text().splitlines()}
        if OUT.exists()
        else set()
    )
    for index, band in enumerate(bands()):
        record = record_of(band)
        if index % parts != part or band.name + record["volume"] in seen or record["nodes"] > BIGGEST_NODES:
            continue
        with open(OUT, "a") as out:
            out.write(json.dumps(row(band)) + "\n")
        print(band.parent.parent.parent.name, band.name, flush=True)


if __name__ == "__main__":
    survey(*[int(v) for v in sys.argv[1:3]])
