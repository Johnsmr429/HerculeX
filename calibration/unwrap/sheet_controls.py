"""Unwrapping controls: a band of every scan with published meshes, as sheet nodes with mesh truth."""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from runner import stage_queue
from scans.catalog.scan_catalog import load, volumes
from scans.catalog.scan_choice import sheet_level
from scans.transfer.fetch_volume import fetch_level
from storage.brick_store import UNKNOWN
from storage.data_roots import OPEN_DATA_MIRROR, SHEET_CONTROLS, calibration_output
from unwrapping.detection.detect_nodes import (
    BAND_ROWS,
    CONNECTIONS,
    band_planes,
    detect_store,
)
from unwrapping.detection.kernel_bindings import CELL

MESH_UM = (7.0, 10.0, 2.0, 2.6)
MIN_MESHES = 3
TRUTH_UM = 100.0
SAME_SHEET_UM = 40.0
MAX_TRUTH_VERTICES = 4_000_000
DUPLICATES = {"20241024131838", "20250820105138", "20260319133554", "20260304131111"}


def wrap_of(segment):
    match = re.search(r"-w(\d+)_", segment)
    return int(match.group(1)) if match else -1


def meshes(sample, volume):
    """Every published mesh on one scan, from the mirror."""
    found = sorted(OPEN_DATA_MIRROR.glob(f"{sample}/segments/*/mesh/*-on-{volume}-*um.tifxyz"))
    return [(path.parent.parent.name, path) for path in found if (path / "z.tif").exists()]


def load_mesh(folder):
    """(3, rows, columns) float32 x, y, z in level-0 voxels."""
    return np.stack([tifffile.imread(Path(folder) / f"{a}.tif") for a in "xyz"])


def valid_points(mesh):
    """Valid vertices as (n, 3) x, y, z with unit normals from the grid's tangents."""
    good = mesh[0] > 0
    du = mesh[:, 1:-1, 2:] - mesh[:, 1:-1, :-2]
    dv = mesh[:, 2:, 1:-1] - mesh[:, :-2, 1:-1]
    inner = good[1:-1, 1:-1] & good[1:-1, 2:] & good[1:-1, :-2] & good[2:, 1:-1] & good[:-2, 1:-1]
    normal = np.cross(du.reshape(3, -1).T, dv.reshape(3, -1).T)
    length = np.linalg.norm(normal, axis=1)
    keep = inner.ravel() & (length > 0)
    return mesh[:, 1:-1, 1:-1].reshape(3, -1).T[keep], normal[keep] / length[keep, None]


def candidates():
    """Scans with enough meshes at survey-like voxel sizes."""
    rows = []
    for row in volumes(load()):
        um = row["voxel_um"] or 0
        if (
            not (MESH_UM[0] <= um <= MESH_UM[1] or MESH_UM[2] <= um <= MESH_UM[3])
            or row["products"].get("ome-zarr") is None
            or row["volume"] in DUPLICATES
        ):
            continue
        found = meshes(row["sample"], row["volume"])
        if len(found) >= MIN_MESHES:
            rows.append(row | {"meshes": len(found)})
    return rows


def plan():
    """Per scan, the band of brick rows at the sheet level that holds the most mesh vertices."""
    out = []
    for row in candidates():
        level = sheet_level(row["voxel_um"])
        z = (
            np.concatenate([z[z > 0] for z in (load_mesh(path)[2] for _, path in meshes(row["sample"], row["volume"]))])
            / 2**level
        )
        rows = np.bincount((z // 128).astype(np.int64))
        totals = (
            np.convolve(rows, np.ones(BAND_ROWS, dtype=np.int64), mode="valid")
            if len(rows) >= BAND_ROWS
            else rows.sum(keepdims=True)
        )
        first = int(np.argmax(totals))
        out.append(
            {
                "sample": row["sample"],
                "volume": row["volume"],
                "voxel_um": row["voxel_um"],
                "level": level,
                "rows": [first, first + BAND_ROWS],
                "meshes": row["meshes"],
                "vertices": int(totals.max()),
            }
        )
    (SHEET_CONTROLS / "plan.json").parent.mkdir(parents=True, exist_ok=True)
    (SHEET_CONTROLS / "plan.json").write_text(json.dumps(out, indent=1))
    return out


def truth_in(sample, volume, level, z_low, z_high):
    """Mesh vertices in the band at the sheet level, with normals, the mesh's index and its wrap number where named."""
    points, normals, mesh_index, wraps, names = [], [], [], [], []
    voxel = next(v["voxel_um"] for v in volumes(load()) if v["volume"] == volume)
    for segment, path in meshes(sample, volume):
        scale = json.loads((path / "meta.json").read_text())["scale"][0]
        step = max(1, round(TRUTH_UM * scale / voxel))
        p, n = valid_points(load_mesh(path)[:, ::step, ::step])
        p = p / 2**level
        inside = (p[:, 2] >= z_low) & (p[:, 2] < z_high)
        if inside.sum() < 50:
            continue
        points.append(p[inside].astype(np.float32))
        normals.append(n[inside].astype(np.float32))
        mesh_index.append(np.full(int(inside.sum()), len(names), dtype=np.int32))
        wraps.append(np.full(int(inside.sum()), wrap_of(segment), dtype=np.int32))
        names.append(segment)
    every = max(1, -(-sum(len(v) for v in points) // MAX_TRUTH_VERTICES))
    truth = {
        "points": np.concatenate(points)[::every],
        "normals": np.concatenate(normals)[::every],
        "mesh": np.concatenate(mesh_index)[::every],
        "wrap": np.concatenate(wraps)[::every],
        "names": np.array(names),
    }
    truth["coincide"] = coincident(truth, SAME_SHEET_UM / (voxel * 2**level))
    return truth


def coincident(truth, tolerance):
    """Keys vertex * meshes + mesh of every other mesh a vertex lies on."""
    p, n, mesh = truth["points"].astype(np.float64), truth["normals"].astype(np.float64), truth["mesh"]
    near, other = cKDTree(p).query(p, k=16, distance_upper_bound=4 * tolerance)
    other = np.minimum(other, len(p) - 1)
    rows, cols = np.nonzero(np.isfinite(near) & (mesh[other] != mesh[:, None]))
    depth = np.abs(((p[other[rows, cols]] - p[rows]) * n[rows]).sum(axis=1))
    keep = depth <= tolerance
    return np.unique(rows[keep].astype(np.int64) * len(truth["names"]) + mesh[other[rows, cols]][keep])


def update_truth():
    """Coincidence keys added to the truth of every control built before they existed."""
    for folder in sorted(SHEET_CONTROLS.glob("*/truth.npz")):
        truth = {k: v for k, v in np.load(folder).items() if k != "sheet"}
        if "coincide" not in truth:
            truth["coincide"] = coincident(truth, SAME_SHEET_UM / stage_queue.record_of(folder.parent)["voxel_um"])
            np.savez(folder, **truth)
            print(
                folder.parent.name,
                len(truth["names"]),
                "meshes",
                round(len(np.unique(truth["coincide"] // len(truth["names"]))) / len(truth["mesh"]), 3),
                "of vertices on another mesh",
                flush=True,
            )


def control_name(row):
    return f"{row['sample']}-{row['volume']}-rows-{row['rows'][0]:04d}-{row['rows'][1]:04d}"


def build(row):
    """Fetch the band at the sheet level, keep only its nodes, and the truth that lies in it."""
    folder = SHEET_CONTROLS / control_name(row)
    if stage_queue.done(folder):
        if not (folder / "truth.npz").exists():
            record = stage_queue.record_of(folder)
            np.savez(
                folder / "truth.npz",
                **truth_in(
                    row["sample"],
                    row["volume"],
                    row["level"],
                    record["cell_planes"][0] * CELL,
                    record["cell_planes"][1] * CELL,
                ),
            )
            print(folder.name, "truth added", flush=True)
        return
    scan = next(v for v in volumes(load()) if v["volume"] == row["volume"])
    work = calibration_output("sheet_controls") / folder.name
    first, last = row["rows"]
    try:
        store = fetch_level(
            scan["products"]["ome-zarr"],
            row["level"],
            work,
            connections=CONNECTIONS,
            root=scan["roots"]["ome-zarr"],
            z_bricks=(first, last),
        )
        if int((store.offsets[first:last] == UNKNOWN).sum()):
            raise OSError("bricks still unknown")
        planes = band_planes(first, last, store.shape[0])
        record = {
            "sample": row["sample"],
            "volume": row["volume"],
            "scan_voxel_um": row["voxel_um"],
            "level": row["level"],
            "voxel_um": row["voxel_um"] * 2 ** row["level"],
            "rows": row["rows"],
        }
        detect_store(work, folder, record, planes)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    truth = truth_in(row["sample"], row["volume"], row["level"], planes[0] * CELL, planes[1] * CELL)
    np.savez(folder / "truth.npz", **truth)
    print(folder.name, len(truth["names"]), "meshes", len(truth["points"]), "vertices", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        for row in plan():
            print(row, flush=True)
    elif sys.argv[1] == "truth":
        update_truth()
    else:
        for row in json.loads((SHEET_CONTROLS / "plan.json").read_text()):
            if len(sys.argv) < 3 or row["sample"] in sys.argv[2:]:
                build(row)
