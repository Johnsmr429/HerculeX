"""Wrap truth of a sheet control."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from calibration.unwrap import mesh_reference as reference
from calibration.unwrap.mesh_surfaces import Axis, grid_normals, unwrap_grid
from calibration.unwrap.sheet_controls import load_mesh, meshes
from runner.stage_queue import record_of
from storage.data_roots import SHEET_CONTROLS, calibration_output
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level import scroll_axis
from unwrapping.level.sheet_graph import filtered, load_field

VERTEX_UM = 150.0
TOLERANCE_UM = 40.0
LATERAL_UM = 160.0
END_UM = 500.0
CANDIDATES = 6
EXCLUDED = ("spiral-output",)
OUT = calibration_output("geometry_wrap") / "truth"
BATCH = 400_000


def mesh_rows(folder, voxel_um, z_low, z_high):
    """Grid rows of a mesh reaching the band, thinned to VERTEX_UM, with normals and validity."""
    grid = load_mesh(folder)
    scale = json.loads((Path(folder) / "meta.json").read_text())["scale"][0]
    step = max(1, round(VERTEX_UM * scale / voxel_um))
    valid = (grid[0] > 0) & (grid[1] > 0) & (grid[2] > 0) & np.isfinite(grid).all(axis=0)
    rows = np.flatnonzero((valid & (grid[2] >= z_low) & (grid[2] < z_high)).any(axis=1))
    if not len(rows):
        return None
    window = slice(rows[0], rows[-1] + 1)
    x, y, z, good = (
        grid[0][window][::step, ::step],
        grid[1][window][::step, ::step],
        grid[2][window][::step, ::step],
        valid[window][::step, ::step],
    )
    normal, fine = grid_normals(x, y, z, good)
    return {"x": x, "y": y, "z": z, "valid": good & fine, "normal": normal, "stride": step, "row0": int(rows[0])}


def vertices(sample, volume, voxel_um, axis, z_low, z_high, only=None):
    """Reference arrays in the geometry benchmark's form, every mesh one tier."""
    columns = {k: [] for k in ("x", "y", "z", "nx", "ny", "nz", "W", "mesh", "unit", "row", "col")}
    names, strides, units = [], [], []
    margin = END_UM / voxel_um
    for segment, folder in meshes(sample, volume):
        if any(word in segment for word in EXCLUDED) or (only is not None and segment not in only):
            continue
        g = mesh_rows(folder, voxel_um, z_low, z_high)
        if g is None:
            continue
        theta = axis.theta(g["x"].astype(np.float64), g["y"].astype(np.float64), g["z"].astype(np.float64))
        phi, part = unwrap_grid(theta, g["valid"])
        keep = g["valid"] & (g["z"] >= z_low + margin) & (g["z"] < z_high - margin)
        r, c = np.nonzero(keep)
        if len(r) < 50:
            continue
        parts = part[r, c]
        present = np.unique(parts)
        lookup = {int(p): len(units) + k for k, p in enumerate(present)}
        units.extend((len(names), int(p)) for p in present)
        for name, value in (("x", g["x"]), ("y", g["y"]), ("z", g["z"])):
            columns[name].append(value[r, c])
        for k, name in enumerate(("nx", "ny", "nz")):
            columns[name].append(g["normal"][r, c, k])
        columns["W"].append(phi[r, c] / (2 * np.pi))
        columns["mesh"].append(np.full(len(r), len(names), dtype=np.int32))
        columns["unit"].append(np.array([lookup[int(p)] for p in parts], dtype=np.int32))
        columns["row"].append(r.astype(np.int32))
        columns["col"].append(c.astype(np.int32))
        names.append(segment)
        strides.append(1)
    ref = {k: np.concatenate(v) for k, v in columns.items()}
    ref["names"], ref["tiers"], ref["strides"] = np.array(names), np.array(["A"] * len(names)), np.array(strides)
    ref["unit_mesh"] = np.array([u[0] for u in units], dtype=np.int32)
    return ref


def node_candidates(ref, xyz, voxel_um):
    """For each vertex the nearest sheet nodes on its plane within the reach, nearest along the normal first."""
    p, n = reference.points_of(ref), reference.normals_of(ref)
    tolerance, lateral = TOLERANCE_UM / voxel_um, LATERAL_UM / voxel_um
    tree = cKDTree(xyz)
    reach = float(np.hypot(tolerance, lateral))
    out = np.full((len(p), CANDIDATES), -1, dtype=np.int32)
    depth_out = np.full((len(p), CANDIDATES), np.inf, dtype=np.float32)
    for start in range(0, len(p), BATCH):
        q, m = p[start : start + BATCH], n[start : start + BATCH]
        distance, index = tree.query(q, k=32, distance_upper_bound=reach, workers=-1)
        found = np.isfinite(distance)
        safe = np.where(found, index, 0)
        offset = q[:, None, :] - xyz[safe]
        depth = np.abs((offset * m[:, None, :]).sum(axis=2))
        side2 = np.maximum((offset**2).sum(axis=2) - depth**2, 0)
        ok = found & (depth <= tolerance) & (side2 <= lateral**2)
        depth = np.where(ok, depth, np.inf)
        order = np.argsort(depth, axis=1, kind="stable")[:, :CANDIDATES]
        rows = np.arange(len(q))[:, None]
        hit = np.isfinite(depth[rows, order])
        out[start : start + BATCH] = np.where(hit, safe[rows, order], -1)
        depth_out[start : start + BATCH] = depth[rows, order]
    return out, depth_out


def build(control, only=None, suffix=""):
    """The truth of one control written to OUT; returns its audit."""
    control = Path(control)
    record = record_of(control)
    voxel_um, level = record["scan_voxel_um"], record["level"]
    scale = 2**level
    raw, cells, rec, grid, ids = load_field([control])
    nodes = filtered(raw, cells, grid, ids)[0]
    del raw, cells
    knots = scroll_axis.axis(nodes, float(nodes["z"].min()), float(nodes["z"].max()) + 1)
    knots_l0 = knots * scale + 0.5 * (scale - 1)
    axis = Axis(knots_l0)
    lo, hi = rec["cell_planes"]
    z_low, z_high = lo * CELL * scale, hi * CELL * scale
    ref = vertices(record["sample"], record["volume"], voxel_um, axis, z_low, z_high, only)
    reference.VOXEL_UM = voxel_um
    edges = reference.coincidence(ref)
    frame = reference.frame(ref, edges, ("A",), False)
    for name in ("keep", "group", "Wg", "duplicate"):
        ref[f"{name}_A"] = frame[name]
    ref["unresolved_A"] = reference.unresolved(ref, "A")
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64) * scale + 0.5 * (scale - 1)
    keep = np.flatnonzero(ref["keep_A"])
    kept = {k: ref[k][keep] for k in ("x", "y", "z", "nx", "ny", "nz")}
    candidates, depth = node_candidates(kept, xyz, voxel_um)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"{control.name}{suffix}.npz",
        candidates=candidates,
        depth=depth.astype(np.float16),
        Wg=ref["Wg_A"][keep],
        group=ref["group_A"][keep],
        mesh=ref["mesh"][keep],
        unit=ref["unit"][keep],
        row=ref["row"][keep],
        col=ref["col"][keep],
        unresolved=ref["unresolved_A"][keep],
        x=kept["x"],
        y=kept["y"],
        z=kept["z"],
        names=ref["names"],
        knots=knots,
        vertex_mm2=np.float64((VERTEX_UM / 1000) ** 2),
        voxel_um=np.float64(voxel_um),
        level=np.int64(level),
    )
    audit = {
        "control": control.name,
        "vertices": len(ref["x"]),
        "kept": len(keep),
        "meshes": len(ref["names"]),
        "groups": int(frame["groups"]),
        "unresolved": round(float(ref["unresolved_A"][keep].mean()), 4),
        "with_node": round(float((candidates[:, 0] >= 0).mean()), 4),
        "cm2": round(float(len(keep) * (VERTEX_UM / 1000) ** 2 / 100), 1),
    }
    print(json.dumps(audit), flush=True)
    return audit


if __name__ == "__main__":
    if sys.argv[1] == "tier-a":
        for name in sys.argv[2:]:
            build(name if "/" in name else SHEET_CONTROLS / name, reference.TIER_A, "-tier-a")
    else:
        for name in sys.argv[1:]:
            build(name if "/" in name else SHEET_CONTROLS / name)
