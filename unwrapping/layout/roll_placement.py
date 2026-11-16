"""Pieces of one height band placed along the roll by their winding."""

import numpy as np
from scipy import ndimage as ni

from runner import stage_queue
from sampling import column_sampling as sampler
from unwrapping.level.scroll_axis import fit_centre

TURN_BIN = 0.02


def outward_grid(xyz, centre):
    """True when the grid's own normal points away from the roll's middle, by the vote of the cells whose neighbours all exist."""
    solid = ni.binary_erosion(xyz[0] >= 0)
    normal = np.cross(np.gradient(xyz, axis=2), np.gradient(xyz, axis=1), axis=0)[:2, solid]
    inward = np.asarray(centre[:2], dtype=np.float64)[:, None] - xyz[:2, solid]
    vote = (normal * inward).sum(axis=0) / np.maximum(
        np.linalg.norm(normal, axis=0) * np.linalg.norm(inward, axis=0), 1e-9
    )
    return float(vote.sum()) < 0


def loaded(unit):
    """The written pieces of a unit."""
    rows = [r for r in stage_queue.record_of(unit)["pieces"] if r["written"]]
    pieces = []
    for row in rows:
        xyz, meta = sampler.load_surface(unit / f"piece-{row['piece']:04d}")
        pieces.append({"row": row, "xyz": xyz.astype(np.float64), "good": xyz[0] >= 0, "meta": meta})
    return pieces


def scroll_centre(pieces):
    """The point the sheets' normals converge on, as the unwrapper finds it."""
    x, y, ux, uy = [], [], [], []
    c0 = np.mean([p["xyz"][:, p["good"]].mean(axis=1) for p in pieces], axis=0)
    for p in pieces:
        normal = sampler.grid_normals(p["xyz"].astype(np.float32), c0, p["meta"]["stride"] * p["meta"]["voxel_um"])
        g = p["good"][::4, ::4]
        n = normal[:2, ::4, ::4][:, g]
        length = np.hypot(*n)
        keep = length > 0.7
        x.append(p["xyz"][0, ::4, ::4][g][keep])
        y.append(p["xyz"][1, ::4, ::4][g][keep])
        ux.append(n[0][keep] / length[keep])
        uy.append(n[1][keep] / length[keep])
    x, y, ux, uy = map(np.concatenate, (x, y, ux, uy))
    return fit_centre(x, y, ux, uy, np.ones_like(x))


def turns(piece, centre):
    """Continuous turn count of the piece's cells, on the winding the unwrapper gave it."""
    near = ni.distance_transform_edt(~piece["good"], return_distances=False, return_indices=True)
    x, y = [piece["xyz"][k][tuple(near)] - centre[k] for k in range(2)]
    theta = np.unwrap(np.unwrap(np.arctan2(y, x), axis=1), axis=0) / (2 * np.pi)
    return theta + piece["row"]["winding"] - np.median(np.round(theta[piece["good"]]))


def pose(piece, centre, chirality):
    """Which way is up and which way right on the piece's grid, as (column, row) steps seen from the written face."""
    ii, jj = np.nonzero(piece["good"])
    a = np.linalg.lstsq(np.c_[ii, jj, np.ones(len(ii))], piece["xyz"][2][piece["good"]], rcond=None)[0]
    up = np.array([a[1], a[0]]) / max(np.hypot(a[0], a[1]), 1e-9) * chirality
    outward = outward_grid(piece["xyz"], centre)
    piece |= {
        "up": up,
        "right": np.array([-up[1], up[0]]) if outward else np.array([up[1], -up[0]]),
        "outward": outward,
        "anchor": np.array([np.median(jj), np.median(ii)]),
        "side": piece["meta"]["stride"] * piece["meta"]["voxel_um"] / 1000,
    }


def placed(pieces, centre, chirality):
    """Place of each piece along the roll in mm from its middle."""
    speeds = []
    for p in pieces:
        pose(p, centre, chirality)
        grid = turns(p, centre) * chirality
        um = p["meta"]["voxel_um"] / 1000
        rows, cols = np.gradient(grid)
        along = np.abs(cols * p["right"][0] + rows * p["right"][1]) / p["side"]
        good = ni.binary_erosion(p["good"], iterations=2) & (along > 1e-5)
        speeds.append(np.c_[grid[good], 1.0 / along[good]])
        t = grid[p["good"]]
        r = np.hypot(p["xyz"][0][p["good"]] - centre[0], p["xyz"][1][p["good"]] - centre[1]) * um
        p |= {
            "turn": float(grid[int(p["anchor"][1]), int(p["anchor"][0])]),
            "radius": float(np.median(r)),
            "height": float(np.median(p["xyz"][2][p["good"]])) * um,
        }
    t, speed = np.concatenate(speeds).T
    edges = np.arange(np.floor(t.min()), np.ceil(t.max()) + TURN_BIN, TURN_BIN)
    index = np.digitize(t, edges) - 1
    order = np.argsort(index, kind="stable")
    cuts = np.flatnonzero(np.diff(index[order])) + 1
    bins = index[order][np.r_[0, cuts]]
    medians = np.array([np.median(s) for s in np.split(speed[order], cuts)])
    length = np.interp(np.arange(len(edges)), bins, medians)
    length = ni.median_filter(length, 5, mode="nearest")
    arc = np.concatenate([[0.0], np.cumsum(length * TURN_BIN)])[: len(edges)]
    for p in pieces:
        p["arc"] = float(np.interp(p["turn"], edges, arc))
    return edges, length, arc
