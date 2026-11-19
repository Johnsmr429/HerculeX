"""Each piece flattened alone along the sheet itself and written as tifxyz at the nodes' own spacing, each grid node
the sheet point at its cell's centre, so neighbouring nodes lie one grid step apart in 3D."""

import ctypes
import json

import numpy as np
import tifffile
from scipy import ndimage as ni
from scipy.spatial import cKDTree

from unwrapping.detection.kernel_bindings import CELL

UNKNOWN, LOCKED, INTERPOLATED = 0, 1, 2
MIN_PIECE_NODES = 400
FOLD_SPREAD = 2.0
FLATTEN_STEPS = 300
FLATTEN_TOLERANCE = 1e-3
FILL_REACH = 3
FILL_ROUNDS = 150  # Jacobi rounds; a hole at most 2 x FILL_REACH across settles within 1e-6 of a cell
EDGE_VOXELS = 8  # how far past the scan a filled edge cell may reach and be left out rather than stop the unit
TEAR_CELLS = 3.0
UP_SLOPE = 0.3
UP_SAMPLES = 2000
UP_NEIGHBOURS = 24
UP_KEEP = 10.0
SLOPE_NODES = 4  # nodes of a cell's 3 x 3 cells to fit its slope
SLOPE_SPREAD = 0.1  # least variance of their flat places, in cells squared, across the narrower direction
SLOPE_LENGTHS = (0.5, 2.0)  # 3D length of a flat step a fitted slope may give
# An isometric piece: of its locked neighbours, ISOMETRIC_SHARE within 10% of a grid step apart in 3D, and from one
# step to the next along a row or column a median change of at most ISOMETRIC_JITTER steps. On real units the grids
# of cell means (the earlier rasterisation) have a median piece at 51-57% and 0.12-0.14, and 0-10% of pieces pass; centred,
# 84-93% and 0.01-0.02, and 96-99% pass (level PHerc1545 rows-0077-0082, grown PHerc1667 band-0012-0024). What the
# centring leaves is stretch over many cells where the sheet is corrugated finer than its normals.
ISOMETRIC_SHARE = 0.7
ISOMETRIC_JITTER = 0.07


def isometry(grid, known, locked, stride):
    """How far a piece's grid is from isometric: each pair of neighbouring nodes' 3D distance over the grid step,
    along rows and columns, over all known nodes and over locked ones, and the median change of that from one step to
    the next (jitter); isometric as ISOMETRIC_SHARE and ISOMETRIC_JITTER say."""
    out = {}
    for name, keep in (("all", known), ("locked", locked)):
        steps, changes = [], []
        for axis in (1, 2):
            length = np.linalg.norm(np.diff(grid, axis=axis), axis=0) / stride
            both = keep[1:, :] & keep[:-1, :] if axis == 1 else keep[:, 1:] & keep[:, :-1]
            steps.append(length[both])
            ahead = (slice(1, None), slice(None)) if axis == 1 else (slice(None), slice(1, None))
            behind = (slice(None, -1), slice(None)) if axis == 1 else (slice(None), slice(None, -1))
            changes.append(np.abs(length[ahead] - length[behind])[both[ahead] & both[behind]])
        steps, changes = np.concatenate(steps), np.concatenate(changes)
        out[name] = (
            {
                "p5_p50_p95": [round(float(p), 3) for p in np.percentile(steps, [5, 50, 95])],
                "within_10pct": round(float(np.mean(np.abs(steps - 1) <= 0.1)), 3),
                "jitter": round(float(np.median(changes)), 3) if changes.size else None,
                "pairs": int(steps.size),
            }
            if steps.size
            else None
        )
    locked_part = out["locked"]
    out["isometric"] = bool(
        locked_part
        and locked_part["within_10pct"] >= ISOMETRIC_SHARE
        and (locked_part["jitter"] is None or locked_part["jitter"] <= ISOMETRIC_JITTER)
    )
    return out


def unit_isometry(rows):
    """The written pieces' isometry in one line of a unit's record, measured and kept, never a reason to drop them."""
    measured = [r["isometry"] for r in rows if r.get("isometry") and r["isometry"]["locked"]]
    if not measured:
        return {"pieces": 0}
    share = float(np.mean([m["isometric"] for m in measured]))
    return {
        "pieces": len(measured),
        "isometric_share": round(share, 3),
        "median_within_10pct": round(float(np.median([m["locked"]["within_10pct"] for m in measured])), 3),
        "median_all_within_10pct": round(float(np.median([m["all"]["within_10pct"] for m in measured])), 3),
        "median_jitter": round(float(np.median([m["locked"]["jitter"] or 0.0 for m in measured])), 3),
    }


def pointer(array):
    return ctypes.c_void_p(array.ctypes.data)


def upright(nodes, u, v, turns, radius):
    """Flat places turned in their plane so the scroll's axis runs along +v where most of the piece lies: the median
    direction of z's slope over UP_SAMPLES local plane fits of UP_NEIGHBOURS nodes, left as it is when within UP_KEEP
    degrees; where the sheet lies nearly level (slopes under UP_SLOPE) the roll's direction is turned onto +u."""
    flat = np.c_[u, v].astype(np.float64)
    if len(flat) < UP_NEIGHBOURS:
        return u, v
    rng = np.random.default_rng(0)
    centres = rng.choice(len(flat), size=min(UP_SAMPLES, len(flat)), replace=False)
    _, near = cKDTree(flat).query(flat[centres], k=UP_NEIGHBOURS)
    design = np.concatenate([flat[near] - flat[centres][:, None, :], np.ones((*near.shape, 1))], axis=2)
    gram = np.einsum("nki,nkj->nij", design, design) + 1e-9 * np.eye(3)

    def median_direction(values):
        target = values[near] - values[centres][:, None]
        slope = np.linalg.solve(gram, np.einsum("nki,nk->ni", design, target)[..., None])[:, :2, 0]
        steep = np.linalg.norm(slope, axis=1) >= UP_SLOPE
        if steep.mean() < 0.5:
            return None
        angle = np.arctan2(slope[steep, 0], slope[steep, 1])
        centre = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        return centre + np.median(np.angle(np.exp(1j * (angle - centre))))

    angle = median_direction(nodes["z"].astype(np.float64))
    if angle is None:
        roll = median_direction(2 * np.pi * turns * radius)
        if roll is None:
            return u, v
        angle = roll - np.pi / 2
    if abs(np.degrees(np.angle(np.exp(1j * angle)))) <= UP_KEEP:
        return u, v
    cos, sin = np.cos(angle), np.sin(angle)
    return (u * cos - v * sin).astype(u.dtype), (u * sin + v * cos).astype(v.dtype)


def rasterise(nodes, normal, u, v):
    """Position per flat cell: the mean of its nodes, moved along the sheet to the cell's centre."""
    iu = ((u - u.min()) / CELL).astype(np.int64)
    iv = ((v - v.min()) / CELL).astype(np.int64)
    shape = (int(iv.max()) + 1, int(iu.max()) + 1)
    flat = iv * shape[1] + iu
    count = np.bincount(flat, minlength=shape[0] * shape[1]).astype(np.float64)
    occupied = count > 0
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64)
    grid = np.stack([np.bincount(flat, weights=xyz[:, k], minlength=count.size) for k in range(3)])
    grid[:, occupied] /= count[occupied]
    depth = ((xyz - grid[:, flat].T) * normal).sum(axis=1)
    spread = np.sqrt(np.bincount(flat, weights=depth**2, minlength=count.size) / np.maximum(count, 1))
    locked = occupied & (spread <= FOLD_SPREAD)
    grid = centred(xyz, u, v, shape, grid, np.ones(len(xyz), dtype=bool), locked[flat])
    return grid.reshape(3, *shape), locked.reshape(shape), np.mean(count[occupied])


def centred(xyz, u, v, shape, mean, take, source):
    """Cell means (3, cells) of the nodes take, each moved along the sheet from where they lie on average to the
    centre of its cell.

    A cell holds one or two nodes anywhere in it, so its mean lies up to half a cell from the cell's centre, and from
    cell to cell those offsets beat with the voxel lattice the nodes came from: a grid of means is 0.6 to 1.6 steps
    from node to node in 3D, and the image of it swirls. The sheet's slope d xyz / d(u, v) of each cell is fitted by
    least squares over the source nodes of its 3 x 3 cells and the mean moved by slope x (centre - mean place), which
    is exact on a plane and keeps each mean's depth. Cells with too few source nodes around, or spread too narrowly
    for a slope, or a slope unlike a sheet laid flat, keep their mean.
    """
    h, w = shape
    u0, v0 = u - u.min(), v - v.min()
    iu, iv = (u0 / CELL).astype(np.int64), (v0 / CELL).astype(np.int64)
    flat = iv * w + iu
    du, dv = u0 - (iu + 0.5) * CELL, v0 - (iv + 0.5) * CELL
    f, a, b, x = flat[source], du[source], dv[source], xyz[source]

    def summed(weights=None):
        return np.pad(np.bincount(f, weights=weights, minlength=h * w).reshape(h, w).astype(np.float64), 1)

    keys = ["n", "u", "v", "uu", "uv", "vv", *[f"{p}{k}" for k in range(3) for p in ("x", "xu", "xv")]]
    cell = dict(zip(keys[:6], (summed(), summed(a), summed(b), summed(a * a), summed(a * b), summed(b * b))))
    for k in range(3):
        cell[f"x{k}"], cell[f"xu{k}"], cell[f"xv{k}"] = summed(x[:, k]), summed(x[:, k] * a), summed(x[:, k] * b)
    m = {key: np.zeros((h, w)) for key in keys}
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            s = {key: c[1 + oy : 1 + oy + h, 1 + ox : 1 + ox + w] for key, c in cell.items()}
            su, sv = ox * CELL, oy * CELL  # a node there lies (du + su, dv + sv) from this cell's centre
            m["n"] += s["n"]
            m["u"] += s["u"] + su * s["n"]
            m["v"] += s["v"] + sv * s["n"]
            m["uu"] += s["uu"] + 2 * su * s["u"] + su * su * s["n"]
            m["vv"] += s["vv"] + 2 * sv * s["v"] + sv * sv * s["n"]
            m["uv"] += s["uv"] + sv * s["u"] + su * s["v"] + su * sv * s["n"]
            for k in range(3):
                m[f"x{k}"] += s[f"x{k}"]
                m[f"xu{k}"] += s[f"xu{k}"] + su * s[f"x{k}"]
                m[f"xv{k}"] += s[f"xv{k}"] + sv * s[f"x{k}"]
    n = np.maximum(m["n"], 1)
    mu, mv = m["u"] / n, m["v"] / n
    cuu, cvv, cuv = m["uu"] / n - mu * mu, m["vv"] / n - mv * mv, m["uv"] / n - mu * mv
    det = cuu * cvv - cuv * cuv
    narrow = (cuu + cvv) / 2 - np.sqrt(np.maximum((cuu - cvv) ** 2 / 4 + cuv * cuv, 0))
    fit = (m["n"] >= SLOPE_NODES) & (narrow > SLOPE_SPREAD * CELL**2)
    det = np.where(fit, det, 1.0)
    slope = np.zeros((3, 2, h, w))
    for k in range(3):
        mx = m[f"x{k}"] / n
        cxu, cxv = m[f"xu{k}"] / n - mx * mu, m[f"xv{k}"] / n - mx * mv
        slope[k, 0], slope[k, 1] = (cxu * cvv - cxv * cuv) / det, (cxv * cuu - cxu * cuv) / det
    length = np.linalg.norm(slope, axis=0)
    fit &= ((length > SLOPE_LENGTHS[0]) & (length < SLOPE_LENGTHS[1])).all(axis=0)
    count = np.bincount(flat[take], minlength=h * w).astype(np.float64)
    move = (fit.ravel() & (count > 0)).reshape(h, w)
    place = [np.bincount(flat[take], weights=d[take], minlength=h * w).reshape(h, w) / np.maximum(count, 1).reshape(h, w)
             for d in (du, dv)]  # fmt: skip
    out = mean.reshape(3, h, w).copy()
    out[:, move] -= (slope[:, 0] * place[0] + slope[:, 1] * place[1])[:, move]
    return out.reshape(mean.shape)


def tears(grid, known):
    """Cells at either end of a mesh edge longer than TEAR_CELLS."""
    torn = np.zeros_like(known)
    for axis in (0, 1):
        here = tuple(slice(None, -1) if a == axis else slice(None) for a in (0, 1))
        there = tuple(slice(1, None) if a == axis else slice(None) for a in (0, 1))
        step = grid[(slice(None), *here)] - grid[(slice(None), *there)]
        long = known[here] & known[there] & ((step**2).sum(axis=0) > (TEAR_CELLS * CELL) ** 2)
        torn[here] |= long
        torn[there] |= long
    return torn


def bridge_holes(grid, locked):
    """Fill holes up to FILL_REACH cells from each side: grown in by normalised convolution, then relaxed to the
    harmonic fill of the locked cells round them, which lays a plane across a hole evenly; the convolution alone
    copies the rim inwards, bunching the filled cells at it (neighbours down to a twentieth of a step apart) and
    stretching them where the two sides meet."""
    pad = FILL_REACH + 1
    hole = ni.binary_closing(np.pad(locked, pad), np.ones((3, 3), dtype=bool), iterations=FILL_REACH)[
        pad:-pad, pad:-pad
    ]
    weight = locked.astype(np.float64)
    filled = grid * weight
    support = weight.copy()
    for _ in range(FILL_REACH):
        s = ni.uniform_filter(support, 3, mode="constant")
        values = [ni.uniform_filter(filled[k], 3, mode="constant") for k in range(3)]
        grow = (support == 0) & (s > 0.5 / 9) & hole
        for k in range(3):
            filled[k][grow] = values[k][grow] / s[grow]
        support[grow] = 1.0
    known = support > 0
    filled = relaxed(filled, known, known & ~locked)
    return filled, known & ~tears(filled, known)


def relaxed(grid, known, free, rounds=FILL_ROUNDS):
    """The free cells of a grid made each the mean of its known 4-neighbours, the others held, by Jacobi rounds."""
    if not free.any():
        return grid
    weight = np.pad(known, 1).astype(np.float64)
    count = weight[:-2, 1:-1] + weight[2:, 1:-1] + weight[1:-1, :-2] + weight[1:-1, 2:]
    free = free & (count > 0)
    grid = grid.copy()
    for _ in range(rounds):
        edged = np.pad(np.where(known, grid, 0.0), ((0, 0), (1, 1), (1, 1)))
        around = edged[:, :-2, 1:-1] + edged[:, 2:, 1:-1] + edged[:, 1:-1, :-2] + edged[:, 1:-1, 2:]
        grid[:, free] = around[:, free] / count[free]
    return grid


def write_piece(folder, grid, known, locked, scale, meta):
    """tifxyz at the nodes' own spacing."""
    folder.mkdir(parents=True, exist_ok=True)
    stride = CELL * scale
    span = np.array([meta["shape"][2], meta["shape"][1], meta["shape"][0]], dtype=np.float64)
    for axis, name in enumerate("xyz"):
        # a hole filled at the scan's edge can carry a cell a voxel or two past it, where there is no CT: that cell is
        # left out; a grid further out than EDGE_VOXELS is wrong and stops the unit
        values = grid[axis] * scale
        beyond = known & ((values < -1) | (values > span[axis]))
        if (known & ((values < -1 - EDGE_VOXELS) | (values > span[axis] + EDGE_VOXELS))).any():
            raise ValueError(
                f"{folder.name}: axis {name} leaves the scan: {values[known].min():.1f} to {values[known].max():.1f}"
            )
        known = known & ~beyond
    for axis, name in enumerate("xyz"):
        tifffile.imwrite(
            folder / f"{name}.tif", np.where(known, grid[axis] * scale + 0.5 * (scale - 1), -1).astype(np.float32)
        )
    quality = np.where(known, np.where(locked, LOCKED, INTERPOLATED), UNKNOWN).astype(np.uint8)
    np.save(folder / "quality.npy", quality)
    points = (grid[:, known] * scale + 0.5 * (scale - 1)).astype(np.float64)
    (folder / "meta.json").write_text(
        json.dumps(
            {
                **meta,
                "format": "tifxyz",
                "type": "seg",
                "uuid": folder.name,
                "bbox": [points.min(axis=1).tolist(), points.max(axis=1).tolist()],
                "scale": [1 / stride, 1 / stride],
                "stride": stride,
                "locked_share": float((quality == LOCKED).mean()),
            },
            indent=1,
        )
    )
