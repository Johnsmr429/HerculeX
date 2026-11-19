"""Frozen truth packages of the bench controls (controls.json), built from the published meshes alone.

A package is a folder under the data root's bench/controls/NAME:
  truth.npz      every mesh grid vertex inside the control's z range, level-0 x, y, z: normal (towards the axis),
                 mesh, grid row and column, continuous winding, frame, area, zone and flags
  crossings.npz  where each mesh crosses the z planes z0, z0 + step, ...: one point per grid column, in column order
  meshes/        each mesh cropped to the z range as tifxyz (-1 outside), with w.tif holding the winding
  axis.json      scroll axis knots (z, x, y) in level-0 voxels: the windings are angles about it
  control.json   definition, sources with sha256, checks, and sha256 of every file written

Meshes come from the library's mirror of the open-data bucket, or from cache/open_data in the outputs where that mirror
holds empty files (PHercParis4 on scan 20230205180739). Since version 3 each mesh grid is cut to its rows within 512
slices of the box before its normals are computed (rows and columns stay those of the source grid); the axis rings
and the winding are still taken over the whole grid, one mesh at a time. Since version 4 the axis lies inside the
innermost traced turn: per 256-slice bin, the point the truth's grid rows wind around most (core_axis), falling back to
the median centre of circles fitted to closed rows. On a crushed roll a fitted centre can lie outside the inner turns
(PHercParis4: 269 voxels off the core on z 7800, on a sheet), and the angle about it then misses turns.

Winding. On each mesh the angle about the axis is unwrapped along the grid (since version 5 along each row, rows joined
at their outermost shared column); tau = sense * angle / 2pi grows by one per
turn along the sheet. A vertex's winding is tau + k for its mesh, so winding - sense * angle / 2pi is a whole number
everywhere and the fraction needs no fitting. On scrolls whose meshes carry wrap names (-wNNN_, or -wNNN-MMM for
a mesh over wraps NNN to MMM) and form a run of consecutive wraps, k is chained from a mesh's last wrap N to the
mesh named N+1 by the radially nearest outward neighbour (winding one higher); one global integer then makes the
wrap's start read its name. Each run of consecutive names is one frame. A control with "winding": "names" (used
for PHercParis4, whose meshes span several wraps each) instead anchors every named mesh by its own name, all of them
in one frame; the checks then measure how far the names agree. Without names each mesh is its own frame, and so is every unnamed mesh among named ones (since version 2; version 1 left
those without a winding, flags or crossings). Since version 2, frames are then joined where their meshes trace one
surface: vertices of two frames within 16 µm along the normal vote for the integer offset between them, and frames
join on at least 200 such votes with 90% on one offset, the strongest pairs first (control.json frame_links).

Checks (no CT), between meshes of one frame: radial order (the nearest truth sheet outward is one winding higher),
contacts (two meshes within 40 µm along the normal differ by 0, the same sheet twice, or by 1, pressed neighbours;
never more), chain purity.

Flags: 1 duplicate (the vertex repeats a larger mesh at the same winding; left out of areas), 2 pressed (another
winding within 80 µm along the normal), 4 overlap with the next wrap's mesh, 8 with the previous wrap's mesh,
16 with another mesh at the same winding, 32 conflict (another mesh within 40 µm along the normal two or more
windings away: the truth contradicts itself there), 64 order (the nearest outward vertex of another mesh is not
higher in winding), 128 touching (another winding within 40 µm along the normal: at the scorer's tolerance the
truth cannot tell the two sheets apart there). Scoring leaves conflict and order vertices unjudged.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import cKDTree

from calibration.bench import truth as truths
from calibration.bench.results import commit_id
from calibration.bench.surface_score import vertex_area
from storage.data_roots import OPEN_DATA_EXTRA
from storage.data_roots import SURFACE_BENCH as BENCH

MANIFEST = Path(__file__).with_name("controls.json")
PACKAGES = BENCH / "controls"
MIRROR = OPEN_DATA_EXTRA
VERSION = 5
ZONES = ("dev", "buffer", "holdout-A", "holdout-B", "train", "val")
SAME_SHEET_UM = 40.0
SAME_SURFACE_UM = 16.0
LINK_VOTES = 200
LINK_PURITY = 0.9
GRID_MARGIN = 512
PRESSED_UM = 80.0
RADIAL_UM = 400.0
LATERAL_SHARE = 0.75
VOTE_SAMPLE = 300_000
CHUNK = 100_000
AXIS_BIN = 256
HAND_WINDOW = 0.05 * 2 * np.pi
AXIS_RINGS = 5
CORE_STEP = 16
CORE_ROW_SLICES = 32
CORE_LEAST_TURNS = 2.0
DUPLICATE, PRESSED, OVERLAP_NEXT, OVERLAP_PREVIOUS, OVERLAP_OTHER, CONFLICT, ORDER, TOUCHING = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
)


def manifest():
    return json.loads(MANIFEST.read_text())


def definition(name):
    data = manifest()
    control = data["controls"][name]
    return control | {"name": name, "scan_entry": data["scans"][control["scan"]]}


def zone_codes(scan_entry, z):
    """Zone code per level-0 z (index into ZONES); 255 where no zone is defined."""
    out = np.full(np.shape(z), 255, dtype=np.uint8)
    for low, high, zone in scan_entry["zones"]:
        out[(z >= low) & (z < high)] = ZONES.index(zone)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mesh_folders(sample, scan):
    """Tifxyz folders of a scan's published meshes: cache/open_data in the outputs (meshes fetched where the library's
    mirror holds empty files) first, then the library's mirror; each segment once, and only with a non-empty z.tif."""
    found = {}
    for root in (MIRROR, truths.OPEN_DATA):
        for folder in sorted(root.glob(f"{sample}/segments/*/mesh/*-on-{scan}-*um.tifxyz")):
            z = folder / "z.tif"
            if z.exists() and z.stat().st_size > 0:
                found.setdefault(folder.parent.parent.name, folder)
    return [found[segment] for segment in sorted(found)]


WRAP_NAME = re.compile(r"-w(\d+)(?:-(\d+))?(?:_|$)")


def wrap_of(segment):
    """First wrap of a wrap-named segment (-wNNN_ or -wNNN-MMM), -1 without a name."""
    match = WRAP_NAME.search(segment)
    return int(match.group(1)) if match else -1


def wrap_end(segment):
    """Last wrap of a wrap-named segment: MMM of -wNNN-MMM, else its first wrap."""
    match = WRAP_NAME.search(segment)
    if not match:
        return -1
    return int(match.group(2) or match.group(1))


def load_grid(folder):
    xyz = np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"], axis=-1).astype(np.float64)
    valid = np.all(xyz > 0, axis=-1) & np.all(np.isfinite(xyz), axis=-1)
    return xyz, valid


def grid_normals(xyz, valid):
    """Unit normals from central differences on the grid (one-sided at edges); NaN where a direction has no step."""
    tangents = []
    for axis in (0, 1):
        ahead = np.roll(xyz, -1, axis)
        behind = np.roll(xyz, 1, axis)
        ok_ahead = np.roll(valid, -1, axis)
        ok_behind = np.roll(valid, 1, axis)
        edge = [slice(None)] * 2
        edge[axis] = -1
        ok_ahead[tuple(edge)] = False
        edge[axis] = 0
        ok_behind[tuple(edge)] = False
        t = np.where(ok_ahead[..., None], ahead, xyz) - np.where(ok_behind[..., None], behind, xyz)
        t[~(ok_ahead | ok_behind)] = np.nan
        tangents.append(t)
    n = np.cross(tangents[0], tangents[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[~valid] = np.nan
    return n


def rings_of(xyz, valid):
    """(z, cx, cy) of a circle fitted to every grid row that closes a turn (at least 1.8 pi about its fitted centre)."""
    found = []
    for r in range(xyz.shape[0]):
        ok = valid[r]
        if ok.sum() < 50:
            continue
        p = xyz[r, ok]
        a = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
        cx, cy, _ = np.linalg.lstsq(a, (p[:, :2] ** 2).sum(axis=1), rcond=None)[0]
        turn = np.ptp(np.unwrap(np.arctan2(p[:, 1] - cy, p[:, 0] - cx)))
        if turn >= 1.8 * np.pi:
            found.append((float(np.median(p[:, 2])), cx, cy))
    return found


def ring_axis(found, bin_z=AXIS_BIN, least=AXIS_RINGS):
    """Axis knots from the truth's own rings (rings_of over every mesh grid): the median centre per z bin. None when
    fewer than three bins have enough rings."""
    if not found:
        return None
    found = np.array(found)
    knots = []
    for low in np.arange(np.floor(found[:, 0].min() / bin_z) * bin_z, found[:, 0].max() + 1, bin_z):
        inside = found[(found[:, 0] >= low) & (found[:, 0] < low + bin_z)]
        if len(inside) >= least:
            knots.append((low + bin_z / 2, np.median(inside[:, 1]), np.median(inside[:, 2])))
    return np.array(knots) if len(knots) >= 3 else None


def core_rows(xyz, valid, rows_by_centre, bin_z=AXIS_BIN, within=CORE_ROW_SLICES):
    """Adds this grid's row nearest each axis bin centre (within `within` slices) to rows_by_centre, as polyline pieces
    (x, y) split where the columns skip, every second vertex."""
    known = np.flatnonzero(valid.sum(axis=1) >= 2)
    if not len(known):
        return
    zrow = np.array([np.median(xyz[r, valid[r], 2]) for r in known])
    for centre in np.arange(bin_z / 2, zrow.max() + bin_z / 2, bin_z):
        i = int(np.argmin(np.abs(zrow - centre)))
        if abs(zrow[i] - centre) > within:
            continue
        r = known[i]
        cols = np.flatnonzero(valid[r])
        line = xyz[r, cols, :2].astype(np.float32)
        for piece in np.split(np.arange(len(cols)), np.flatnonzero(np.diff(cols) > 1) + 1):
            piece = piece[::2] if len(piece) > 2 else piece
            if len(piece) >= 2:
                rows_by_centre.setdefault(float(centre), []).append(line[piece])


def winding_field(pieces, xs, ys):
    """Winding number (turns, signed) of the polylines about every point of the grid ys x xs: per piece, the signed
    crossings of the ray from the point towards +x plus the angle from its first to its last vertex (taken in [0, 2pi)
    from that ray)."""
    a = np.concatenate([p[:-1] for p in pieces]).astype(np.float64)
    b = np.concatenate([p[1:] for p in pieces]).astype(np.float64)
    field = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        up = (a[:, 1] <= y) & (b[:, 1] > y)
        down = (b[:, 1] <= y) & (a[:, 1] > y)
        s = up | down
        if not s.any():
            continue
        t = (y - a[s, 1]) / (b[s, 1] - a[s, 1])
        xc = a[s, 0] + t * (b[s, 0] - a[s, 0])
        order = np.argsort(xc)
        xc, sign = xc[order], np.where(up[s], 1.0, -1.0)[order]
        right = np.concatenate([np.cumsum(sign[::-1])[::-1], [0.0]])
        field[i] = right[np.searchsorted(xc, xs, side="right")]
    gx, gy = np.meshgrid(xs, ys)
    for p in pieces:
        last = np.arctan2(p[-1, 1] - gy, p[-1, 0] - gx) % (2 * np.pi)
        first = np.arctan2(p[0, 1] - gy, p[0, 0] - gx) % (2 * np.pi)
        field += (last - first) / (2 * np.pi)
    return field


def core_axis(rows_by_centre, step=CORE_STEP, least=CORE_LEAST_TURNS):
    """Axis knots (z, x, y) inside the innermost traced turn. Per bin centre, the traced rows (core_rows) wind around
    some points more than around any other: those inside every turn they trace. Of the points (every `step` voxels)
    within half a turn of the most, the one farthest from any row vertex. A fitted centre can lie outside the inner turns
    of a crushed roll; the angle about it then misses turns. None when fewer than three bins have rows winding at least
    `least` turns about some point."""
    knots = []
    for centre, pieces in sorted(rows_by_centre.items()):
        every = np.concatenate(pieces).astype(np.float64)
        lo, hi = every.min(axis=0), every.max(axis=0)
        xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
        if len(xs) < 2 or len(ys) < 2:
            continue
        field = np.abs(winding_field(pieces, xs, ys))
        if field.max() < least:
            continue
        iy, ix = np.nonzero(field >= field.max() - 0.5)
        candidates = np.column_stack([xs[ix], ys[iy]])
        clearance, _ = cKDTree(every).query(candidates)
        x, y = candidates[int(np.argmax(clearance))]
        knots.append((centre, float(x), float(y)))
    return np.array(knots) if len(knots) >= 3 else None


def axis_knots(sample, scan, found, points, normals, cores=None):
    """Axis knots (z, x, y) in level-0 voxels: inside the innermost traced turn where the truth traces closed turns
    (core_axis over core_rows; since version 4), else from the truth's closed rings (found by rings_of), else v1's
    wrap-truth knots for every band of this scan, else fitted to the truth's normals. The winding is the angle about
    this axis, unwrapped."""
    core = core_axis(cores) if cores else None
    if core is not None:
        return (
            core,
            f"inside the innermost traced turn: the point the truth rows wind around most, {AXIS_BIN} slices per knot",
        )
    rings = ring_axis(found)
    if rings is not None:
        return rings, f"median centre of circles fitted to closed truth rings, {AXIS_BIN} slices per knot"
    rows = []
    for path in sorted(truths.WRAP_TRUTH.glob(f"{sample}-{scan}-rows-*.npz")):
        if path.stem.endswith("tier-a"):
            continue
        data = np.load(path)
        scale = 2 ** int(data["level"])
        rows.append(data["knots"] * scale + 0.5 * (scale - 1))
    if rows:
        knots = np.concatenate(rows)
        knots = knots[np.argsort(knots[:, 0])]
        _, first = np.unique(np.round(knots[:, 0]), return_index=True)
        return knots[first], "v1 wrap truth knots (all bands of the scan)"
    return truths.fitted_axis(points, normals, slabs=12), "fitted to the truth normals"


def handedness(grids, window=HAND_WINDOW):
    """+1 when a sheet runs outward along +angle. On each grid row (one z) that spans more than a turn, the vertices
    a turn past the row's start lie at the same angle as its first ones: the sheet has gone outward if they are
    further from the axis. Votes are rows; without any, the sign of the angle-radius correlation (weak where the
    roll is not round)."""
    votes = 0
    rows = 0
    for g in grids:
        phi, radius = g["phi"], g["radius"]
        for r in range(phi.shape[0]):
            known = np.isfinite(phi[r])
            if known.sum() < 20:
                continue
            p, rad = phi[r, known], radius[r, known]
            low, high = p.min(), p.max()
            if high - low < 2 * np.pi + window:
                continue
            head = (p >= low) & (p < low + window)
            tail = (p >= low + 2 * np.pi) & (p < low + 2 * np.pi + window)
            far = (p <= high) & (p > high - window)
            back = (p <= high - 2 * np.pi) & (p > high - 2 * np.pi - window)
            for a, b in ((head, tail), (back, far)):
                if a.any() and b.any():
                    votes += np.sign(np.median(rad[b]) - np.median(rad[a]))
                    rows += 1
    if rows:
        return (1 if votes >= 0 else -1), {"rows": rows, "net": int(votes)}, "rows passing their start angle again"
    total = 0.0
    for g in grids:
        known = np.isfinite(g["phi"]) & g["valid"]
        if known.sum() > 100 and np.ptp(g["phi"][known]) > 0.5:
            total += np.corrcoef(g["phi"][known], g["radius"][known])[0, 1] * known.sum()
    return (1 if total >= 0 else -1), {"rows": 0}, "angle-radius correlation (no row spans a turn)"


def unwrap_mesh(xyz, valid, knots):
    """Angle about the axis unwrapped along each grid row (a row is one z), then the rows joined: two pieces (runs of
    valid columns) of neighbouring rows are joined at their shared column farthest from the axis. A row whose own
    turns are miscounted (the axis outside its inner turns at that z) then stays wrong alone; over a spanning tree of
    the whole grid it passed its error on to every row reached through it (PHercParis4's 0701 w010-027 lost two turns
    in every row so, from its top row). NaN outside the largest group of joined pieces. Returns the grid of angles and
    the share of valid vertices dropped."""
    theta = np.full(valid.shape, np.nan)
    radius = np.full(valid.shape, np.nan)
    theta[valid], radius[valid] = truths.angle(knots, xyz[valid])
    phi = np.full(valid.shape, np.nan)
    piece = np.full(valid.shape, -1, dtype=np.int64)
    count = 0
    for r in range(valid.shape[0]):
        cols = np.flatnonzero(valid[r])
        if not len(cols):
            continue
        for run in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1):
            phi[r, run] = np.unwrap(theta[r, run])
            piece[r, run] = count
            count += 1
    if not count:
        return phi, 0.0
    links = []
    for r in range(valid.shape[0] - 1):
        shared = np.flatnonzero(valid[r] & valid[r + 1])
        if not len(shared):
            continue
        a, b = piece[r, shared], piece[r + 1, shared]
        reach = np.minimum(radius[r, shared], radius[r + 1, shared])
        pairs, inverse = np.unique(np.stack([a, b], axis=1), axis=0, return_inverse=True)
        for i, (pa, pb) in enumerate(pairs):
            c = shared[inverse.ravel() == i]
            c = c[int(np.argmax(reach[inverse.ravel() == i]))]
            step = np.angle(np.exp(1j * (theta[r + 1, c] - theta[r, c])))
            links.append((pa, pb, np.round((phi[r, c] + step - phi[r + 1, c]) / (2 * np.pi))))
    links = np.array(links).reshape(-1, 3)
    graph = coo_matrix(
        (np.ones(len(links)), (links[:, 0].astype(np.int64), links[:, 1].astype(np.int64))), shape=(count, count)
    )
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(piece[valid], minlength=count)
    keep = int(np.argmax(np.bincount(label, weights=sizes)))
    root = int(np.flatnonzero(label == keep)[np.argmax(sizes[label == keep])])
    order, parent = breadth_first_order(graph, root, directed=False, return_predecessors=True)
    turns = {}
    for pa, pb, t in links:
        turns[(int(pa), int(pb))] = t
        turns[(int(pb), int(pa))] = -t
    offset = np.zeros(count)
    for p in order[1:]:
        offset[p] = offset[parent[p]] + turns[(int(parent[p]), int(p))]
    kept = label[np.maximum(piece, 0)] == keep
    out = np.where(valid & kept, phi + 2 * np.pi * offset[np.maximum(piece, 0)], np.nan)
    return out, float(1.0 - (valid & kept).sum() / max(valid.sum(), 1))


def turn_ends(phi, share=0.05):
    """The grid's first and last turn (angle / 2pi), robust to a few miscounted rows: the `share` quantile of the rows'
    first turns and the 1 - `share` quantile of their last turns."""
    known = np.isfinite(phi)
    rows = np.flatnonzero(known.sum(axis=1) >= 2)
    if not len(rows):
        values = phi[known] / (2 * np.pi)
        return float(values.min()), float(values.max())
    first = np.array([np.nanmin(phi[r]) for r in rows]) / (2 * np.pi)
    last = np.array([np.nanmax(phi[r]) for r in rows]) / (2 * np.pi)
    return float(np.quantile(first, share)), float(np.quantile(last, 1 - share))


def neighbours(points, normals, pick, radius, k):
    """For the picked points, CHUNK at a time: neighbour index (-1 none), distance along the picked normal (signed),
    across, total. A generator, so only one chunk's neighbours are held at a time."""
    tree = cKDTree(points)
    for start in range(0, len(pick), CHUNK):
        rows = pick[start : start + CHUNK]
        distance, found = tree.query(points[rows], k=k, distance_upper_bound=radius, workers=-1)
        hit = np.isfinite(distance)
        other = np.where(hit, found, 0)
        offset = points[other] - points[rows][:, None, :]
        along = np.einsum("ijk,ik->ij", offset, normals[rows])
        total = np.where(hit, distance, np.inf)
        across = np.sqrt(np.maximum(np.where(hit, distance, 0) ** 2 - along**2, 0))
        yield rows, np.where(hit, other, -1), along, across, total


def pair_votes(points, normals, mesh, tau, voxel_um, spacing_um, rng):
    """Counts of round(tau_b - tau_a) for contacts (within 40 µm along a's normal, lateral within 0.75 spacing), for
    one surface traced twice (within 16 µm), and for the radially nearest outward vertex of another mesh (beyond
    80 µm, within 37 degrees of the normal)."""
    usable = np.flatnonzero(np.isfinite(tau) & np.all(np.isfinite(normals), axis=1))
    pick = rng.choice(usable, min(VOTE_SAMPLE, len(usable)), replace=False)
    contact, radial, surface = Counter(), Counter(), Counter()
    lateral = LATERAL_SHARE * spacing_um / voxel_um
    for rows, other, along, across, total in neighbours(points, normals, pick, RADIAL_UM / voxel_um, 64):
        valid = (other >= 0) & (mesh[np.maximum(other, 0)] != mesh[rows][:, None])
        valid &= np.isfinite(tau[np.maximum(other, 0)])
        delta = np.round(np.nan_to_num(tau[np.maximum(other, 0)] - tau[rows][:, None])).astype(np.int64)
        touch = valid & (np.abs(along) <= SAME_SHEET_UM / voxel_um) & (across <= lateral)
        a, j = np.nonzero(touch)
        tally(contact, mesh[rows[a]], mesh[other[a, j]], delta[a, j])
        a, j = np.nonzero(touch & (np.abs(along) <= SAME_SURFACE_UM / voxel_um))
        tally(surface, mesh[rows[a]], mesh[other[a, j]], delta[a, j])
        outward = valid & (-along > PRESSED_UM / voxel_um) & (-along >= 0.8 * total)
        first = np.argmin(np.where(outward, total, np.inf), axis=1)
        every = np.arange(len(rows))
        has = outward[every, first]
        tally(radial, mesh[rows[has]], mesh[other[every, first][has]], delta[every, first][has])
    return contact, radial, surface


def tally(counter, a, b, d):
    if not len(d):
        return
    keys, counts = np.unique(np.stack([a, b, d]).astype(np.int64), axis=1, return_counts=True)
    for (ka, kb, kd), c in zip(keys.T.tolist(), counts.tolist(), strict=True):
        counter[(ka, kb, kd)] += c


def chain_offsets(wraps, radial, ends=None):
    """k per mesh chained along consecutive wrap names: k_b - k_a = 1 - mode(round(tau_b - tau_a)) over a's nearest
    outward vertices on b, b named one wrap above a's last wrap (ends: last wrap per mesh, for -wNNN-MMM names).
    Where several meshes carry the next name, a's link goes to the one not yet chained with most votes. Returns k
    (NaN where unchained), the chain record and the run of each mesh (-1 unnamed)."""
    ends = wraps if ends is None else ends
    order = [m for m in np.argsort(wraps, kind="stable") if wraps[m] >= 0]
    k = np.full(len(wraps), np.nan)
    run = np.full(len(wraps), -1)
    links = []
    runs = 0
    for m in order:
        if np.isnan(k[m]):
            k[m] = 0.0
            run[m] = runs
            runs += 1
        nxt = [n for n in order if wraps[n] == ends[m] + 1]
        if not nxt:
            continue
        tallies = {n: Counter({d: c for (a, b, d), c in radial.items() if a == m and b == n}) for n in nxt}
        free = [n for n in nxt if np.isnan(k[n])] or nxt
        n = max(free, key=lambda n: sum(tallies[n].values()))
        votes = tallies[n]
        if not votes:
            links.append({"from": int(wraps[m]), "to": int(wraps[n]), "votes": 0})
            continue
        delta, count = votes.most_common(1)[0]
        k[n] = k[m] + 1 - delta
        run[n] = run[m]
        links.append(
            {
                "from": int(wraps[m]),
                "to": int(wraps[n]),
                "votes": int(sum(votes.values())),
                "purity": round(count / sum(votes.values()), 4),
                "step": int(1 - delta),
            }
        )
    return k, {"runs": runs, "links": links}, run


def link_frames(frame_of_mesh, k, surface):
    """Frames joined where their meshes trace one surface: frame b's windings move by -mode(W_b - W_a) over the
    same-surface votes between frames a and b, when there are at least LINK_VOTES of them and LINK_PURITY on the
    mode; the pairs with most votes first. Returns the frame per mesh (renumbered), k, and every pair with votes."""
    frame_of_mesh, k = frame_of_mesh.copy(), k.copy()
    pairs = {}
    for (a, b, d), c in surface.items():
        fa, fb = int(frame_of_mesh[a]), int(frame_of_mesh[b])
        if fa < 0 or fb < 0 or fa == fb or not (np.isfinite(k[a]) and np.isfinite(k[b])):
            continue
        step = int(d + k[b] - k[a])
        key, step = ((fa, fb), step) if fa < fb else ((fb, fa), -step)
        pairs.setdefault(key, Counter())[step] += c
    parent = {int(f): int(f) for f in np.unique(frame_of_mesh[frame_of_mesh >= 0])}
    offset = dict.fromkeys(parent, 0)

    def find(f):
        """Root of frame f and o_f - o_root."""
        total = 0
        while parent[f] != f:
            total += offset[f]
            f = parent[f]
        return f, total

    record = []
    for (fa, fb), votes in sorted(pairs.items(), key=lambda item: -sum(item[1].values())):
        total = sum(votes.values())
        mode, count = votes.most_common(1)[0]
        entry = {"frames": [fa, fb], "votes": total, "purity": round(count / total, 4), "offset": -mode}
        if total >= LINK_VOTES and count / total >= LINK_PURITY:
            (ra, sa), (rb, sb) = find(fa), find(fb)
            if ra == rb:
                entry["joined"] = "already"
                entry["consistent"] = bool(sb - sa == -mode)
            else:
                if ra < rb:
                    parent[rb], offset[rb] = ra, -mode - sb + sa
                else:
                    parent[ra], offset[ra] = rb, mode + sb - sa
                entry["joined"] = "yes"
        else:
            entry["joined"] = "no"
        record.append(entry)
    roots = sorted({find(f)[0] for f in parent})
    for m in range(len(frame_of_mesh)):
        if frame_of_mesh[m] >= 0:
            root, shift = find(int(frame_of_mesh[m]))
            k[m] += shift
            frame_of_mesh[m] = roots.index(root)
    return frame_of_mesh, k, record


def check_pairs(contact, radial, k, frame_of_mesh):
    """Radial order and contacts judged with the solved winding, W_b - W_a = round(tau_b - tau_a) + k_b - k_a, for
    pairs of meshes in one frame (the offset between frames is unknown)."""
    outward, touch = Counter(), Counter()
    for (a, b, d), c in radial.items():
        if np.isfinite(k[a]) and np.isfinite(k[b]) and frame_of_mesh[a] == frame_of_mesh[b]:
            outward[int(d + k[b] - k[a])] += c
    for (a, b, d), c in contact.items():
        if np.isfinite(k[a]) and np.isfinite(k[b]) and frame_of_mesh[a] == frame_of_mesh[b]:
            touch[int(abs(d + k[b] - k[a]))] += c
    n_out, n_touch = sum(outward.values()), sum(touch.values())
    return {
        "radial_pairs": n_out,
        "radial_step_1": round(outward[1] / n_out, 4) if n_out else None,
        "radial_step_2_or_more": round(sum(c for s, c in outward.items() if s >= 2) / n_out, 4) if n_out else None,
        "radial_step_0_or_less": round(sum(c for s, c in outward.items() if s <= 0) / n_out, 4) if n_out else None,
        "contact_pairs": n_touch,
        "contact_same_winding": round(touch[0] / n_touch, 4) if n_touch else None,
        "contact_one_apart": round(touch[1] / n_touch, 4) if n_touch else None,
        "contact_two_or_more": round(sum(c for s, c in touch.items() if s >= 2) / n_touch, 4) if n_touch else None,
    }


def vertex_flags(points, normals, mesh, winding, frame, mesh_wrap, size_rank, voxel_um, spacing_um):
    """Flags per vertex from its neighbours on other meshes. Windings compare only within one frame; another frame's
    vertex within 40 µm along the normal is touching (and a duplicate of the larger mesh for areas), within 80 µm
    pressed, whatever its winding, since the offset between frames is unknown."""
    flags = np.zeros(len(points), dtype=np.uint8)
    lateral = LATERAL_SHARE * spacing_um / voxel_um
    usable = np.flatnonzero(np.isfinite(winding) & np.all(np.isfinite(normals), axis=1))
    radius = float(np.hypot(PRESSED_UM / voxel_um, lateral))
    for rows, other, along, across, _ in neighbours(points, normals, usable, radius, 24):
        o = np.maximum(other, 0)
        valid = (other >= 0) & (mesh[o] != mesh[rows][:, None]) & np.isfinite(winding[o]) & (across <= lateral)
        one = frame[o] == frame[rows][:, None]
        step = np.where(one, np.abs(winding[o] - winding[rows][:, None]), np.nan)
        near = np.abs(along) <= SAME_SHEET_UM / voxel_um
        close = np.abs(along) <= PRESSED_UM / voxel_um
        same = valid & one & near & (step < 0.5)
        across_frames = valid & ~one
        pressed = valid & close & ((one & (step >= 0.5)) | ~one)
        larger = size_rank[mesh[o]] < size_rank[mesh[rows]][:, None]
        mine = mesh_wrap[mesh[rows]][:, None]
        theirs = mesh_wrap[mesh[o]]
        duplicate = (same | (across_frames & near)) & larger
        flags[rows] |= np.where(duplicate.any(axis=1), DUPLICATE, 0).astype(np.uint8)
        flags[rows] |= np.where(pressed.any(axis=1), PRESSED, 0).astype(np.uint8)
        named = (mine >= 0) & (theirs >= 0)
        flags[rows] |= np.where((same & named & (theirs == mine + 1)).any(axis=1), OVERLAP_NEXT, 0).astype(np.uint8)
        flags[rows] |= np.where((same & named & (theirs == mine - 1)).any(axis=1), OVERLAP_PREVIOUS, 0).astype(np.uint8)
        other_same = same & ~(named & (np.abs(theirs - mine) == 1))
        flags[rows] |= np.where(other_same.any(axis=1), OVERLAP_OTHER, 0).astype(np.uint8)
        conflict = valid & one & near & (step >= 1.5)
        flags[rows] |= np.where(conflict.any(axis=1), CONFLICT, 0).astype(np.uint8)
        touching = valid & near & ((one & (step >= 0.5)) | ~one)
        flags[rows] |= np.where(touching.any(axis=1), TOUCHING, 0).astype(np.uint8)
    for rows, other, along, _, total in neighbours(points, normals, usable, RADIAL_UM / voxel_um, 64):
        o = np.maximum(other, 0)
        valid = (other >= 0) & (mesh[o] != mesh[rows][:, None]) & np.isfinite(winding[o])
        valid &= frame[o] == frame[rows][:, None]
        outward = valid & (-along > PRESSED_UM / voxel_um) & (-along >= 0.8 * total)
        first = np.argmin(np.where(outward, total, np.inf), axis=1)
        every = np.arange(len(rows))
        has = outward[every, first]
        wrong = has & (winding[o[every, first]] - winding[rows] < 0.5)
        flags[rows] |= np.where(wrong, ORDER, 0).astype(np.uint8)
    return flags


def crossings(grid, step, z_range, knots):
    """Points where a mesh grid crosses the planes z0 + i * step inside z_range, on vertical (row) edges: plane z,
    x, y, fractional row, column, winding and normal (towards the axis) interpolated along the edge."""
    xyz, valid, w, n = grid["xyz"], grid["valid"], grid["winding"], grid["normals"]
    ok = valid[:-1] & valid[1:] & np.isfinite(w[:-1]) & np.isfinite(w[1:])
    za, zb = xyz[:-1, :, 2], xyz[1:, :, 2]
    low, high = np.minimum(za, zb), np.maximum(za, zb)
    z0 = z_range[0]
    first = np.ceil((low - z0) / step).astype(np.int64)
    last = np.ceil((high - z0) / step).astype(np.int64) - 1
    first = np.maximum(first, 0)
    last = np.minimum(last, (z_range[1] - 1 - z0) // step)
    count = np.where(ok & (high > low), np.maximum(last - first + 1, 0), 0)
    r, c = np.nonzero(count)
    repeat = count[r, c]
    plane = np.repeat(first[r, c], repeat) + (np.arange(repeat.sum()) - np.repeat(np.cumsum(repeat) - repeat, repeat))
    r, c = np.repeat(r, repeat), np.repeat(c, repeat)
    zp = z0 + plane * step
    t = (zp - za[r, c]) / (zb[r, c] - za[r, c])
    point = xyz[r, c] + t[:, None] * (xyz[r + 1, c] - xyz[r, c])
    normal = n[r, c] + t[:, None] * (n[r + 1, c] - n[r, c])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    cx = np.interp(point[:, 2], knots[:, 0], knots[:, 1])
    cy = np.interp(point[:, 2], knots[:, 0], knots[:, 2])
    toward = (cx - point[:, 0]) * normal[:, 0] + (cy - point[:, 1]) * normal[:, 1]
    normal[toward < 0] *= -1
    order = np.lexsort((c, plane))
    return {
        "z": zp[order].astype(np.float32),
        "xy": point[order, :2].astype(np.float32),
        "row": (r + t)[order].astype(np.float32),
        "col": c[order].astype(np.int32),
        "winding": (w[r, c] + t * (w[r + 1, c] - w[r, c]))[order],
        "normal": normal[order].astype(np.float32),
        "vertex_row": np.where(t < 0.5, r, r + 1)[order].astype(np.int32),
    }


def write_npz(path, **arrays):
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_tifxyz(folder, xyz, keep, winding, meta):
    folder.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate("xyz"):
        tifffile.imwrite(folder / f"{a}.tif", np.where(keep, xyz[..., i], -1).astype(np.float32))
    tifffile.imwrite(folder / "w.tif", np.where(keep, winding, np.nan).astype(np.float32))
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def build(name, rng_seed=0):
    spec = definition(name)
    entry = spec["scan_entry"]
    sample, scan, voxel_um = entry["sample"], entry["scan"], entry["voxel_um"]
    z_range = tuple(spec["z"])
    out = PACKAGES / name
    if (out / "control.json").exists():
        raise FileExistsError(f"{out} exists; a package is frozen once built (remove it by hand to rebuild)")
    started = time.time()
    folders = mesh_folders(sample, scan)
    grids, sources, rings, cores = [], {}, [], {}
    for folder in folders:
        segment = folder.parent.parent.name
        xyz, valid = load_grid(folder)
        inside = valid & (xyz[..., 2] >= z_range[0]) & (xyz[..., 2] < z_range[1])
        sources[segment] = {
            "folder": str(folder),
            "sha256": {
                f: sha256(folder / f) for f in ("x.tif", "y.tif", "z.tif", "meta.json") if (folder / f).exists()
            },
            "in_range": int(inside.sum()),
        }
        if inside.any():
            rings += rings_of(xyz, valid)
            core_rows(xyz, valid, cores)
            near = valid & (xyz[..., 2] >= z_range[0] - GRID_MARGIN) & (xyz[..., 2] < z_range[1] + GRID_MARGIN)
            rows = np.flatnonzero(near.any(axis=1))
            r0, r1 = rows.min(), rows.max() + 1
            grids.append(
                {
                    "segment": segment,
                    "folder": folder,
                    "xyz": xyz[r0:r1].copy(),
                    "valid": valid[r0:r1].copy(),
                    "inside": inside[r0:r1].copy(),
                    "row0": int(r0),
                    "scale": json.loads((folder / "meta.json").read_text()).get("scale", [0.05, 0.05])
                    if (folder / "meta.json").exists()
                    else [0.05, 0.05],
                }
            )
    if not grids:
        raise ValueError(f"no mesh vertex of {sample} {scan} in z {z_range}")
    for g in grids:
        g["normals"] = grid_normals(g["xyz"], g["valid"])
    knots, axis_source = axis_knots(
        sample,
        scan,
        rings,
        np.concatenate([g["xyz"][g["inside"]] for g in grids]),
        np.concatenate([g["normals"][g["inside"]] for g in grids]),
        cores,
    )
    del cores
    for g in grids:
        full_xyz, full_valid = load_grid(g["folder"])
        phi, g["dropped_parts"] = unwrap_mesh(full_xyz, full_valid, knots)
        g["turn_quantiles"] = turn_ends(phi)
        g["phi"] = phi[g["row0"] : g["row0"] + g["valid"].shape[0]].copy()
        del full_xyz, full_valid, phi
        _, g["radius"] = truths.angle(knots, g["xyz"].reshape(-1, 3))
        g["radius"] = g["radius"].reshape(g["valid"].shape)
    sense, sense_votes, sense_source = handedness(grids)
    wraps = np.array([wrap_of(g["segment"]) for g in grids], dtype=np.int64)
    named = bool((wraps >= 0).sum() >= 2)
    mesh_parts, row_parts, col_parts = [], [], []
    for m, g in enumerate(grids):
        r, c = np.nonzero(g["inside"])
        mesh_parts.append(np.full(len(r), m, dtype=np.int16))
        row_parts.append((r + g["row0"]).astype(np.int32))
        col_parts.append(c.astype(np.int32))
    mesh = np.concatenate(mesh_parts)
    row, col = np.concatenate(row_parts), np.concatenate(col_parts)
    points = np.concatenate([g["xyz"][g["inside"]] for g in grids])
    normals = np.concatenate([g["normals"][g["inside"]] for g in grids])
    tau = np.concatenate([sense * g["phi"][g["inside"]] / (2 * np.pi) for g in grids])
    spacing_um = float(
        np.median(
            np.concatenate(
                [
                    np.linalg.norm(np.diff(g["xyz"], axis=1), axis=-1)[g["valid"][:, 1:] & g["valid"][:, :-1]]
                    for g in grids
                ]
            )
        )
        * voxel_um
    )
    cx = np.interp(points[:, 2], knots[:, 0], knots[:, 1])
    cy = np.interp(points[:, 2], knots[:, 0], knots[:, 2])
    toward = (cx - points[:, 0]) * normals[:, 0] + (cy - points[:, 1]) * normals[:, 1]
    normals[toward < 0] *= -1
    rng = np.random.default_rng(rng_seed)
    contact, radial, surface = pair_votes(points, normals, mesh, tau, voxel_um, spacing_um, rng)
    if named and spec.get("winding") == "names":
        k = np.full(len(grids), np.nan)
        for m in np.flatnonzero(wraps >= 0):
            low, high = grids[m]["turn_quantiles"]
            k[m] = wraps[m] - np.floor(low if sense > 0 else -high)
        chain = {"runs": 1, "links": [], "from": "names: each mesh anchored by its own name"}
        frame_of_mesh = np.where(np.isfinite(k), 0, -1)
        unnamed = np.flatnonzero(wraps < 0)
        k[unnamed] = 0.0
        frame_of_mesh[unnamed] = frame_of_mesh.max() + 1 + np.arange(len(unnamed))
    elif named:
        ends = np.array([wrap_end(g["segment"]) for g in grids], dtype=np.int64)
        k, chain, frame_of_mesh = chain_offsets(wraps, radial, ends)
        unnamed = np.flatnonzero(wraps < 0)
        k[unnamed] = 0.0
        frame_of_mesh[unnamed] = frame_of_mesh.max() + 1 + np.arange(len(unnamed))
    else:
        k, chain = np.zeros(len(grids)), {"runs": 0, "links": []}
        frame_of_mesh = np.arange(len(grids))
    frame_of_mesh, k, frame_links = link_frames(frame_of_mesh, k, surface)
    winding = tau + k[mesh]
    anchor = 0
    if named:
        starts = {
            m: np.quantile(winding[mesh == m][np.isfinite(winding[mesh == m])], 0.001)
            for m in range(len(grids))
            if wraps[m] >= 0 and np.isfinite(winding[mesh == m]).any()
        }
        anchor = Counter(int(wraps[m] - np.floor(s)) for m, s in starts.items()).most_common(1)[0][0]
        winding += anchor
        name_offsets = {grids[m]["segment"]: int(wraps[m] - np.floor(s)) - anchor for m, s in starts.items()}
    else:
        name_offsets = {}
    checks = check_pairs(contact, radial, k + 0.0, frame_of_mesh)
    sizes = np.bincount(mesh, minlength=len(grids))
    size_rank = np.argsort(np.argsort(-sizes, kind="stable"), kind="stable")
    flags = vertex_flags(points, normals, mesh, winding, frame_of_mesh[mesh], wraps, size_rank, voxel_um, spacing_um)
    area = np.zeros(len(points), dtype=np.float32)
    start = 0
    for g in grids:
        a = vertex_area(g["xyz"], g["valid"]) * (voxel_um * 1e-3) ** 2
        n = int(g["inside"].sum())
        area[start : start + n] = a[g["inside"]]
        start += n
    zone = zone_codes(entry, points[:, 2])
    frame = frame_of_mesh[mesh].astype(np.int32)
    frame[~np.isfinite(winding)] = -1
    out.mkdir(parents=True, exist_ok=False)
    names = np.array([g["segment"] for g in grids])
    write_npz(
        out / "truth.npz",
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
        mesh=mesh,
        row=row,
        col=col,
        winding=winding,
        frame=frame,
        area_mm2=area,
        zone=zone,
        flags=flags,
        names=names,
        mesh_wrap=wraps,
        sense=np.int64(sense),
        voxel_um=np.float64(voxel_um),
        spacing_um=np.float64(spacing_um),
        knots=knots,
    )
    planes = []
    meshes_dir = out / "meshes"
    start = 0
    for m, g in enumerate(grids):
        n = int(g["inside"].sum())
        grid_w = np.full(g["valid"].shape, np.nan)
        grid_w[g["inside"]] = winding[start : start + n]
        grid_flags = np.zeros(g["valid"].shape, dtype=np.uint8)
        grid_flags[g["inside"]] = flags[start : start + n]
        start += n
        r, c = np.nonzero(g["inside"])
        r0, r1, c0, c1 = r.min(), r.max() + 2, c.min(), c.max() + 2
        r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
        crop = (slice(r0, r1), slice(c0, c1))
        normals_toward = g["normals"].copy()
        gx = np.interp(g["xyz"][..., 2], knots[:, 0], knots[:, 1])
        gy = np.interp(g["xyz"][..., 2], knots[:, 0], knots[:, 2])
        flip = ((gx - g["xyz"][..., 0]) * normals_toward[..., 0] + (gy - g["xyz"][..., 1]) * normals_toward[..., 1]) < 0
        normals_toward[flip] *= -1
        full_w = np.full(g["valid"].shape, np.nan)
        known = np.isfinite(g["phi"])
        full_w[known] = sense * g["phi"][known] / (2 * np.pi) + k[m] + anchor
        found = crossings(
            {"xyz": g["xyz"], "valid": g["valid"], "winding": full_w, "normals": normals_toward},
            spec["crossing_step"],
            z_range,
            knots,
        )
        found["mesh"] = np.full(len(found["z"]), m, dtype=np.int16)
        found["flags"] = grid_flags[found["vertex_row"], found["col"]]
        found["row"] += g["row0"]
        found["vertex_row"] += g["row0"]
        planes.append(found)
        write_tifxyz(
            meshes_dir / f"{g['segment']}.tifxyz",
            g["xyz"][crop],
            g["inside"][crop],
            grid_w[crop],
            {
                "format": "tifxyz",
                "type": "seg",
                "scale": g["scale"],
                "segment": g["segment"],
                "source": str(g["folder"]),
                "wrap_name": int(wraps[m]),
                "frame": int(frame_of_mesh[m]),
                "grid_origin_row_col": [int(r0 + g["row0"]), int(c0)],
                "z_range": list(z_range),
                "winding_range": [round(float(np.nanmin(grid_w)), 4), round(float(np.nanmax(grid_w)), 4)],
                "note": "x, y, z level-0 voxels (-1 outside the control's z range); w.tif continuous winding",
            },
        )
    joined = {key: np.concatenate([p.pop(key) for p in planes]) for key in list(planes[0])}
    del planes
    order = np.lexsort((joined["col"], joined["mesh"], joined["z"]))
    for key in joined:
        joined[key] = joined[key][order]
    del order
    joined["zone"] = zone_codes(entry, joined["z"])
    write_npz(out / "crossings.npz", **joined, names=names, step=np.int64(spec["crossing_step"]))
    (out / "axis.json").write_text(json.dumps({"knots_z_x_y": knots.tolist(), "source": axis_source}, indent=1))
    kept = (flags & DUPLICATE) == 0
    by_zone = {
        ZONES[z]: {
            "vertices": int((zone == z).sum()),
            "area_cm2": round(float(area[(zone == z) & kept].sum() / 100), 2),
            "pressed_share": round(float(((flags & PRESSED) > 0)[(zone == z) & kept].mean()), 4)
            if ((zone == z) & kept).any()
            else None,
            "unjudged_share": round(float(((flags & (CONFLICT | ORDER)) > 0)[(zone == z) & kept].mean()), 4)
            if ((zone == z) & kept).any()
            else None,
            "touching_share": round(float(((flags & TOUCHING) > 0)[(zone == z) & kept].mean()), 4)
            if ((zone == z) & kept).any()
            else None,
            "crossings": int((joined["zone"] == z).sum()),
        }
        for z in np.unique(zone)
        if z < len(ZONES)
    }
    record = {
        "name": name,
        "version": VERSION,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 1),
        "code": commit_id(),
        "definition": {k: v for k, v in spec.items() if k != "scan_entry"},
        "scan": entry,
        "sources": sources,
        "meshes_in_range": len(grids),
        "named": named,
        "sense": sense,
        "sense_votes": sense_votes,
        "sense_source": sense_source,
        "axis_source": axis_source,
        "spacing_um": round(spacing_um, 2),
        "unwrap_dropped_share": {g["segment"]: round(g["dropped_parts"], 4) for g in grids if g["dropped_parts"]},
        "turns_in_range": {
            g["segment"]: round(float(np.ptp(winding[(mesh == m) & np.isfinite(winding)])), 3)
            for m, g in enumerate(grids)
            if np.isfinite(winding[mesh == m]).any()
        },
        "chain": chain,
        "frame_links": frame_links,
        "frames": {g["segment"]: int(frame_of_mesh[m]) for m, g in enumerate(grids)},
        "anchor": int(anchor),
        "name_start_offsets": {s: o for s, o in name_offsets.items() if o},
        "checks": checks,
        "zones": by_zone,
        "flags": {
            "duplicate": int(((flags & DUPLICATE) > 0).sum()),
            "pressed": int(((flags & PRESSED) > 0).sum()),
            "overlap_next": int(((flags & OVERLAP_NEXT) > 0).sum()),
            "overlap_previous": int(((flags & OVERLAP_PREVIOUS) > 0).sum()),
            "overlap_other": int(((flags & OVERLAP_OTHER) > 0).sum()),
            "conflict": int(((flags & CONFLICT) > 0).sum()),
            "order": int(((flags & ORDER) > 0).sum()),
            "touching": int(((flags & TOUCHING) > 0).sum()),
        },
    }
    record["files"] = {
        str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "control.json"
    }
    (out / "control.json").write_text(json.dumps(record, indent=1))
    return record


def verify(name):
    """Every file of a package and every source mesh file against the sha256 recorded when it was built."""
    out = PACKAGES / name
    record = json.loads((out / "control.json").read_text())
    bad = [f for f, h in record["files"].items() if not (out / f).exists() or sha256(out / f) != h]
    extra = [
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p.name != "control.json" and str(p.relative_to(out)) not in record["files"]
    ]
    changed = []
    for segment, source in record["sources"].items():
        folder = Path(source["folder"])
        for f, h in source["sha256"].items():
            if not (folder / f).exists() or sha256(folder / f) != h:
                changed.append(f"{segment}/{f}")
    return {"name": name, "files": len(record["files"]), "bad": bad, "extra": extra, "sources_changed": changed}


def load(name):
    """A package's truth as a dict of arrays plus its record."""
    out = PACKAGES / name
    data = dict(np.load(out / "truth.npz"))
    data["record"] = json.loads((out / "control.json").read_text())
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build packages (frozen: an existing package is never overwritten)")
    b.add_argument("names", nargs="+")
    v = sub.add_parser("verify", help="check package files and source meshes against their sha256")
    v.add_argument("names", nargs="*")
    sub.add_parser("list", help="controls in the manifest and whether their package exists")
    args = parser.parse_args()
    if args.command == "build":
        for name in args.names:
            record = build(name)
            print(
                json.dumps(
                    {k: record[k] for k in ("name", "seconds", "meshes_in_range", "named", "sense", "checks", "zones")}
                )
            )
    elif args.command == "verify":
        for name in args.names or sorted(p.parent.name for p in PACKAGES.glob("*/control.json")):
            print(json.dumps(verify(name)))
    else:
        for name, spec in manifest()["controls"].items():
            built = (PACKAGES / name / "control.json").exists()
            print(f"{name:16s} {spec['scan']:28s} z {spec['z'][0]:>6}-{spec['z'][1]:<6} {'built' if built else '-'}")


if __name__ == "__main__":
    main()
