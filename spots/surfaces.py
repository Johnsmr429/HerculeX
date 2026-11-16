"""A circled spot's pieces narrowed to the circle, rebuilt on the m7 surface prediction, and rendered 67 layers deep.

Narrowing keeps, of each piece the circle covers, the nodes whose sheet place lies within the circle's box widened
by MARGIN_MM; pieces holding less than LEAST_SHARE of the circle are left out.

Two surfaces are built from the same nodes:

    original   the piece as production reads it: bridged across small holes and smoothed by 1.5 cells
    m7         every node moved along its normal to the centre of the nearest run of predicted sheet within
               +-15 voxels of the published m7 surface prediction (threshold 0.2, full resolution), the offset field
               rid of jumps over 6 voxels from its 5x5 median, filled from its neighbours and smoothed by one cell,
               as the PHerc0343 First Letters winner snapped his windings (snap_meshes.py, defaults); then bridged
               like the original but not smoothed again

Rendering samples LAYERS = 67 layers one voxel apart along the normal turned away from the scroll axis: layer k lies
k - 33 voxels from the surface, so depth d = k - 33 and + is away from the axis, the convention of the PHerc0343
renders (whose 66 layers put the surface between layers 32 and 33). A 17-layer reader window centred on d spans
layers 33 + d - 8 .. 33 + d + 8, so d runs over -15..+15 with ten layers to spare at each end.
"""

import json
import shutil
import urllib.request
import warnings

import numpy as np
import tifffile
from scipy import ndimage

from sampling import column_sampling as sampler
from sampling.piece_images import axis_of_pieces, centre_at, surface_of
from sampling.touched_planes import mark_ranges
from scans.transfer.parallel_fetch import fetch_all
from spots.catalogue import node_places, piece_folder
from storage.brick_store import UNKNOWN, BrickStore
from storage.data_roots import SCAN_CATALOG, SPOT_CANDIDATES

LAYERS, MIDDLE = 67, 33
MARGIN_MM = 3.0
LEAST_SHARE = 0.08
MAX_OFF, JUMP, SIGMA = 15, 6.0, 1.0
GROW_ROUNDS, GROW_REACH = 15, 6  # up to 30 cells (4.5 mm) past an edge; new nodes look 6 voxels each way for sheet
SURFACES = ("original", "m7", "grown")


def work_of(spot_id):
    return SPOT_CANDIDATES / "spots" / spot_id


# ---- the published m7 prediction, fetched chunk by chunk -------------------------------------------------------


class Prediction:
    """The published m7 surface prediction of one scan, its chunks kept compressed as they arrive."""

    def __init__(self, scroll, scan, url=None):
        """url: the zarr's level-0 array (ROOT/PATH/0); without it the scan catalogue's products.jsonl names it."""
        if url:
            row = {"path": url}
        else:
            row = next(
                json.loads(line)
                for line in (SCAN_CATALOG / "products.jsonl").read_text().splitlines()
                if '"surface-prediction-zarr"' in line
                and json.loads(line)["sample"] == scroll
                and json.loads(line)["owner"] == scan
                and "-L0-" in json.loads(line)["path"]
            )
        self.base = url.rstrip("/") if url else f"{row['root']}/{row['path'].strip('/')}/0"
        self.folder = SPOT_CANDIDATES / "m7" / scroll / scan
        self.folder.mkdir(parents=True, exist_ok=True)
        meta = self.folder / ".zarray"
        if not meta.exists():
            meta.write_bytes(urllib.request.urlopen(f"{self.base}/.zarray", timeout=60).read())
        info = json.loads(meta.read_text())
        self.shape, self.chunks = tuple(info["shape"]), tuple(info["chunks"])
        self.path = row["path"]
        from numcodecs import Blosc

        self.codec = Blosc()

    def chunk_file(self, index):
        return self.folder / ".".join(map(str, index))

    def fetch(self, indices):
        wanted = [tuple(i) for i in indices if not self.chunk_file(tuple(i)).exists()]
        if not wanted:
            return 0

        def deliver(index, body):
            part = self.folder / f"{self.chunk_file(index).name}.part"
            part.write_bytes(b"" if body is None else body)
            part.rename(self.chunk_file(index))

        failures = fetch_all(self.base, [(i, "/".join(map(str, i))) for i in wanted], deliver)
        if failures:
            raise SystemExit(f"{len(failures)} m7 chunks could not be fetched, first {failures[0]}")
        return len(wanted)

    def chunk(self, index):
        body = self.chunk_file(index).read_bytes()
        if not body:
            return None
        return np.frombuffer(self.codec.decode(body), dtype=np.uint8).reshape(self.chunks)

    def sample(self, zyx):
        """The prediction at the nearest voxel of each (z, y, x) point; 0 outside the scan."""
        point = np.rint(zyx).astype(np.int64)
        out = np.zeros(len(point), dtype=np.uint8)
        inside = np.all((point >= 0) & (point < np.array(self.shape)), axis=1)
        index = point[inside] // np.array(self.chunks)
        keys, inverse = np.unique(index, axis=0, return_inverse=True)
        self.fetch(keys)
        where = np.nonzero(inside)[0]
        order = np.argsort(inverse.ravel(), kind="stable")
        bounds = np.searchsorted(inverse.ravel()[order], np.arange(len(keys) + 1))
        for k, key in enumerate(keys):
            block = self.chunk(tuple(key))
            if block is None:
                continue
            these = where[order[bounds[k] : bounds[k + 1]]]
            local = point[these] - key * np.array(self.chunks)
            out[these] = block[local[:, 0], local[:, 1], local[:, 2]]
        return out


# ---- narrowing ------------------------------------------------------------------------------------------------


def narrowed(spot, p, margin_mm=MARGIN_MM):
    """Rows and columns (r0, r1, c0, c1) of a piece's grid within the circle's box widened by margin_mm."""
    outline = np.asarray(spot["outline_mm"])
    low, high = outline.min(axis=0) - margin_mm, outline.max(axis=0) + margin_mm
    x, y = node_places(p)
    near = (x >= low[0]) & (x <= high[0]) & (y >= low[1]) & (y <= high[1])
    rows, columns = np.nonzero(near)
    if not len(rows):
        return None
    return int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1


def axis_for(sheet):
    kept = sheet.pieces.parent / "sheet" / "axis.json"
    if kept.exists():
        return np.array(json.loads(kept.read_text()), dtype=np.float64)
    return axis_of_pieces(sorted(sheet.pieces.glob("rows-*")))


# ---- snapping -------------------------------------------------------------------------------------------------


def nan_median5(field):
    h, w = field.shape
    padded = np.pad(field, 2, constant_values=np.nan)
    stack = np.stack([padded[i : i + h, j : j + w] for i in range(5) for j in range(5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def normconv(field, sigma):
    known = np.isfinite(field).astype(np.float32)
    value = np.where(np.isfinite(field), field, 0).astype(np.float32)
    num, den = ndimage.gaussian_filter(value, sigma), ndimage.gaussian_filter(known, sigma)
    out = np.full(field.shape, np.nan, dtype=np.float32)
    ok = den > 1e-3
    out[ok] = num[ok] / den[ok]
    return out


def nearest_run_centres(profile):
    """Per row of a boolean (n, 2K+1) profile, the offset of the centre of the predicted-sheet run nearest 0, sub-voxel."""
    n, width = profile.shape
    k = width // 2
    out = np.full(n, np.nan, dtype=np.float32)
    padded = np.concatenate([np.zeros((n, 1), bool), profile, np.zeros((n, 1), bool)], axis=1).astype(np.int8)
    edges = np.diff(padded, axis=1)
    smooth = ndimage.convolve1d(profile.astype(np.float32), [1, 2, 1], axis=1, mode="constant")
    for i in np.nonzero(profile.any(axis=1))[0]:
        starts, ends = np.nonzero(edges[i] == 1)[0], np.nonzero(edges[i] == -1)[0]
        centres = (starts + ends - 1) / 2.0 - k
        j = int(np.argmin(np.abs(centres)))
        c = min(max(round(centres[j] + k), 1), width - 2)
        y0, y1, y2 = smooth[i, c - 1], smooth[i, c], smooth[i, c + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
        out[i] = centres[j] if abs(frac) > 0.5 else 0.5 * (centres[j] + (c - k + frac))
    return out


def snapped(xyz, normal, prediction):
    """A grid's nodes moved onto the nearest predicted sheet, and what the move did."""
    valid = xyz[0] > 0
    rows, columns = np.nonzero(valid)
    offsets = np.arange(-MAX_OFF, MAX_OFF + 1, dtype=np.float32)
    base = xyz[:, rows, columns].T  # x, y, z
    aim = normal[:, rows, columns].T
    points = base[:, None, :] + offsets[None, :, None] * aim[:, None, :]
    values = prediction.sample(points.reshape(-1, 3)[:, ::-1]).reshape(len(base), len(offsets))
    profile = values > 0
    found = nearest_run_centres(profile)
    field = np.full(valid.shape, np.nan, dtype=np.float32)
    field[rows, columns] = found
    median = nan_median5(field)
    rejected = np.isfinite(field) & np.isfinite(median) & (np.abs(field - median) > JUMP)
    kept = np.where(rejected, np.nan, field)
    filled = normconv(kept, max(SIGMA, 1.0))
    whole = np.where(np.isfinite(kept), kept, filled)
    whole = np.clip(normconv(whole, SIGMA), -MAX_OFF, MAX_OFF)
    apply = valid & np.isfinite(whole) & np.isfinite(filled)
    out = xyz.copy()
    out[:, apply] += whole[apply][None] * normal[:, apply]
    out[:, ~valid] = -1.0
    moved = np.abs(whole[apply])
    report = {
        "nodes": int(valid.sum()),
        "with_sheet": round(float(profile.any(axis=1).mean()), 4) if len(profile) else 0.0,
        "snapped": round(float(apply.sum() / max(valid.sum(), 1)), 4),
        "rejected": round(float(rejected.sum() / max(np.isfinite(field).sum(), 1)), 4),
        "median_move_vox": round(float(np.median(moved)), 2) if moved.size else None,
        "p90_move_vox": round(float(np.quantile(moved, 0.9)), 2) if moved.size else None,
        "already_on": round(float((np.abs(found[np.isfinite(found)]) <= 2).mean()), 4)
        if np.isfinite(found).any()
        else 0.0,
    }
    return out.astype(np.float32), report


# ---- building and rendering -----------------------------------------------------------------------------------


def write_tifxyz(folder, xyz, meta):
    folder.mkdir(parents=True, exist_ok=True)
    for a, plane in zip("xyz", xyz):
        tifffile.imwrite(folder / f"{a}.tif", plane.astype(np.float32))
    good = xyz[:, xyz[0] > 0]
    meta = meta | {"bbox": [good.min(axis=1).tolist(), good.max(axis=1).tolist()]} if good.size else meta
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def side_of(sheet, p):
    """The production read's written face of a piece, as the order a stack along the inward normal is read in."""
    path = sheet.pieces.parent / "faces" / "level-0" / p["unit"] / f"piece-{p['piece']:04d}-taught-side.json"
    try:
        side = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    turn = side.get("turn", p.get("turn", 1))
    return "inward" if (turn == 1) == (side.get("written") == "near") else "outward"


def built(spot, sheet, placed):
    """Each kept piece narrowed and rebuilt, written as tifxyz folders; returns the pieces' records."""
    work = work_of(spot["spot_id"])
    axis = axis_for(sheet)
    prediction = Prediction(sheet.scroll, sheet.scan)
    by_key = {(p["unit"], int(p["piece"])): p for p in placed}
    records = []
    for entry in spot["pieces"]:
        if entry["share"] < LEAST_SHARE:
            continue
        p = by_key[(entry["unit"], entry["piece"])]
        crop = narrowed(spot, p)
        if crop is None:
            continue
        r0, r1, c0, c1 = crop
        source = piece_folder(sheet, p)
        xyz, meta = sampler.load_surface(source)
        xyz = np.ascontiguousarray(xyz[:, r0:r1, c0:c1])
        if (xyz[0] > 0).sum() < 16:
            continue
        name = f"{p['unit']}-piece-{p['piece']:04d}"
        middle = np.median(xyz[:, xyz[0] > 0], axis=1)
        inward = sampler.grid_normals(xyz, centre_at(axis, middle[2], middle), meta["stride"] * meta["voxel_um"])
        moved, report = snapped(xyz, inward, prediction)
        crop_meta = meta | {"crop": [r0, r1, c0, c1], "source": str(source)}
        write_tifxyz(work / "surfaces" / "original" / name, xyz, crop_meta)
        write_tifxyz(work / "surfaces" / "m7" / name, moved, crop_meta | {"snap": report})
        records.append(
            {
                "name": name,
                "unit": p["unit"],
                "piece": int(p["piece"]),
                "crop": [r0, r1, c0, c1],
                "share": entry["share"],
                "winding": entry["winding"],
                "reading_side": side_of(sheet, p),
                "snap": report,
                "layout": {
                    k: p[k] for k in ("x_mm", "y_mm", "flip_rows", "flip_cols", "um", "stride", "shape", "area")
                },
            }
        )
        print(spot["spot_id"], name, "crop", crop, "snap", report, flush=True)
    (work / "pieces.json").write_text(json.dumps({"axis": axis.tolist(), "pieces": records}, indent=1))
    return records


def sheet_places(lay, rows, columns):
    """Sheet mm (x, y) of grid rows and columns of a piece, which may lie past its own grid."""
    step = lay["stride"] * lay["um"]
    count_r, count_c = lay["shape"]
    r = count_r - 1 - rows if lay["flip_rows"] else rows
    c = count_c - 1 - columns if lay["flip_cols"] else columns
    return lay["x_mm"] + c * step, lay["y_mm"] + r * step


def evenly_spaced(grid, stride, low=0.5, high=1.6):
    """Nodes whose every valid 4-neighbour lies between low and high node spacings away, and that have one at all.

    Carrying a grid on by straight extrapolation compounds its errors; a node that has run off the sheet shows as a step
    to its neighbours far from the grid's spacing, and is dropped before the next round extrapolates from it.
    """
    good = grid[0] > 0
    ok = good.copy()
    near = np.zeros(good.shape, dtype=bool)
    for axis in (1, 2):
        for step in (1, -1):
            shifted = np.roll(grid, step, axis=axis)
            other = np.roll(good, step, axis=axis - 1)
            edge = [slice(None)] * 2
            edge[axis - 1] = slice(0, 1) if step == 1 else slice(-1, None)
            other[tuple(edge)] = False
            distance = np.linalg.norm(grid - shifted, axis=0)
            both = good & other
            fine = (distance >= low * stride) & (distance <= high * stride)
            ok &= ~both | fine
            near |= both & fine
    return ok & near


def grown_surface(xyz, meta, crop, lay, frame, axis, prediction, inside=None, rounds=GROW_ROUNDS):
    """A snapped grid grown on m7 past its edges and into its holes, cut to the frame; with its new crop and counts.

    Each round carries the grid two cells on (sheet_images.extended) and keeps a new node only where predicted sheet
    lies within GROW_REACH voxels along its normal, moved to that sheet's centre. The whole grid is then snapped once
    more by the winners' rule, which smooths the new nodes with the old, and cut to the circle's frame. With `inside`
    (a function of the new nodes' x, y, z, n x 3, returning which may stay) the frame is that rule instead.
    """
    from sampling.sheet_images import MARGIN_CELLS, extended

    grid, pad, added = xyz.copy(), 0, []
    node_um = meta["stride"] * meta["voxel_um"]
    for _ in range(rounds):
        before = np.pad(grid[0] > 0, MARGIN_CELLS)
        wider = extended(grid)
        new = (wider[0] > 0) & ~before
        pad += MARGIN_CELLS
        r0, c0 = crop[0] - pad, crop[2] - pad
        rows, columns = np.nonzero(new)
        if inside is None:
            x, y = sheet_places(lay, rows + r0, columns + c0)
            outside = (x < frame[0]) | (x > frame[2]) | (y < frame[1]) | (y > frame[3])
        else:
            outside = ~inside(wider[:, rows, columns].T)
        wider[:, rows[outside], columns[outside]] = -1.0
        new &= wider[0] > 0
        if not new.any():
            grid = wider
            added.append(0)
            break
        middle = np.median(wider[:, wider[0] > 0], axis=1)
        normal = sampler.grid_normals(wider, centre_at(axis, middle[2], middle), node_um)
        rows, columns = np.nonzero(new)
        offsets = np.arange(-GROW_REACH, GROW_REACH + 1, dtype=np.float32)
        base, aim = wider[:, rows, columns].T, normal[:, rows, columns].T
        points = base[:, None, :] + offsets[None, :, None] * aim[:, None, :]
        values = prediction.sample(points.reshape(-1, 3)[:, ::-1]).reshape(len(base), len(offsets))
        found = nearest_run_centres(values > 0)
        keep = np.isfinite(found)
        wider[:, rows[keep], columns[keep]] += found[keep][None] * normal[:, rows[keep], columns[keep]]
        wider[:, rows[~keep], columns[~keep]] = -1.0
        for _ in range(2):
            wider[:, new & ~evenly_spaced(wider, meta["stride"])] = -1.0
        added.append(int((new & (wider[0] > 0)).sum()))
        grid = wider
    middle = np.median(grid[:, grid[0] > 0], axis=1)
    normal = sampler.grid_normals(grid, centre_at(axis, middle[2], middle), node_um)
    grid, report = snapped(grid, normal, prediction)
    good = grid[0] > 0
    rows, columns = np.nonzero(good)
    a0, a1, b0, b1 = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    grid = np.ascontiguousarray(grid[:, a0:a1, b0:b1])
    new_crop = [int(crop[0] - pad + a0), int(crop[0] - pad + a1), int(crop[2] - pad + b0), int(crop[2] - pad + b1)]
    report |= {"added_by_round": added, "nodes_before": int((xyz[0] > 0).sum())}
    return grid, new_crop, report


def grow_all(spot, records):
    """The grown surface of every built piece, kept beside the others; the records gain its crop."""
    work = work_of(spot["spot_id"])
    data = json.loads((work / "pieces.json").read_text())
    axis = np.array(data["axis"])
    prediction = Prediction(spot["scroll"], spot["scan"])
    outline = np.asarray(spot["outline_mm"])
    low, high = outline.min(axis=0) - MARGIN_MM, outline.max(axis=0) + MARGIN_MM
    frame = (low[0], low[1], high[0], high[1])
    for r in records:
        xyz, meta = sampler.load_surface(work / "surfaces" / "m7" / r["name"])
        grid, crop, report = grown_surface(xyz, meta, r["crop"], r["layout"], frame, axis, prediction)
        write_tifxyz(work / "surfaces" / "grown" / r["name"], grid, meta | {"crop": crop, "grow": report})
        r.setdefault("crops", {})["grown"] = crop
        r["grow"] = {k: report[k] for k in ("nodes", "nodes_before", "added_by_round", "median_move_vox")}
        print(
            spot["spot_id"], r["name"], "grown from", report["nodes_before"], "to", report["nodes"], "nodes", flush=True
        )
    data["pieces"] = records
    (work / "pieces.json").write_text(json.dumps(data, indent=1))
    return records


def readable(surface, folder, axis):
    """A built surface as the renderer reads it, with its margin and its normals turned away from the axis."""
    xyz, meta = sampler.load_surface(folder)
    cells = sampler.MESH_SMOOTHING_CELLS if surface == "original" else 0
    margined, inward = surface_of(xyz, meta, axis, cells=cells)
    return margined, -inward, meta


def ct_store(scroll, scan, spot_id, surfaces):
    """The CT bricks the renders read, fetched into the spot's own store."""
    folder = work_of(spot_id) / "ct"
    store, url = sampler.scan_store(scroll, scan, folder, 0)
    ranges = np.empty((*store.grid, 2), dtype=np.int32)
    ranges[..., 0].fill(store.brick[0])
    ranges[..., 1].fill(-1)
    keep = sampler.LAYERS
    sampler.LAYERS = LAYERS
    try:
        for margined, outward, meta in surfaces:
            mark_ranges(ranges, store, margined, outward, meta["stride"])
    finally:
        sampler.LAYERS = keep
    chosen = (ranges[..., 1] >= 0) & (store.offsets == UNKNOWN)
    indices = np.argwhere(chosen)
    first, last = ranges[..., 0][chosen], ranges[..., 1][chosen]
    plane = store.brick[1] * store.brick[2]
    begins = {tuple(i): int(f) * plane for i, f in zip(indices.tolist(), first)}

    def deliver(index, body):
        if body is None or len(body) == store.size:
            store.put(index, body)
        else:
            whole = bytearray(store.size)
            whole[begins[index] : begins[index] + len(body)] = body
            store.put(index, bytes(whole))

    items = [
        (tuple(i), "/".join(map(str, i)), (int(f) * plane, (int(t) + 1) * plane - 1))
        for i, f, t in zip(indices.tolist(), first, last)
    ]
    failures = sampler.fetched_all(url, items, deliver)
    store.save()
    if failures:
        raise SystemExit(f"{len(failures)} CT bricks failed")
    print(scroll, spot_id, "CT bricks fetched", len(items), flush=True)
    return BrickStore(folder)


def render_all(spot, sheet, records):
    """Every built surface of a spot rendered LAYERS deep, kept as uint8 .npy stacks (layer, row, column)."""
    work = work_of(spot["spot_id"])
    axis = np.array(json.loads((work / "pieces.json").read_text())["axis"])
    surfaces = {
        (surface, r["name"]): readable(surface, work / "surfaces" / surface / r["name"], axis)
        for surface in SURFACES
        for r in records
        if (work / "surfaces" / surface / r["name"]).exists()
        and not (work / "stacks" / surface / f"{r['name']}.npy").exists()
    }
    if not surfaces:
        return
    store = ct_store(sheet.scroll, sheet.scan, spot["spot_id"], surfaces.values())
    for (surface, name), (margined, outward, meta) in surfaces.items():
        out = work / "stacks" / surface / f"{name}.npy"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        stack = sampler.flatten(margined, meta["stride"], outward, store, layers=LAYERS).cube
        part = out.with_suffix(".part.npy")
        np.save(part, stack)
        part.rename(out)
        print(spot["spot_id"], surface, name, "rendered", stack.shape, flush=True)


def clear_ct(spot_id):
    shutil.rmtree(work_of(spot_id) / "ct", ignore_errors=True)
