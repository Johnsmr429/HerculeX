"""Level-route read blocks of one source piece stitched back around a labelled case, sampled once, labels carried with raw and with readable-mesh validity."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ni
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from calibration.ink.surface_ink_retention import PAD, TOUCH_UM
from ink.datasets.labelled_cases import load_truth
from ink.datasets.own_mesh_labels import (
    ROUTES,
    carried,
    case_points,
    census_path,
    keep_window,
    layer_agreement,
    published_pixels,
    published_stack,
)
from ink.model.input_stack import face_layer_of
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points, surface_of
from sampling.sheet_images import MARGIN_CELLS, extended
from storage.data_roots import LABELLED_CASES

TOUCH_CASE_UM = 60.0
CROP_MARGIN_CELLS = 6
SMALL, MEDIUM = 3, 11
SAMPLED_PIXELS = 40000
LEAST_GROUP_CELLS = 10
GROUP_GAP_CELLS = 24


def stitched_source(base, source):
    """Every written block of one source piece placed back on the source's grid."""
    blocks = []
    for folder in sorted(base.glob("piece-*")):
        meta = json.loads((folder / "meta.json").read_text())
        if meta.get("source", meta["piece"]) == source:
            blocks.append((folder, meta, sampler.load_surface(folder)[0], np.load(folder / "quality.npy")))
    height = max(m.get("block", [0, 0])[0] + xyz.shape[1] for _, m, xyz, _ in blocks)
    width = max(m.get("block", [0, 0])[1] + xyz.shape[2] for _, m, xyz, _ in blocks)
    grid = np.full((3, height, width), -1.0, dtype=np.float32)
    quality = np.zeros((height, width), dtype=np.uint8)
    for _, meta, xyz, q in blocks:
        r0, c0 = meta.get("block", [0, 0])
        h, w = xyz.shape[1:]
        here = xyz[0] > 0
        grid[:, r0 : r0 + h, c0 : c0 + w][:, here] = xyz[:, here]
        quality[r0 : r0 + h, c0 : c0 + w] = np.maximum(quality[r0 : r0 + h, c0 : c0 + w], q)
    return grid, quality, blocks[0][1]


def passages(base, case, census):
    """Every source piece the case's census names, stitched and cut to each group of its cells on the case's sheet."""
    points = case_points(case)[0]
    sources = {}
    for pair in census:
        if pair["case"] == case["case_id"]:
            meta = json.loads((base / f"piece-{pair['piece']:04d}" / "meta.json").read_text())
            key = meta.get("source", meta["piece"])
            sources[key] = sources.get(key, 0) + pair["points"]
    tree = cKDTree(points[::7])
    for source in sorted(sources, key=lambda k: -sources[k]):
        grid, quality, meta = stitched_source(base, source)
        good = grid[0] > 0
        cells = np.argwhere(good)
        distance = tree.query(grid[:, good].T, workers=2)[0]
        touch = np.zeros(good.shape, dtype=bool)
        touch[tuple(cells[distance * case["voxel_um"] <= TOUCH_CASE_UM].T)] = True
        blobs, count = ni.label(ni.binary_closing(touch, iterations=3))
        found = ni.find_objects(blobs)
        sizes = np.bincount(blobs.ravel(), minlength=count + 1)[1:]
        kept = [k for k in range(count) if sizes[k] >= LEAST_GROUP_CELLS]
        close = np.array(
            [
                [
                    all(
                        x.start - GROUP_GAP_CELLS < y.stop and y.start - GROUP_GAP_CELLS < x.stop
                        for x, y in zip(found[i], found[j])
                    )
                    for j in kept
                ]
                for i in kept
            ],
            dtype=bool,
        ).reshape(len(kept), len(kept))
        _, group = connected_components(close, directed=False) if kept else (0, np.zeros(0, dtype=np.int64))
        for g in sorted(set(group.tolist()), key=lambda g: -sum(sizes[k] for k, h in zip(kept, group) if h == g)):
            members = [k + 1 for k, h in zip(kept, group) if h == g]
            near = np.argwhere(np.isin(blobs, members) & touch)
            (r0, c0), (r1, c1) = near.min(0) - CROP_MARGIN_CELLS, near.max(0) + CROP_MARGIN_CELLS + 1
            r0, c0 = max(r0, 0), max(c0, 0)
            yield grid[:, r0:r1, c0:c1], quality[r0:r1, c0:c1], meta, source, sources, (int(r0), int(c0))


def write_piece(folder, grid, quality, meta, extra):
    """A stitched passage as a tifxyz piece."""
    folder.mkdir(parents=True, exist_ok=True)
    for axis, name in enumerate("xyz"):
        tifffile.imwrite(folder / f"{name}.tif", grid[axis])
    np.save(folder / "quality.npy", quality)
    good = grid[0] > 0
    keys = ("sample", "scan", "voxel_um", "level_of_nodes", "shape", "format", "type", "scale", "stride")
    row = {k: meta[k] for k in keys} | extra
    row |= {
        "uuid": folder.name,
        "bbox": [grid[:, good].min(1).tolist(), grid[:, good].max(1).tolist()],
        "area_mm2": float(good.sum()) * (meta["stride"] * meta["voxel_um"] / 1000) ** 2,
    }
    (folder / "meta.json").write_text(json.dumps(row))


def readable_pixels(folder, points, um):
    """As own_pixels, but valid wherever the readable mesh is, the mesh the stack is sampled on."""
    xyz, meta = sampler.load_surface(folder)
    split = max(1, round(meta["stride"] / 4))
    readable = sampler.readable_mesh(xyz, meta["stride"])
    valid = ni.binary_erosion(sampler.columns((readable[0] > 0).astype(np.float32), split) > 0.999)
    fine = np.stack([sampler.columns(p, split) for p in readable])
    rows, columns = np.nonzero(valid)
    distance, index = cKDTree(fine[:, valid].T).query(points, workers=2)
    step, margin = meta["stride"] / split, MARGIN_CELLS * meta["stride"]
    return (
        (rows[index] * step + margin).astype(np.int64),
        (columns[index] * step + margin).astype(np.int64),
        distance * um <= TOUCH_UM,
    )


def cell_classes(folder):
    """Each raw cell: 0 own mesh, 1 a hole of at most SMALL cells, 2 at most MEDIUM, 3 a larger hole, 4 outside the mesh."""
    known = np.load(folder / "quality.npy") > 0
    filled = ni.binary_fill_holes(known)
    holes, _ = ni.label(filled & ~known)
    size = np.bincount(holes.ravel())[holes]
    inner = np.where(size <= SMALL, 1, np.where(size <= MEDIUM, 2, 3))
    return np.where(known, 0, np.where(~filled, 4, inner)).astype(np.uint8)


def carried_readable(case, folder, stack, um, rng):
    """carried() with the readable mesh's validity, each kept pixel with the class of its raw cell."""
    points, _, rows, columns = case_points(case)
    r, c, touching = readable_pixels(folder, points, um)
    r, c = np.minimum(r[touching], stack.shape[1] - 1), np.minimum(c[touching], stack.shape[2] - 1)
    top, left = max(int(r.min()) - PAD, 0), max(int(c.min()) - PAD, 0)
    cut = np.ascontiguousarray(stack[:, top : int(r.max()) + 1 + PAD, left : int(c.max()) + 1 + PAD])
    at_row, at_column, seen = published_pixels(r, c, rows[touching], columns[touching], top, left, *cut.shape[1:])
    ink, known = load_truth(case)
    at_row = np.clip(at_row.round().astype(np.int64), 0, ink.shape[0] - 1)
    at_column = np.clip(at_column.round().astype(np.int64), 0, ink.shape[1] - 1)
    known_here = seen & known[at_row, at_column] & (cut[cut.shape[0] // 2] > 0)
    ys, xs = np.nonzero(known_here)
    some = rng.permutation(len(ys))[:SAMPLED_PIXELS]
    published = np.asarray(published_stack(case))[:, at_row[ys[some], xs[some]], at_column[ys[some], xs[some]]]
    row = layer_agreement(cut[:, ys[some], xs[some]].astype(np.float32), published.astype(np.float32), rng)
    row |= {"pixels": int(known_here.sum()), "shape": list(cut.shape)}
    row["face"] = int(face_layer_of(stack[::-1] if row["reversed"] else stack))
    stride = json.loads((folder / "meta.json").read_text())["stride"]
    classes = cell_classes(folder)
    cr, cc = (ys + top - MARGIN_CELLS * stride) // stride, (xs + left - MARGIN_CELLS * stride) // stride
    inside = (cr >= 0) & (cc >= 0) & (cr < classes.shape[0]) & (cc < classes.shape[1])
    labels = {
        "ink": ink[at_row, at_column] & known_here,
        "known": known_here,
        "published": np.stack([ys, xs, at_row[ys, xs], at_column[ys, xs]]).astype(np.int32),
        "pixel_class": np.where(
            inside, classes[np.clip(cr, 0, classes.shape[0] - 1), np.clip(cc, 0, classes.shape[1] - 1)], 4
        ).astype(np.uint8),
    }
    return cut[::-1] if row["reversed"] else cut, labels, row


def build(band, out, *case_ids):
    """Stitched windows per labelled case of a level band, one per group of source cells on its sheet, under out/{level_route,level_route_readable}/SCROLL."""
    base, out = ROUTES["level"] / band / "pieces", Path(out)
    record = record_of(base)
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    census = json.loads(census_path("level", band).read_text())
    axis = axis_points(sample, volume)
    rng = np.random.default_rng(3)
    for case_id in case_ids or sorted({p["case"] for p in census}):
        case = json.loads((LABELLED_CASES / case_id / "case.json").read_text())
        for n, (grid, quality, meta, source, sources, corner) in enumerate(passages(base, case, census)):
            name = f"{case_id}-stitched" + (f"-{n}" if n else "")
            if (out / "level_route" / sample / f"{name}.json").exists():
                continue
            folder = out / "stitched" / name
            extra = {"source": int(source), "corner": corner, "sources": {str(k): v for k, v in sources.items()}}
            write_piece(folder, grid, quality, meta, extra)
            bricks = out / "bricks" / name
            store, url = sampler.scan_store(sample, volume, bricks, 0)
            xyz, pmeta = sampler.load_surface(folder)
            fetched = sampler.fetch_near(store, url, [extended(sampler.readable_mesh(xyz, pmeta["stride"]))])
            margined, normal = surface_of(xyz, pmeta, axis)
            stack = sampler.flatten(margined, pmeta["stride"], normal, store).cube
            shutil.rmtree(bricks)
            np.save(folder / "stack.npy", stack)
            cut, labels, found = carried(case, folder, stack, um, rng)
            row = {"case": case_id, "piece": name, "source": int(source), "points": int(labels["known"].sum())}
            keep_window(out / "level_route" / sample, name, cut, labels, row | {"fetched": fetched} | found)
            cut, labels, found = carried_readable(case, folder, stack, um, rng)
            found["classes"] = {str(k): int(v) for k, v in zip(*np.unique(labels["pixel_class"], return_counts=True))}
            keep_window(out / "level_route_readable" / sample, name, cut, labels, row | found)
            print(name, json.dumps(found), flush=True)


if __name__ == "__main__":
    build(*sys.argv[1:])
