"""How much ink signal a surface keeps."""

import json
import shutil
import sys

import numpy as np
from scipy import ndimage as ni
from scipy.spatial import cKDTree

from calibration.ink.ink_yield import rank_auc
from calibration.ink.stock_ink_scoring import window
from calibration.unwrap.sheet_controls import load_mesh, meshes
from ink.datasets.labelled_cases import load_truth
from ink.model.stock_model import SURE, InkModel
from ink.training.model_training import PAST_FACE, PATCH, face_layer_of
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points, surface_of
from sampling.sheet_images import MARGIN_CELLS, extended
from storage.brick_store import BrickStore
from storage.data_roots import (
    LABELLED_CASES,
    SURFACE_RECORDS,
    TRAINED_WEIGHTS,
    calibration_output,
)

SIDE = 1536
STEP = 4
TOUCH_UM = 60.0
READ_STEP = 120
SEED = "hybrid_3d2d-seed42/step-075000.pth"
LEFT_OUT = {
    "PHerc0500P2": "F-without-0500P2.pth",
    "PHerc0841": "F-without-0841.pth",
    "PHerc0009B": "F-without-0009B.pth",
}
PAD = 192
MOST_PIECES = 3
MOST_POINTS = 2000
SIGMAS = (0, 1.5, 2.0, 2.5)


def published_points(case):
    """Scored points of a case."""
    ink, known = load_truth(case)
    top, bottom, left, right = window(ink, known, SIDE)
    y0, _, x0, _ = case["box"]
    box = [y0 + top, y0 + bottom, x0 + left, x0 + right]
    folder = next(
        path
        for name, path in meshes(case["sample"], case["scan"].split("volume-")[1])
        if name.startswith(case["segment"])
    )
    meta = json.loads((folder / "meta.json").read_text())
    mesh = load_mesh(folder)
    rows, columns = np.meshgrid(np.arange(box[0], box[1], STEP), np.arange(box[2], box[3], STEP), indexing="ij")
    grid = [rows * meta["scale"][1], columns * meta["scale"][0]]
    good = ni.map_coordinates((mesh[0] > 0).astype(np.float32), grid, order=1, cval=0) > 0.999
    xyz = np.stack([ni.map_coordinates(p, grid, order=1, cval=-1) for p in mesh], axis=-1)
    label, seen = ink[rows - y0, columns - x0], known[rows - y0, columns - x0]
    use = good & seen
    return box, rows[use] - box[0], columns[use] - box[2], xyz[use], label[use]


def own_stacks(folder, sample, volume, axis, name, sigmas, reach=0, keep=False):
    """A piece's layers for each smoothing of its mesh, its bricks dropped afterwards unless kept for the next piece."""
    bricks = bricks_of(name)
    store, url = sampler.scan_store(sample, volume, bricks, 0)
    xyz, meta = sampler.load_surface(folder)
    sampler.fetch_near(store, url, [extended(sampler.readable_mesh(xyz, meta["stride"]))])
    for sigma in sigmas:
        margined, normal = surface_of(xyz, meta, axis, sigma, reach)
        yield sigma, sampler.flatten(margined, meta["stride"], normal, store).cube
    if not keep:
        shutil.rmtree(bricks)


def bricks_of(name):
    return calibration_output("surface_ink") / "bricks" / name


def own_pixels(folder, points, um, cells=sampler.MESH_SMOOTHING_CELLS, reach=sampler.BRIDGE_CELLS):
    """For scored points, the nearest pixel of a piece's stack and whether it touches the piece: positions on the mesh the stack is sampled on, only where the piece's own mesh holds sheet, not over bridged holes."""
    xyz, meta = sampler.load_surface(folder)
    split = max(1, round(meta["stride"] / 4))
    valid = ni.binary_erosion(sampler.columns((xyz[0] > 0).astype(np.float32), split) > 0.999)
    xyz = sampler.readable_mesh(xyz, meta["stride"], cells, reach)
    fine = np.stack([sampler.columns(p, split) for p in xyz])
    rows, columns = np.nonzero(valid)
    tree = cKDTree(fine[:, valid].T)
    distance, index = tree.query(points, workers=2)
    step = meta["stride"] / split
    margin = MARGIN_CELLS * meta["stride"]
    return (
        (rows[index] * step + margin).astype(np.int64),
        (columns[index] * step + margin).astype(np.int64),
        distance * um <= TOUCH_UM,
    )


def read_box(model, cube, box, past=PAST_FACE, start=None):
    """A model's read of a box of a stack, with room around it."""
    top, bottom, left, right = box
    y0, x0, y1, x1 = (
        max(top - PAD, 0),
        max(left - PAD, 0),
        min(bottom + PAD, cube.shape[1]),
        min(right + PAD, cube.shape[2]),
    )
    first = (face_layer_of(cube) if start is None else start) + past
    first = min(max(first, 0), cube.shape[0] - model.depth)
    return model.read(
        np.ascontiguousarray(cube[first : first + model.depth, y0:y1, x0:x1]), patch=PATCH, step=READ_STEP
    )[top - y0 : bottom - y0, left - x0 : right - x0]


def scored(values, ink):
    return float(rank_auc(values[ink], values[~ink]))


def run(names, control, sigmas=SIGMAS):
    """Published against our own pieces, per labelled case of a control's band, at each smoothing of the mesh."""
    base = calibration_output("surface_ink") / control
    record = record_of(base / "pieces")
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    model = InkModel(names=(SEED,), weights=[TRAINED_WEIGHTS / LEFT_OUT[sample]])
    axis = axis_points(sample, volume)
    folders = {r["piece"]: base / "pieces" / f"piece-{r['piece']:04d}" for r in record["pieces"] if r.get("written")}
    trees = {}
    for n, folder in folders.items():
        xyz, _ = sampler.load_surface(folder)
        trees[n] = cKDTree(xyz[:, xyz[0] > 0].T)
    rows = []
    for name in names:
        case = json.loads((LABELLED_CASES / name / "case.json").read_text())
        box, y, x, points, ink = published_points(case)
        store = BrickStore(LABELLED_CASES / name / "layers")
        cube = store.read((0, store.shape[0]), tuple(box[:2]), tuple(box[2:]))
        published = read_box(model, cube, (0, cube.shape[1], 0, cube.shape[2]))[y, x]
        near = sorted(
            ((int((tree.query(points[::5], workers=2)[0] * um < 100).sum()), n) for n, tree in trees.items()),
            reverse=True,
        )[:MOST_PIECES]
        for count, n in near:
            r, c, touching = own_pixels(folders[n], points, um)
            if count == 0 or touching.sum() < MOST_POINTS:
                continue
            row = {
                "case": name,
                "piece": n,
                "points": int(touching.sum()),
                "ink_share": round(float(ink[touching].mean()), 3),
                "published": round(scored(published[touching], ink[touching]), 4),
            }
            for sigma, stack in own_stacks(folders[n], sample, volume, axis, f"{control}-{n:04d}", sigmas):
                rr, cc = np.minimum(r[touching], stack.shape[1] - 1), np.minimum(c[touching], stack.shape[2] - 1)
                area = (int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1)
                y0, x0 = max(area[0] - PAD, 0), max(area[2] - PAD, 0)
                whole = stack[:, y0 : area[1] + PAD, x0 : area[3] + PAD]
                inside = (area[0] - y0, area[1] - y0, area[2] - x0, area[3] - x0)
                faces = [read_box(model, np.ascontiguousarray(f), inside) for f in (whole, whole[::-1])]
                read = max(faces, key=lambda m: float((m >= SURE).mean()))
                row[f"mesh {sigma}"] = round(scored(read[rr - area[0], cc - area[2]], ink[touching]), 4)
            rows.append(row)
            print(json.dumps(row), flush=True)
    SURFACE_RECORDS.mkdir(exist_ok=True)
    (SURFACE_RECORDS / f"ink-retention-{control}.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    run(sys.argv[2:], sys.argv[1])
