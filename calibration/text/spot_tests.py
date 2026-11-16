"""Whether strokes seen at one spot of a sheet hold up."""

import json
import shutil
import sys

import numpy as np
from scipy import ndimage as ni

from calibration.ink.invention_guard import scrambled
from ink.model.stock_model import NO_INK, SURE, InkModel
from ink.reading.piece_reading import WEIGHTS
from ink.training.model_training import PAST_FACE, PATCH, face_layer_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points, surface_of
from sampling.sheet_images import MARGIN_CELLS, extended
from storage.data_roots import (
    INK_RECORDS,
    PUBLISHED_PAIRS,
    ROUTE_OUTPUTS,
    TRAINED_WEIGHTS,
    calibration_output,
    named_folder,
    stage_folder,
)

SPOT_STACKS = calibration_output("spot_stacks")
PAD = 192
MOVED = (37, 53)
SCAN_SMOOTHING_MM = 0.3
SEEDS = ("hybrid_3d2d-seed42/step-075000.pth", "hybrid_3d2d-seed43/step-075000.pth")
STEP = 120
LEFT_OUT = [TRAINED_WEIGHTS / name for name in ("F-without-0841.pth", "without-0841-seed43.pth")]


def piece_stack(sample, volume, unit, piece, turn):
    """A piece's layers, fetched once and kept."""
    out = SPOT_STACKS / f"{sample}-{unit}-{piece:04d}.npy"
    if out.exists():
        return np.load(out, mmap_mode="r")
    SPOT_STACKS.mkdir(parents=True, exist_ok=True)
    folder = ROUTE_OUTPUTS / sample / volume / "pieces" / unit / f"piece-{piece:04d}"
    work = SPOT_STACKS / f"work-{unit}-{piece:04d}"
    shutil.rmtree(work, ignore_errors=True)
    store, url = sampler.scan_store(sample, volume, work / "bricks", 0)
    xyz, meta = sampler.load_surface(folder)
    sampler.fetch_near(store, url, [extended(sampler.readable_mesh(xyz, meta["stride"]))])
    margined, normal = surface_of(xyz, meta, axis_points(sample, volume))
    np.save(out, sampler.flatten(margined, meta["stride"], normal * turn, store).cube)
    shutil.rmtree(work)
    return np.load(out, mmap_mode="r")


def box_on_piece(p, x_mm, y_mm, width_mm, height_mm, shape):
    """A box of the sheet as (top, bottom, left, right) on the piece's own image."""
    pixel = p["um"]
    margin = MARGIN_CELLS * p["stride"]
    left, top = round((x_mm - p["x_mm"]) / pixel) + margin, round((y_mm - p["y_mm"]) / pixel) + margin
    right, bottom = left + round(width_mm / pixel), top + round(height_mm / pixel)
    if p["flip_rows"]:
        top, bottom = shape[0] - bottom, shape[0] - top
    if p["flip_cols"]:
        left, right = shape[1] - right, shape[1] - left
    return max(top, 0), min(bottom, shape[0]), max(left, 0), min(right, shape[1])


def read_at(models, cube, box, past, shift=(0, 0)):
    """Each model's read of a box, the patch grid moved by shift, from a depth window past the face."""
    top, bottom, left, right = box
    y0, x0 = max(top - PAD - shift[0], 0), max(left - PAD - shift[1], 0)
    y1, x1 = min(bottom + PAD, cube.shape[1]), min(right + PAD, cube.shape[2])
    start = min(face_layer_of(cube) + past, cube.shape[0] - models[0].depth)
    stack = np.ascontiguousarray(cube[start : start + models[0].depth, y0:y1, x0:x1])
    return [m.read(stack, patch=PATCH, step=STEP)[top - y0 : bottom - y0, left - x0 : right - x0] for m in models]


def agreement(a, b, seen):
    return float(np.corrcoef(a[seen], b[seen])[0, 1])


def sure(read, seen):
    return float((read[seen] >= SURE).mean())


def tests(models, cube, box):
    """The numbers of a spot, and its maps."""
    top, bottom, left, right = box
    cube = np.asarray(cube)
    seen = cube[cube.shape[0] // 2, top:bottom, left:right] > 0
    one, two = read_at(models, cube, box, PAST_FACE)
    both = (one + two) / 2
    moved = np.mean(read_at(models, cube, box, PAST_FACE, MOVED), axis=0)
    shallow = np.mean(read_at(models, cube, box, 0), axis=0)
    deep = np.mean(read_at(models, cube, box, 2 * PAST_FACE), axis=0)
    turned = cube[::-1]
    other = np.mean(read_at(models, turned, box, PAST_FACE), axis=0)
    rng = np.random.default_rng(1)
    twin = scrambled(
        np.ascontiguousarray(cube[:, max(top - PAD, 0) : bottom + PAD, max(left - PAD, 0) : right + PAD]), rng
    )
    inner = (
        top - max(top - PAD, 0),
        top - max(top - PAD, 0) + bottom - top,
        left - max(left - PAD, 0),
        left - max(left - PAD, 0) + right - left,
    )
    structureless = np.mean(read_at(models, twin, inner, PAST_FACE), axis=0)
    brightest = ni.gaussian_filter(
        cube[:, top:bottom, left:right].max(axis=0).astype(np.float32), SCAN_SMOOTHING_MM / 0.009362
    )
    numbers = {
        "seeds": agreement(one, two, seen),
        "grid moved": agreement(both, moved, seen),
        "two layers up": agreement(both, shallow, seen),
        "two layers down": agreement(both, deep, seen),
        "sure": sure(both, seen),
        "sure on the other face": sure(other, seen),
        "sure on the twin": sure(structureless, seen),
        "other face against this": agreement(both, other, seen),
        "scan brightness": agreement(both, brightest, seen),
    }
    return numbers, {
        "seed one": one,
        "seed two": two,
        "both": both,
        "grid moved": moved,
        "two layers up": shallow,
        "two layers down": deep,
        "other face": other,
    }


def spot(models, stock, sample, volume, name, p, written, x_mm, y_mm, width_mm, height_mm):
    cube = piece_stack(sample, volume, p["unit"], p["piece"], p.get("turn", 1))
    box = box_on_piece(p, x_mm, y_mm, width_mm, height_mm, cube.shape[1:])
    if written == "far":
        cube = cube[::-1]
    numbers, maps = tests(models, cube, box)
    maps["stock"] = np.mean(read_at(stock, np.asarray(cube), box, PAST_FACE), axis=0)
    numbers["stock against taught"] = agreement(
        maps["both"], maps["stock"], np.asarray(cube)[cube.shape[0] // 2, box[0] : box[1], box[2] : box[3]] > 0
    )
    print(name, {k: round(v, 3) for k, v in numbers.items()}, flush=True)
    return numbers, maps


def shown(read):
    return (np.clip((read - NO_INK) / (1 - 2 * NO_INK), 0, 1) * 255).astype(np.uint8)


def run(sample, volume, spots_path):
    """Every spot of a list {name."""
    base = ROUTE_OUTPUTS / sample / volume
    placed = json.loads((stage_folder(base, "sheet") / "placed-taught.json").read_text())["pieces"]
    faces = stage_folder(base, "faces") / "level-0"
    models = [
        InkModel(names=(seed,), weights=[weights])
        for seed, weights in zip(SEEDS, (WEIGHTS, named_folder(WEIGHTS.parent, "reader-seed43.pth", "seed43.pth")))
    ]
    stock = [InkModel(names=(seed,)) for seed in SEEDS]
    left_out = [InkModel(names=(seed,), weights=[weights]) for seed, weights in zip(SEEDS, LEFT_OUT)]
    out = calibration_output("spot_tests")
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    cube = np.load(PUBLISHED_PAIRS / "PHerc0841" / "20260221022814-896-1920.npy")
    for name, box in {
        "PHerc0841 letters top row": (250, 780, 0, 1500),
        "PHerc0841 letters second row": (950, 1500, 0, 2048),
    }.items():
        rows[name], maps = tests(left_out, cube, box)
        print(name, {k: round(v, 3) for k, v in rows[name].items()}, flush=True)
        np.savez_compressed(out / f"{name.replace(' ', '-')}.npz", **{k: shown(v) for k, v in maps.items()})
    for name, (x, y, width, height) in json.loads(open(spots_path).read()).items():
        p = max(
            (
                p
                for p in placed
                if p["x_mm"] < x + width
                and p["x_mm"] + p["width_mm"] > x
                and p["y_mm"] < y + height
                and p["y_mm"] + p["height_mm"] > y
            ),
            key=lambda p: (
                (min(x + width, p["x_mm"] + p["width_mm"]) - max(x, p["x_mm"]))
                * (min(y + height, p["y_mm"] + p["height_mm"]) - max(y, p["y_mm"]))
            ),
        )
        written = json.loads((faces / p["unit"] / f"piece-{p['piece']:04d}-taught-side.json").read_text())["written"]
        rows[name], maps = spot(models, stock, sample, volume, name, p, written, x, y, width, height)
        rows[name] |= {"piece": f"{p['unit']} {p['piece']}", "written": written}
        np.savez_compressed(
            out / f"{name.replace(' ', '-')}.npz",
            **{k: shown(v) for k, v in maps.items()},
            flip_rows=p["flip_rows"],
            flip_cols=p["flip_cols"],
        )
    (INK_RECORDS / "spot-tests.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    run(*sys.argv[1:4])
