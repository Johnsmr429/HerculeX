"""Scrolls read on the Neural Engine by a model that never saw them."""

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from ink.model.neural_engine import EngineReader
from ink.model.stock_model import SURE
from ink.reading.piece_reading import window
from sampling import column_sampling as sampler
from sampling.fibre_faces import gap_state, recto_face
from sampling.piece_images import (
    axis_of_pieces,
    axis_points,
    fetch_bricks,
    surface_of,
)
from sampling.sheet_images import extended
from storage.brick_store import BrickStore
from storage.data_roots import (
    ROUTE_OUTPUTS,
    TRAINED_WEIGHTS,
    calibration_output,
    ct_bricks,
)

SURVEYED = calibration_output("scroll_survey")
REPLICATED, SECOND_MODEL, ROW_SAMPLE = SURVEYED / "replicate", SURVEYED / "second-model", SURVEYED / "row-sample"
WEIGHTS = TRAINED_WEIGHTS / "F-without-0841.pth"
SECOND_WEIGHTS = TRAINED_WEIGHTS / "F-without-0500P2.pth"
PIECES_AT_ONCE = 2
RULER_SIDE, RULER_WINDOWS = 1024, 2
RULER = "both directions read whole"
SMALLEST_RULER_PIXELS = 2**17
CELL = 8
LETTER_BAND_MM = (0.4, 3.0)
GAP_TILE = 512
LOOK_AREA_MM2, LOOK_DEPTH_MM = (100, 350), 2
LOOK_CANDIDATES, LOOK_BRICKS = 60, 2500
REPLICATE_SEED, REPLICATE_PIECES = 7, 16
ROW_SAMPLE_SEED, ROW_SAMPLE_MM = 8, 17.0
_store = None
_reader = None


def start_worker(store_folder, weights):
    global _store, _reader
    _store = BrickStore(store_folder)
    _reader = EngineReader(weights)


def cells_of(image):
    h, w = image.shape[0] // CELL * CELL, image.shape[1] // CELL * CELL
    return image[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL).mean(axis=(1, 3))


def letter_band(image, seen, cell_mm):
    weight = seen.astype(np.float32)
    values = np.where(seen, image, 0).astype(np.float32)
    fine, coarse = (
        ni.gaussian_filter(values, mm / cell_mm) / np.maximum(ni.gaussian_filter(weight, mm / cell_mm), 1e-3)
        for mm in LETTER_BAND_MM
    )
    return fine - coarse


def fullest(sheet):
    """Corners of the windows of a piece most covered by sheet, none overlapping."""
    h, w = sheet.shape
    side_y, side_x = min(RULER_SIDE, h), min(RULER_SIDE, w)
    mass = np.pad(sheet.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    spots = [
        (y, x)
        for y in range(0, h - side_y + 1, max(side_y // 2, 1))
        for x in range(0, w - side_x + 1, max(side_x // 2, 1))
    ]
    spots.sort(
        key=lambda s: (
            -(
                mass[s[0] + side_y, s[1] + side_x]
                - mass[s[0], s[1] + side_x]
                - mass[s[0] + side_y, s[1]]
                + mass[s[0], s[1]]
            )
        )
    )
    chosen = []
    for y, x in spots:
        if all(abs(y - a) >= side_y or abs(x - b) >= side_x for a, b in chosen):
            chosen.append((y, x))
        if len(chosen) == RULER_WINDOWS:
            break
    return chosen, side_y, side_x


def open_columns(face):
    """Columns of a stack whose near face looks onto a gap."""
    state = np.full(face.shape[1:], 2, dtype=np.uint8)
    for y in range(0, face.shape[1], GAP_TILE):
        for x in range(0, face.shape[2], GAP_TILE):
            block = np.ascontiguousarray(face[:, y : y + GAP_TILE, x : x + GAP_TILE])
            if block.shape[1] >= 48 and block.shape[2] >= 48 and block.any():
                state[y : y + GAP_TILE, x : x + GAP_TILE] = gap_state(block)
    return state == 0


def ruler_row(face, rear, read, voxel_mm, out):
    """Sure ink and letter-band energy forward and reversed on a piece's fullest windows, on open columns, both directions read whole."""
    sheet = face[face.shape[0] // 2] > 0
    row = {
        "ruler": RULER,
        "pixels": 0,
        "sheet_pixels": 0,
        "forward": 0.0,
        "reversed": 0.0,
        "band_forward": 0.0,
        "band_reversed": 0.0,
        "cells": 0,
        "faces_open": {name: [0, 0, 0] for name in ("both", "near", "far", "neither")},
    }
    if sheet.sum() < SMALLEST_RULER_PIXELS:
        return row
    spots, side_y, side_x = fullest(sheet)
    backward = _reader.read(rear)
    Image.fromarray(np.where(sheet, np.rint(backward * 254) + 1, 0).astype(np.uint8)).save(
        out.with_name(f"reversed-{out.name}.png"), compress_level=3
    )
    kept = {}
    for k, (y, x) in enumerate(spots):
        near, far = read[y : y + side_y, x : x + side_x], backward[y : y + side_y, x : x + side_x]
        part = face[:, y : y + side_y, x : x + side_x]
        seen = part[part.shape[0] // 2] > 0
        gap, behind = seen & open_columns(part), seen & open_columns(part[::-1])
        for name, where in (
            ("both", gap & behind),
            ("near", gap & ~behind),
            ("far", behind & ~gap),
            ("neither", seen & ~gap & ~behind),
        ):
            counts = row["faces_open"][name]
            counts[0] += int(where.sum())
            counts[1] += int((near[where] >= SURE).sum())
            counts[2] += int((far[where] >= SURE).sum())
        row["pixels"] += int(gap.sum())
        row["sheet_pixels"] += int(seen.sum())
        row["forward"] += float((near[gap] >= SURE).sum())
        row["reversed"] += float((far[gap] >= SURE).sum())
        cell_mm = voxel_mm * CELL
        whole, opened = cells_of(seen.astype(np.float32)) > 0.9, cells_of(gap.astype(np.float32)) > 0.9
        for name, image in (("band_forward", near), ("band_reversed", far)):
            row[name] += float((letter_band(cells_of(image), whole, cell_mm)[opened] ** 2).sum())
        row["cells"] += int(opened.sum())
        kept |= {
            f"forward{k}": cells_of(near).astype(np.float16),
            f"reversed{k}": cells_of(far).astype(np.float16),
            f"sheet{k}": cells_of(seen.astype(np.float32)).astype(np.float16),
            f"open{k}": cells_of(gap.astype(np.float32)).astype(np.float16),
        }
    np.savez_compressed(out.with_suffix(".npz"), **kept)
    return row


def read_piece(job):
    """One piece."""
    folder, out, axis = job
    start = time.time()
    try:
        xyz, meta = sampler.load_surface(folder)
        margined, normal = surface_of(xyz, meta, axis)
        cube = sampler.flatten(margined, meta["stride"], normal, _store).cube
        side = recto_face(cube)
        face, back = (cube[::-1], cube) if side["face"] == "far" else (cube, cube[::-1])
        front = window(face)
        read = _reader.read(front)
        sheet = face[face.shape[0] // 2] > 0
        Image.fromarray(np.where(sheet, np.rint(read * 254) + 1, 0).astype(np.uint8)).save(
            out.with_suffix(".png"), compress_level=3
        )
        row = {
            "piece": folder.name,
            "face": side["face"],
            "polarity": side["polarity"],
            "area_mm2": meta.get("area_mm2", 0),
            "sure": float((read[sheet] >= SURE).mean()) if sheet.any() else 0.0,
        }
        row |= ruler_row(face, window(back), read, meta["voxel_um"] / 1000, out)
        row["seconds"] = round(time.time() - start, 1)
        out.with_suffix(".json").write_text(json.dumps(row))
        return row
    except Exception as error:
        return {"piece": folder.name, "failed": repr(error)[:200]}


def pieces_of(sample, volume, band):
    """The folder of a band's pieces."""
    base = ROUTE_OUTPUTS / sample / volume / "pieces" / band
    return base if base.exists() else calibration_output("surface_ink") / f"{sample}-{volume}-{band}" / "pieces"


def axis_of(sample, volume, base):
    axis = axis_points(sample, volume)
    return axis_of_pieces([base]) if axis is None else axis


def jobs_of(sample, volume, band):
    """What is left to read of a band, largest pieces first."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    out = SURVEYED / sample
    out.mkdir(parents=True, exist_ok=True)
    folders = sorted(base.glob("piece-*"), key=lambda f: -json.loads((f / "meta.json").read_text()).get("area_mm2", 0))
    return [(f, out / f.name, axis) for f in folders if not (out / f"{f.name}.json").exists()]


def body_pieces(base, axis):
    """Pieces of a usable size at least 2 mm inside the outer surface of their band, with the bricks each touches."""
    angles = np.linspace(-np.pi, np.pi, 73)
    found, points = [], []
    for folder in sorted(base.glob("piece-*")):
        xyz, meta = sampler.load_surface(folder)
        good = xyz[0] > 0
        if good.sum() >= 30:
            found.append((folder, meta))
            points.append(xyz[:, good][:, ::3])

    def polar(p, um):
        cx, cy = np.interp(p[2], axis[:, 0], axis[:, 1]), np.interp(p[2], axis[:, 0], axis[:, 2])
        return np.digitize(np.arctan2(p[1] - cy, p[0] - cx), angles), np.hypot(p[0] - cx, p[1] - cy) * um

    um = found[0][1]["voxel_um"] / 1000
    every = np.concatenate(points, axis=1)
    turn, radius = polar(every, um)
    key = (every[2] // 300).astype(np.int64) * 100 + turn
    order = np.argsort(key)
    unique, first = np.unique(key[order], return_index=True)
    outer = dict(zip(unique.tolist(), np.maximum.reduceat(radius[order], first).tolist()))
    pool = []
    for (folder, meta), p in zip(found, points):
        if LOOK_AREA_MM2[0] <= meta.get("area_mm2", 0) <= LOOK_AREA_MM2[1]:
            t, r = polar(p, um)
            z = (p[2] // 300).astype(np.int64)
            if (
                float(np.median(np.array([outer[a * 100 + b] for a, b in zip(z.tolist(), t.tolist())]) - r))
                >= LOOK_DEPTH_MM
            ):
                pool.append((folder, meta))
    return pool


def bricks_of(folder, brick=128, reach=17):
    xyz, _ = sampler.load_surface(folder)
    points = extended(xyz)
    points = points[:, points[0] > 0].T[:, ::-1]
    touched = set()
    for corner in np.array(np.meshgrid(*[[-reach, reach]] * 3)).reshape(3, -1).T:
        touched.update(map(tuple, np.unique(((points + corner) // brick).astype(np.int64), axis=0)))
    return touched


def look_jobs(sample, volume, band):
    """Body pieces from all round the band, chosen so a few bricks serve as much sheet as they can, up to the look's budget."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    out = SURVEYED / sample
    out.mkdir(parents=True, exist_ok=True)
    pool = body_pieces(base, axis)
    rng = np.random.default_rng(3)
    rng.shuffle(pool)
    pool = [(f, m.get("area_mm2", 0), bricks_of(f)) for f, m in pool[:LOOK_CANDIDATES]]
    have, chosen = set(), []
    while pool and len(have) < LOOK_BRICKS:
        best = max(pool, key=lambda p: p[1] / max(len(p[2] - have), 1))
        pool.remove(best)
        have |= best[2]
        chosen.append(best[0])
    return [(f, out / f.name, axis) for f in chosen if not (out / f"{f.name}.json").exists()]


def replicate_jobs(sample, volume, band):
    """Body pieces drawn at random from the pool a first look chose from, none it read, so the sample owes nothing to the first."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    out = REPLICATED / sample
    out.mkdir(parents=True, exist_ok=True)
    pool = [folder for folder, _ in body_pieces(base, axis) if not (SURVEYED / sample / f"{folder.name}.json").exists()]
    np.random.default_rng(REPLICATE_SEED).shuffle(pool)
    return [(f, out / f.name, axis) for f in pool[:REPLICATE_PIECES] if not (out / f"{f.name}.json").exists()]


def second_model_jobs(sample, volume, band):
    """The replicate's pieces again, for a second left-out model."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    out = SECOND_MODEL / sample
    out.mkdir(parents=True, exist_ok=True)
    return [
        (base / p.stem, out / p.stem, axis)
        for p in sorted((REPLICATED / sample).glob("piece-*.json"))
        if not (out / p.name).exists()
    ]


def row_sample_jobs(sample, volume, band):
    """A third sample for the row test alone."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    out = ROW_SAMPLE / sample
    out.mkdir(parents=True, exist_ok=True)
    read = {p.stem for folder in (SURVEYED / sample, REPLICATED / sample) for p in folder.glob("piece-*.json")}
    pool = []
    for folder, _ in body_pieces(base, axis):
        xyz, meta = sampler.load_surface(folder)
        if folder.name not in read and min(xyz.shape[1:]) * meta["stride"] * meta["voxel_um"] / 1000 >= ROW_SAMPLE_MM:
            pool.append(folder)
    np.random.default_rng(ROW_SAMPLE_SEED).shuffle(pool)
    return [(f, out / f.name, axis) for f in pool[:REPLICATE_PIECES] if not (out / f"{f.name}.json").exists()]


def reread_jobs(sample, volume, band):
    """Pieces of both stages whose rows came from an older ruler, the same pieces read again."""
    base = pieces_of(sample, volume, band)
    axis = axis_of(sample, volume, base)
    rows = [
        p
        for folder in (SURVEYED / sample, REPLICATED / sample)
        for p in sorted(folder.glob("piece-*.json"))
        if json.loads(p.read_text()).get("ruler") != RULER
    ]
    return [(base / p.stem, p.with_suffix(""), axis) for p in rows]


def store_of(kind, sample):
    """Where a set's bricks wait."""
    return ct_bricks(sample, f"survey-{kind}" if kind in ("reread", "row-sample") else "survey")


def summary(folder):
    """A scroll's ruler from its pieces' rows."""
    sample = folder.name
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("piece-*.json"))]
    rows = [r for r in rows if r.get("pixels", 0) > 0]
    if not rows:
        return {"sample": sample, "pieces": 0}
    excess = np.array([(r["forward"] - r["reversed"]) / r["pixels"] for r in rows])
    ratio = np.array([np.sqrt(r["band_forward"] / max(r["band_reversed"], 1e-9)) for r in rows])
    pixels = sum(r["pixels"] for r in rows)
    sheet = sum(r.get("sheet_pixels", 0) for r in rows)
    return {
        "sample": sample,
        "pieces": len(rows),
        "open_share": pixels / sheet if sheet else None,
        "forward": sum(r["forward"] for r in rows) / pixels,
        "reversed": sum(r["reversed"] for r in rows) / pixels,
        "excess": float(excess.mean()),
        "excess_error": float(excess.std(ddof=1) / len(excess) ** 0.5),
        "band_ratio": float(
            np.sqrt(sum(r["band_forward"] for r in rows) / max(sum(r["band_reversed"] for r in rows), 1e-9))
        ),
        "median_piece": float(np.median(ratio)),
        "high": int((ratio > 1.2).sum()),
        "low": int((ratio < 1 / 1.2).sum()),
    }


def read_set(sample, jobs, store_folder, weights=WEIGHTS):
    start, failed = time.time(), 0
    start_worker(str(store_folder), str(weights))
    with ThreadPoolExecutor(PIECES_AT_ONCE) as readers:
        for k, row in enumerate(readers.map(read_piece, jobs)):
            failed += "failed" in row
            if k % 50 == 49 or "failed" in row:
                print(sample, k + 1, "of", len(jobs), row, round(time.time() - start), "s", flush=True)
    shutil.rmtree(store_folder)
    print(
        json.dumps(summary(jobs[0][1].parent) | {"failed": failed, "minutes": round((time.time() - start) / 60, 1)}),
        flush=True,
    )


def run(kind, bands):
    """Bands as sample/volume/band, each set read while the next is fetched."""
    plan, weights = PLANS.get(kind, (jobs_of, WEIGHTS))
    with ThreadPoolExecutor(1) as ahead:

        def prepared(item):
            sample, volume, band = item.split("/")
            jobs = plan(sample, volume, band)
            folder = store_of(kind, sample)
            if jobs:
                fetch_bricks(sample, volume, jobs, folder)
            return sample, jobs, folder

        coming = ahead.submit(prepared, bands[0]) if bands else None
        for k in range(len(bands)):
            sample, jobs, folder = coming.result()
            if k + 1 < len(bands):
                coming = ahead.submit(prepared, bands[k + 1])
            if jobs:
                read_set(sample, jobs, folder, weights)


PLANS = {
    "look": (look_jobs, WEIGHTS),
    "replicate": (replicate_jobs, WEIGHTS),
    "reread": (reread_jobs, WEIGHTS),
    "row-sample": (row_sample_jobs, WEIGHTS),
    "second": (second_model_jobs, SECOND_WEIGHTS),
}

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
