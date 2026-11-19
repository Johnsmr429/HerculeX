"""Every piece at the scan's own voxels, one unit at a time: layer stack and scan images."""

import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from ink.model.stock_model import NO_INK
from sampling import column_sampling as sampler
from sampling.sheet_images import across_sheet, extended, largest_mosaic, sheet_layers
from sampling.touched_planes import mark_ranges
from scans.catalog.scan_catalog import load, volumes
from scans.transfer.parallel_fetch import fetch_all
from storage.brick_store import ABSENT, UNKNOWN, BrickStore
from storage.data_roots import ROUTE_OUTPUTS, ct_bricks, stage_folder

PROCESSES = int(os.environ.get("HERCULEX_FACE_PROCESSES", "2"))
ALL_LAYERS = bool(os.environ.get("HERCULEX_FACE_ALL_LAYERS"))
FACE_LAYER = 10
_store = None


def axis_points(sample, volume):
    """The published axis of a scan as (z, x, y) control points, in level-0 voxels, or None."""
    kept = stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "axis.json"
    if kept.exists():
        return np.array(json.loads(kept.read_text()), dtype=np.float64)
    row = next((v for v in volumes(load()) if v["sample"] == sample and v["volume"] == volume), None)
    if row is None:
        raise ValueError(f"scan {volume} for {sample} is missing from the local catalog")
    if "umbilicus" not in row["products"]:
        return None
    url = f"{row['roots']['umbilicus']}/{row['products']['umbilicus'].lstrip('/')}"
    for attempt in range(5):
        try:
            points = json.load(urllib.request.urlopen(url, timeout=60))["control_points"]
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    axis = sorted((p["z"], p["x"], p["y"]) for p in points)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(axis))
    return np.array(axis, dtype=np.float64)


def centre_at(axis, z, fallback):
    return (
        fallback
        if axis is None
        else np.array([np.interp(z, axis[:, 0], axis[:, 1]), np.interp(z, axis[:, 0], axis[:, 2]), z])
    )


def axis_of_pieces(units):
    """Without a published axis."""
    means = []
    for unit in units:
        for folder in sorted(unit.glob("piece-*"))[::50]:
            xyz, _ = sampler.load_surface(folder)
            valid = xyz[0] > 0
            if valid.any():
                means.append(xyz[:, valid].mean(axis=1))
    if not means:
        raise ValueError("no valid mesh points to estimate the scroll axis")
    x, y, _ = np.mean(means, axis=0)
    return np.array([[0.0, x, y], [1e7, x, y]])


def start_worker(store_folder):
    global _store
    _store = BrickStore(store_folder)


def ink_window(stack, depth):
    """The layers the model reads."""
    sample = stack[:, ::8, ::8].reshape(stack.shape[0], -1).astype(np.float32)
    profile = sample[:, sample.min(axis=0) > 0].mean(axis=1) if (sample.min(axis=0) > 0).any() else sample.mean(axis=1)
    middle = stack.shape[0] // 2
    fall = middle - 2 + int(np.argmin(np.diff(profile)[middle - 2 : middle + 8]))
    start = min(max(fall - FACE_LAYER, 0), stack.shape[0] - depth)
    return stack[start : start + depth]


def shown(ink, sheet):
    """An ink map as the sheets show it."""
    return np.where(sheet > 0, np.clip((ink - NO_INK) / (1 - 2 * NO_INK), 0, 1) * 255, 0).astype(np.uint8)


def side_record(out):
    return out.with_name(f"{out.stem}-side.json")


def read_turn(out):
    """Which way a piece was last read."""
    record = side_record(out)
    return json.loads(record.read_text())["turn"] if record.exists() else 1


def surface_of(xyz, meta, axis, cells=sampler.MESH_SMOOTHING_CELLS, reach=sampler.BRIDGE_CELLS):
    """A piece's readable surface with its margin and its normals towards the scroll's centre."""
    xyz = sampler.readable_mesh(xyz, meta["stride"], cells, reach)
    good = xyz[0] > 0
    margined = extended(np.where(good, xyz, -1.0).astype(np.float32))
    middle = np.median(xyz[:, good], axis=1)
    return margined, sampler.grid_normals(
        margined, centre_at(axis, middle[2], middle), meta["stride"] * meta["voxel_um"]
    )


def read_face(job):
    """One piece's images."""
    folder, out, axis, turn = job
    start = time.time()
    xyz, meta = sampler.load_surface(folder)
    stride, voxel = meta["stride"], meta["voxel_um"]
    try:
        margined, normal = surface_of(xyz, meta, axis)
        stack = sampler.flatten(
            margined, stride, normal * turn, _store, layers=min(sampler.LAYERS, sheet_layers(voxel))
        )
        sheet, layers = across_sheet(stack, voxel, all_layers=ALL_LAYERS)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sheet).save(out, compress_level=3)
        for depth, layer in layers.items():
            Image.fromarray(layer).save(out.with_name(f"{out.stem}-layer{depth}.png"), compress_level=3)
        side_record(out).write_text(json.dumps({"written": "near", "turn": turn}))
        return {"piece": folder.name, "unit": folder.parent.name, "seconds": round(time.time() - start, 1)}
    except Exception as error:
        return {"piece": folder.name, "unit": folder.parent.name, "failed": repr(error)[:200]}


def jobs_of(sample, volume, units, out, chosen):
    axis = axis_points(sample, volume)
    axis = axis_of_pieces(units) if axis is None else axis
    jobs = []
    for unit in units:
        for r in (
            r
            for r in json.loads((unit / "record.json").read_text())["pieces"]
            if r.get("written") and (unit.name, r["piece"]) in chosen
        ):
            folder = unit / f"piece-{r['piece']:04d}"
            target = out / unit.name / f"{folder.name}.png"
            turn = chosen[(unit.name, r["piece"])]
            if not target.exists() or not side_record(target).exists() or read_turn(target) != turn:
                jobs.append((r["area_mm2"], folder, target, axis, turn))
    return [job[1:] for job in sorted(jobs, key=lambda job: -job[0])]


def ranges_of(store, jobs):
    """Each brick left to fetch, as rows of its index, and the first and last z-plane its stacks read."""
    ranges = np.empty((*store.grid, 2), dtype=np.int32)
    first, last = ranges[..., 0], ranges[..., 1]
    first.fill(store.brick[0])
    last.fill(-1)
    for job in jobs:
        xyz, meta = sampler.load_surface(job[0])
        margined, normal = surface_of(xyz, meta, job[2])
        mark_ranges(ranges, store, margined, normal, meta["stride"])
    chosen = (last >= 0) & (store.offsets == UNKNOWN)
    return np.argwhere(chosen), first[chosen].astype(np.intp), last[chosen].astype(np.intp)


def unit_stamp(jobs, volume=None):
    """Digest of what a store's planes depend on: the scan, the stack's depth, and the records of the units its pieces
    come from. A store made for another scan, depth or build of the unit is fetched again."""
    digest = hashlib.sha256(json.dumps([volume, sampler.LAYERS]).encode())
    for record in sorted({Path(job[0]).parent / "record.json" for job in jobs}):
        if record.is_file():
            digest.update(record.read_bytes())
    return digest.hexdigest()[:16]


@contextmanager
def store_lock(folder):
    """One process fetches into a store at a time; the next waits, then finds what it needs there."""
    path = folder.with_name(f".{folder.name}.fetching")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def fetch_bricks(sample, volume, jobs, store_folder, report=sampler.keep_room, still=None):
    """The z-planes these pieces read, each in its brick's place; with `still`, only if it holds once the store is
    this process's (False is returned then)."""
    folder = Path(store_folder)
    with store_lock(folder):
        if still is not None and not still():
            return False
        fetch_bricks_locked(sample, volume, jobs, folder, report)
    return True


def fetch_bricks_locked(sample, volume, jobs, folder, report):
    start = time.time()
    wanted, stamp = {str(job[0]) for job in jobs}, unit_stamp(jobs, volume)
    planned, stamped = folder / "planned.json", folder / "unit.json"
    stale = planned.exists() and not wanted <= set(json.loads(planned.read_text()))
    if stale or (stamped.exists() and json.loads(stamped.read_text()) != stamp):
        shutil.rmtree(folder)
    store, url = sampler.scan_store(sample, volume, folder, 0)
    if not planned.exists():
        planned.write_text(json.dumps(sorted(wanted)))
        stamped.write_text(json.dumps(stamp))
    if not (store.offsets == UNKNOWN).any():
        print(sample, folder.name, "pieces", len(jobs), "fetched 0", flush=True)
        return
    indices, first, last = ranges_of(store, jobs)
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
    failures = sampler.fetched_all(url, items, deliver, report=lambda _: report(store), fetch=fetch_all)
    store.save()
    if failures:
        raise OSError(f"{len(failures)} bricks failed, first {failures[0]}")
    if wanted < set(json.loads(planned.read_text())):
        planned.write_text(json.dumps(sorted(wanted)))  # the rest of the plan's planes were never fetched
    store.offsets[store.offsets == UNKNOWN] = ABSENT
    store.save()
    moved = int((last - first + 1).sum()) * plane
    print(
        sample,
        folder.name,
        "pieces",
        len(jobs),
        "fetched",
        len(items),
        "bricks,",
        round(moved / 2**30, 2),
        "GiB against",
        round(len(items) * store.size / 2**30, 2),
        "whole,",
        round(time.time() - start),
        "s",
        flush=True,
    )


def read_all(jobs, store_folder, out):
    """The pieces read from their store, the store dropped."""
    start = time.time()
    done = []
    with multiprocessing.get_context("spawn").Pool(PROCESSES, start_worker, (str(store_folder),)) as pool:
        for k, row in enumerate(pool.imap_unordered(read_face, jobs, chunksize=1)):
            done.append(row)
            if k % 50 == 49:
                print(store_folder.name, k + 1, "of", len(jobs), round(time.time() - start), "s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "faces.jsonl").open("a") as f:
        for row in done:
            f.write(json.dumps(row) + "\n")
    failed = [r for r in done if "failed" in r]
    print(
        store_folder.name,
        "done",
        len(done),
        "failed",
        len(failed),
        failed[:2],
        round(time.time() - start),
        "s",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(jobs)} faces in {store_folder.name} failed. Rerun to resume.")
    shutil.rmtree(store_folder)


def run(sample, volume, only=None, largest=False):
    """Every piece of a sheet at the scan's own voxels, or its largest continuous area alone."""
    base = ROUTE_OUTPUTS / sample / volume
    out = stage_folder(base, "faces") / "level-0"
    units = [
        u
        for u in sorted((base / "pieces").glob("rows-*")) or sorted((base / "pieces").glob("band-*"))
        if only is None or u.name in only
    ]
    placed = json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]
    if largest:
        chosen = {(p["unit"], p["piece"]): p.get("turn", 1) for p in largest_mosaic(placed)}
        queue = [(ct_bricks(sample, "largest-area"), jobs_of(sample, volume, units, out, chosen))]
    else:
        chosen = {(p["unit"], p["piece"]): p.get("turn", 1) for p in placed}
        queue = [(ct_bricks(sample, u.name), jobs_of(sample, volume, [u], out, chosen)) for u in units]
    queue = sorted((q for q in queue if q[1]), key=lambda q: not q[0].exists())
    with ThreadPoolExecutor(1) as ahead:
        coming = ahead.submit(fetch_bricks, sample, volume, queue[0][1], queue[0][0]) if queue else None
        for k, (folder, jobs) in enumerate(queue):
            coming.result()
            coming = (
                ahead.submit(fetch_bricks, sample, volume, queue[k + 1][1], queue[k + 1][0])
                if k + 1 < len(queue)
                else None
            )
            read_all(jobs, folder, out)


if __name__ == "__main__":
    run(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "all" else None,
        len(sys.argv) > 4,
    )
