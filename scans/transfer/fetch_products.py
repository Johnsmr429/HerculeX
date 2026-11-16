"""Everything published about the unwrapped scrolls, mirrored to published/open_data."""

import fcntl
import io
import json
import random
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tifffile
import zstandard

from scans.catalog.product_inventory import PATH as PRODUCTS
from scans.catalog.product_inventory import listing
from scans.catalog.scan_catalog import BUCKET, load
from storage.data_roots import OPEN_DATA_MIRROR

PLAN = OPEN_DATA_MIRROR / "plan.jsonl"
DONE = OPEN_DATA_MIRROR / "done.jsonl"
FLOOR_GIB = 30
WORKERS = 12
PASSES = 5
NOT_MIRRORED = {"surface-volumes", "predictions", "spiral"}
THINNED = {"ink-detection", "alpha-render"}
UNREADABLE = (ValueError, TypeError, tifffile.TiffFileError)


def skipped_folder(prefix):
    name = prefix.rstrip("/").rsplit("/", 1)[1]
    return name in NOT_MIRRORED or name.endswith(".zarr")


def walk_folder(prefix):
    """Every key below a folder, and the brick stores left out, from the bucket itself and not the catalog."""
    keys, folders, _ = listing(BUCKET, prefix, "/", pages=1000)
    found, skipped = list(keys), []
    for folder in folders:
        if skipped_folder(folder):
            skipped.append(folder)
        else:
            below, left = walk_folder(folder)
            found += below
            skipped += left
    return found, skipped


def starting_points(sample):
    """A sample's segments one by one, so that a large scroll spreads over the pool."""
    keys, folders, _ = listing(BUCKET, f"{sample}/", "/", pages=1000)
    points = []
    for folder in folders:
        if folder.endswith("/segments/"):
            points += listing(BUCKET, folder, "/", pages=1000)[1]
        else:
            points.append(folder)
    return keys, points


def walk():
    thinned = {r["path"] for r in map(json.loads, PRODUCTS.read_text().splitlines()) if r["kind"] in THINNED}
    with ThreadPoolExecutor(max_workers=12) as pool:
        starts = list(pool.map(starting_points, sorted(load())))
        found = listing(BUCKET, "", "/")[0] + [key for keys, _ in starts for key in keys]
        points = [point for _, group in starts for point in group]
        walked = list(pool.map(walk_folder, points))
    found += [key for keys, _ in walked for key in keys]
    left_out = sorted({folder for _, skipped in walked for folder in skipped})
    wanted = [(key, size) for key, size in found if key not in thinned and not key.endswith("/")]
    OPEN_DATA_MIRROR.mkdir(parents=True, exist_ok=True)
    PLAN.write_text("".join(json.dumps({"key": k, "bytes": s}) + "\n" for k, s in sorted(wanted)))
    (OPEN_DATA_MIRROR / "left-out.json").write_text(
        json.dumps({"brick_stores": left_out, "thinned": sorted(thinned)}, indent=0)
    )
    return wanted, left_out


def stored_path(key, root=OPEN_DATA_MIRROR):
    return root / (f"{key}.zst" if key.endswith(".obj") else key)


def is_mesh_plane(key):
    return key.endswith(".tif") and ("/mesh/" in key or "/segments/raw/" in key)


def pack_text(response, path):
    seen = 0
    with open(path, "wb") as out, zstandard.ZstdCompressor(level=6).stream_writer(out) as writer:
        while chunk := response.read(1 << 20):
            writer.write(chunk)
            seen += len(chunk)
    return seen


def pack_mesh(raw, path):
    """Recompress a mesh plane."""
    packed = raw
    try:
        image = tifffile.imread(io.BytesIO(raw))
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, image, compression="zstd", compressionargs={"level": 9}, predictor=True)
        again = tifffile.imread(io.BytesIO(buffer.getvalue()))
        if again.dtype == image.dtype and again.shape == image.shape and again.tobytes() == image.tobytes():
            packed = min(raw, buffer.getvalue(), key=len)
    except UNREADABLE:
        pass
    path.write_bytes(packed)
    return len(raw)


def copy_stream(response, path):
    with open(path, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
        return out.tell()


def fetch_key(key, size):
    target = stored_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120) as response:
        if key.endswith(".obj"):
            seen = pack_text(response, temporary)
        elif is_mesh_plane(key):
            seen = pack_mesh(response.read(), temporary)
        else:
            seen = copy_stream(response, temporary)
    if seen != size:
        temporary.unlink()
        raise OSError(f"{seen} bytes of {size}")
    temporary.rename(target)
    return target.stat().st_size


def wait_for_room():
    while shutil.disk_usage(OPEN_DATA_MIRROR).free / 2**30 < FLOOR_GIB:
        print("waiting for room on the internal disc", flush=True)
        time.sleep(300)


def finished():
    return {json.loads(line)["key"] for line in DONE.read_text().splitlines()} if DONE.exists() else set()


def interleaved(rows):
    """Small files beside large ones."""
    half = (len(rows) + 1) // 2
    pairs = zip(rows[:half], reversed(rows[half:]))
    return [row for pair in pairs for row in pair] + (rows[half - 1 : half] if len(rows) % 2 else [])


def fetch():
    lock = open(OPEN_DATA_MIRROR / "fetch.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = [json.loads(line) for line in PLAN.read_text().splitlines()]
    writing = threading.Lock()
    log = open(DONE, "a")
    count = [0, 0, 0]

    def one(row):
        wait_for_room()
        try:
            stored = fetch_key(row["key"], row["bytes"])
        except (OSError, ValueError) as error:
            print("failed", row["key"], str(error)[:100], flush=True)
            return row
        with writing:
            log.write(json.dumps({**row, "stored": stored}) + "\n")
            log.flush()
            count[0] += 1
            count[1] += row["bytes"]
            count[2] += stored
            if count[0] % 500 == 0:
                print(
                    count[0],
                    "files",
                    round(count[1] / 1e9, 2),
                    "GB fetched",
                    round(count[2] / 1e9, 2),
                    "GB held",
                    flush=True,
                )
        return None

    done = finished()
    todo = interleaved(sorted((r for r in plan if r["key"] not in done), key=lambda r: r["bytes"]))
    for _ in range(PASSES):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            todo = [row for row in pool.map(one, todo) if row]
        if not todo:
            break
        time.sleep(120)
    (OPEN_DATA_MIRROR / "unfinished.json").write_text(json.dumps([row["key"] for row in todo]))
    print("finished, unfinished", len(todo), flush=True)
    lock.close()


def decodes_to(key, held, raw):
    """Whether a held file is its published bytes once decoded."""
    if key.endswith(".obj"):
        return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(held)).read() == raw
    if key.endswith(".tif") and held != raw:
        return tifffile.imread(io.BytesIO(held)).tobytes() == tifffile.imread(io.BytesIO(raw)).tobytes()
    return held == raw


def same_as_published(key):
    """Fetch a key again and compare it with what is held, decoded."""
    raw = urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120).read()
    return decodes_to(key, stored_path(key).read_bytes(), raw)


def verify(checks=60):
    """Walk the bucket afresh and account for every key."""
    wanted, left_out = walk()
    done = finished()
    missing = [key for key, _ in wanted if key not in done]
    absent = [key for key in done if not stored_path(key).exists()]
    rows = [json.loads(line) for line in DONE.read_text().splitlines()]
    print(
        len(wanted),
        "keys published and wanted",
        len(done),
        "held",
        len(missing),
        "missing",
        len(absent),
        "absent on disc",
    )
    print(
        round(sum(r["bytes"] for r in rows) / 1e9, 2),
        "GB published",
        round(sum(r["stored"] for r in rows) / 1e9, 2),
        "GB held",
    )
    print(Counter(Path(key).suffix for key in missing).most_common(8), missing[:5])
    small = [r["key"] for r in rows if r["bytes"] < 200e6]
    sample = random.Random(0).sample(small, min(checks, len(small)))
    print(sum(same_as_published(key) for key in sample), "of", len(sample), "identical to published when decoded")
    print(len(left_out), "brick stores left out")


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "walk":
        wanted, left_out = walk()
        print(
            len(wanted), "keys", round(sum(s for _, s in wanted) / 1e9, 2), "GB", len(left_out), "brick stores left out"
        )
        print(Counter(Path(k).suffix for k, _ in wanted).most_common(14))
    elif action == "fetch":
        fetch()
    else:
        verify()
