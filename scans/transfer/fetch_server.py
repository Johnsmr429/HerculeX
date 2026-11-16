"""Unwrapped products only dl.ash2txt.org holds, mirrored to published/dl_ash2txt."""

import fcntl
import json
import random
import re
import shutil
import sys
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from threading import Lock

from scans.catalog.server_inventory import PATH as SERVER_FILES
from scans.catalog.server_inventory import SERVER
from scans.transfer.fetch_products import (
    copy_stream,
    decodes_to,
    pack_mesh,
    pack_text,
    stored_path,
)
from storage.data_roots import DL_ASH2TXT_MIRROR

ROOT = DL_ASH2TXT_MIRROR
PLAN = ROOT / "plan.jsonl"
DONE = ROOT / "done.jsonl"
FLOOR_GIB = 30
NO_ROOM = 3
WORKERS = 8
PASSES = 3
BATCH = 5000
WHOLE_IN_MEMORY = 300e6
REBUILT = {".ppm", ".vcps", ".vcano", ".orig", ".ply", ".psd", ".mp4", ".mkv"}
SEGMENTS = ("/paths/", "/working/", "/multispectral/", "/volumetric-instance-labels/")
LABELS = ("/datasets/seg-derived-recto-surfaces/labelsTr/", "/datasets/fiber-skeletons/", "/other/annotators/")
PATCHES = re.compile(
    r"/datasets/spiral_datasets/PHercParis4/(verified_patches|eval_fibers|winding_inference|outer_shell|[^/]+\.json)"
)


def rank(path):
    """Fetching order, the irreplaceable first."""
    if path.startswith(LABELS):
        return 2
    if PATCHES.match(path):
        return 4
    if not path.startswith(("/full-scrolls/", "/fragments/")) or not any(part in path for part in SEGMENTS):
        return None
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(name).suffix
    if (
        suffix in REBUILT
        or name.endswith(("_points.obj", "cellmap.tif"))
        or ("/Scroll5/" in path and "/working/" in path)
    ):
        return None
    return {".obj": 1, ".tif": 3}.get(suffix, 0)


def held_path(key):
    return stored_path(key.lstrip("/"), ROOT)


def plan():
    rows = [json.loads(line) for line in SERVER_FILES.read_text().splitlines()]
    wanted = [
        {"key": r["path"], "bytes": r["bytes"], "rank": rank(r["path"])} for r in rows if rank(r["path"]) is not None
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text("".join(json.dumps(row) + "\n" for row in sorted(wanted, key=lambda r: (r["rank"], r["bytes"]))))
    return wanted


def opened(key):
    request = urllib.request.Request(SERVER + urllib.parse.quote(key), headers={"User-Agent": "herculex-inventory"})
    return urllib.request.urlopen(request, timeout=120)


def fetch_key(key):
    target = held_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    with opened(key) as response:
        size = int(response.headers["Content-Length"])
        if key.endswith(".obj"):
            seen = pack_text(response, temporary)
        elif key.endswith(".tif"):
            seen = pack_mesh(response.read(), temporary)
        else:
            seen = copy_stream(response, temporary)
    if seen != size:
        temporary.unlink()
        raise OSError(f"{seen} bytes of {size}")
    temporary.rename(target)
    return seen, target.stat().st_size


def heavy(row):
    return row["key"].endswith(".tif") and row["bytes"] > WHOLE_IN_MEMORY


def finished():
    return {json.loads(line)["key"] for line in DONE.read_text().splitlines()} if DONE.exists() else set()


def fetch():
    lock = open(ROOT / "fetch.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    writing, log, count = Lock(), open(DONE, "a"), [0, 0, 0]

    def one(row):
        if shutil.disk_usage(ROOT).free / 2**30 < FLOOR_GIB:
            return row
        try:
            seen, stored = fetch_key(row["key"])
        except (OSError, ValueError) as error:
            print("failed", row["key"], str(error)[:100], flush=True)
            return row
        with writing:
            log.write(json.dumps({"key": row["key"], "bytes": seen, "stored": stored}) + "\n")
            log.flush()
            count[:] = [count[0] + 1, count[1] + seen, count[2] + stored]
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

    def run(rows, workers):
        left = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(rows), BATCH):
                left += [row for row in pool.map(one, rows[start : start + BATCH]) if row]
        return left

    done = finished()
    todo = [row for row in map(json.loads, PLAN.read_text().splitlines()) if row["key"] not in done]
    for _ in range(PASSES):
        with ThreadPoolExecutor(max_workers=1) as lane:
            alone = lane.submit(run, [r for r in todo if heavy(r)], 1)
            todo = run([r for r in todo if not heavy(r)], WORKERS) + alone.result()
        if not todo or shutil.disk_usage(ROOT).free / 2**30 < FLOOR_GIB:
            break
    (ROOT / "unfinished.json").write_text(json.dumps([row["key"] for row in todo]))
    print("finished, unfinished", len(todo), flush=True)
    lock.close()
    if todo:
        sys.exit(NO_ROOM)


def verify(checks=40):
    """Account for every planned file and compare a random sample with the server, decoded."""
    wanted, done = plan(), finished()
    rows = [json.loads(line) for line in DONE.read_text().splitlines()]
    print(
        len(wanted),
        "wanted",
        len(done),
        "held",
        sum(r["key"] not in done for r in wanted),
        "missing",
        sum(not held_path(key).exists() for key in done),
        "absent on disc",
    )
    print(
        round(sum(r["bytes"] for r in rows) / 1e9, 2),
        "GB published",
        round(sum(r["stored"] for r in rows) / 1e9, 2),
        "GB held",
    )
    small = [r["key"] for r in rows if r["bytes"] < 100e6]
    sample = random.Random(0).sample(small, min(checks, len(small)))
    same = 0
    for key in sample:
        with opened(key) as response:
            same += decodes_to(key, held_path(key).read_bytes(), response.read())
    print(same, "of", len(sample), "identical to the server when decoded")


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "plan":
        wanted = plan()
        print(len(wanted), "files", round(sum(r["bytes"] for r in wanted) / 1e9, 2), "GB")
        print(sorted(Counter(r["rank"] for r in wanted).items()))
    elif action == "fetch":
        fetch()
    else:
        verify()
