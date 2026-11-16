"""Stream one OME-Zarr level of a scan into a brick store."""

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scans.catalog.product_inventory import listing
from scans.catalog.scan_catalog import BUCKET
from scans.transfer.parallel_fetch import fetch_all
from storage.brick_store import ABSENT, BrickStore


def published_bricks(root, prefix, rows):
    """List the published bricks of each z row, so masked-out bricks need no GET."""

    def row_keys(z):
        start = f"{prefix}/{z}/"
        keys, _, complete = listing(root, start, pages=64)
        if not complete:
            raise OSError(f"incomplete brick listing for {start}")
        bricks = set()
        for key, _ in keys:
            y, sep, x = key.removeprefix(start).partition("/")
            if sep and y.isdecimal() and x.isdecimal():
                bricks.add((z, int(y), int(x)))
        return bricks

    with ThreadPoolExecutor(12) as pool:
        return set().union(*pool.map(row_keys, rows))


def fetch_level(path, level, out, connections=32, root=BUCKET, z_bricks=None, *, list_bricks=False):
    base = f"{root}/{path.strip('/')}/{level}"
    array = json.load(urllib.request.urlopen(f"{base}/.zarray", timeout=60))
    if array["dtype"] != "|u1" or array["compressor"] is not None:
        raise ValueError(f"unsupported array: {array}")
    store = BrickStore(out, tuple(array["shape"]), tuple(array["chunks"]), source=f"{path}/{level}")
    todo = [tuple(int(v) for v in i) for i in store.missing()]
    if z_bricks is not None:
        todo = [i for i in todo if z_bricks[0] <= i[0] < z_bricks[1]]
    listed = False
    if list_bricks and todo and urllib.parse.urlsplit(root).hostname == urllib.parse.urlsplit(BUCKET).hostname:
        try:
            published = published_bricks(root, f"{path.strip('/')}/{level}", sorted({z for z, _, _ in todo}))
        except OSError as error:
            print(f"brick listing failed, checking individual bricks: {error}", flush=True)
        else:
            absent = [i for i in todo if i not in published]
            for i in absent:
                store.offsets[i] = ABSENT
            store.save()
            print(f"{len(absent)} masked-out bricks skipped", flush=True)
            todo = [i for i in todo if i in published]
            listed = True
    start, done = time.time(), {"bytes": 0, "count": 0}

    def deliver(index, body):
        if listed and body is None:
            raise OSError(f"listed brick {index} is unavailable")
        store.put(index, body)
        done["count"] += 1
        done["bytes"] += len(body) if body else 0
        if done["count"] % 200 == 0:
            store.save()
            rate = done["bytes"] / 1e6 / (time.time() - start)
            print(f"{done['count']}/{len(todo)} bricks {rate:.1f} MB/s", flush=True)

    items = [(i, "/".join(str(v) for v in i)) for i in todo]
    failures = fetch_all(base, items, deliver, connections)
    store.save()
    print(
        f"{path} level {level}: {done['count']} bricks, {done['bytes'] / 1e9:.2f} GB, "
        f"{time.time() - start:.0f} s, {len(failures)} failures",
        flush=True,
    )
    return store


if __name__ == "__main__":
    fetch_level(sys.argv[1], int(sys.argv[2]), sys.argv[3])
