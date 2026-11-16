"""Read a two-dimensional zarr v3 mask stored in blosc-compressed shards, keeping only its non-empty chunks."""

import json
import threading
import urllib.request

import imagecodecs
import numpy as np

from scans.transfer.parallel_fetch import fetch_all


def fetch(url):
    return urllib.request.urlopen(url, timeout=120).read()


def read_cells(base, level=0, connections=16):
    """(height, width, {(y0, x0)."""
    meta = json.loads(fetch(f"{base}/{level}/zarr.json"))
    height, width = meta["shape"]
    shard_h, shard_w = meta["chunk_grid"]["configuration"]["chunk_shape"]
    inner_h, inner_w = meta["codecs"][0]["configuration"]["chunk_shape"]
    per_y, per_x = shard_h // inner_h, shard_w // inner_w
    size = per_y * per_x * 16 + 4
    cells, lock = {}, threading.Lock()

    def deliver(key, shard):
        if shard is None:
            return
        sy, sx = key
        index = np.frombuffer(shard[-size:-4], dtype="<u8").reshape(per_y, per_x, 2)
        for iy in range(per_y):
            for ix in range(per_x):
                offset, nbytes = (int(v) for v in index[iy, ix])
                if offset == 2**64 - 1:
                    continue
                raw = imagecodecs.blosc_decode(shard[offset : offset + nbytes])
                cell = np.frombuffer(raw, dtype=np.uint8).reshape(inner_h, inner_w)
                y0, x0 = sy * shard_h + iy * inner_h, sx * shard_w + ix * inner_w
                h, w = min(inner_h, height - y0), min(inner_w, width - x0)
                if cell[:h, :w].any():
                    bits = np.packbits(cell[:h, :w] > 0, axis=1)
                    with lock:
                        cells[(y0, x0)] = bits

    shards = [((y, x), f"{y}/{x}") for y in range(-(-height // shard_h)) for x in range(-(-width // shard_w))]
    failures = fetch_all(f"{base}/{level}/c", shards, deliver, connections)
    if failures:
        raise OSError(f"{len(failures)} shards of {base} failed")
    return height, width, cells
