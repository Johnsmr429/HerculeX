"""A scan level or surface volume as one file of raw bricks and an offset table."""

import itertools
import json
import threading
from pathlib import Path

import numpy as np

UNKNOWN, ABSENT = -1, -2


class BrickStore:
    def __init__(self, folder, shape=None, brick=(128, 128, 128), source=None):
        self.folder = Path(folder)
        meta = self.folder / "meta.json"
        if shape is not None and not meta.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({"shape": list(shape), "brick": list(brick), "source": source}))
            grid = tuple(-(-s // b) for s, b in zip(shape, brick))
            np.save(self.folder / "offsets.npy", np.full(grid, UNKNOWN, dtype=np.int64))
        info = json.loads(meta.read_text())
        self.shape, self.brick = tuple(info["shape"]), tuple(info["brick"])
        self.source = info.get("source")
        self.offsets = np.load(self.folder / "offsets.npy")
        self.grid = self.offsets.shape
        self.path = self.folder / "bricks.bin"
        self.path.touch()
        self.size = int(np.prod(self.brick))
        self.lock = threading.Lock()
        self.data = None

    def missing(self):
        return np.argwhere(self.offsets == UNKNOWN)

    def put(self, index, raw):
        with self.lock:
            if raw is None:
                self.offsets[index] = ABSENT
                return
            with open(self.path, "ab") as out:
                offset = out.tell()
                out.write(raw)
            self.offsets[index] = offset

    def save(self):
        with self.lock:
            temporary = self.folder / "offsets.tmp.npy"
            np.save(temporary, self.offsets)
            temporary.replace(self.folder / "offsets.npy")

    def _open(self):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self.data

    def read_brick(self, index):
        offset = self.offsets[index]
        if offset < 0:
            return None
        return self._open()[offset : offset + self.size].reshape(self.brick)

    def read(self, z, y, x):
        """The (z, y, x) ranges of the level as an array."""
        box = (z, y, x)
        out = np.zeros(tuple(b - a for a, b in box), dtype=np.uint8)
        spans = [range(a // n, -(-b // n)) for (a, b), n in zip(box, self.brick)]
        for index in itertools.product(*spans):
            cube = self.read_brick(index)
            if cube is None:
                continue
            target, source = [], []
            for i, n, (a, b) in zip(index, self.brick, box):
                lo, hi = max(a, i * n), min(b, (i + 1) * n)
                target.append(slice(lo - a, hi - a))
                source.append(slice(lo - i * n, hi - i * n))
            out[tuple(target)] = cube[tuple(source)]
        return out
