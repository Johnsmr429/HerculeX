"""A sheet's image pyramid as one pack of tiles and one index, read by byte range."""

import hashlib
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image

INDEX = "tiles.index"
MAGIC = b"HXTILES1"


def cells_of(grid):
    return sum(columns * rows for columns, rows in grid)


def cell(entry, level, c, r):
    """Where a tile stands in its layer's table, None outside the part of the sheet the layer covers."""
    grid = entry["grid"]
    if not 0 <= level < len(grid):
        return None
    c0, r0 = entry["origin"][level] if "origin" in entry else (0, 0)
    if not (0 <= c - c0 < grid[level][0] and 0 <= r - r0 < grid[level][1]):
        return None
    return cells_of(grid[:level]) + (r - r0) * grid[level][0] + c - c0


def packed(folder):
    return (Path(folder) / INDEX).exists()


class TileStore:
    """Every layer of a sheet folder, a tile at a time."""

    def __init__(self, folder):
        self.folder = Path(folder)
        data = (self.folder / INDEX).read_bytes()
        if data[:8] != MAGIC:
            raise ValueError(f"not a tile index: {self.folder / INDEX}")
        size = struct.unpack_from("<I", data, 8)[0]
        header = json.loads(data[12 : 12 + size])
        start, cells = -(-(12 + size) // 8) * 8, header["cells"]
        self.offsets = np.frombuffer(data, dtype="<f8", count=cells, offset=start).astype(np.int64)
        self.lengths = np.frombuffer(data, dtype="<u4", count=cells, offset=start + 8 * cells)
        self.layers, self.pack = header["layers"], self.folder / header["pack"]

    def place(self, layer, level, c, r):
        """(offset, length) of a tile in the pack, None where the sheet shows nothing."""
        entry = self.layers[layer]
        k = cell(entry, level, c, r)
        if k is None or not self.lengths[entry["first"] + k]:
            return None
        return int(self.offsets[entry["first"] + k]), int(self.lengths[entry["first"] + k])

    def tile(self, layer, level, c, r):
        """A tile as stored: PNG at level 0 of a raw layer, JPEG elsewhere."""
        found = self.place(layer, level, c, r)
        if found is None:
            return None
        with open(self.pack, "rb") as pack:
            pack.seek(found[0])
            return pack.read(found[1])

    def image(self, layer, level, c, r):
        blob = self.tile(layer, level, c, r)
        return None if blob is None else np.asarray(Image.open(io.BytesIO(blob)).convert("L"))

    def tiles(self, layer):
        """(level, c, r, offset, length) of every tile of a layer, in pack order."""
        entry = self.layers[layer]
        found = []
        for level, (columns, rows) in enumerate(entry["grid"]):
            first = entry["first"] + cells_of(entry["grid"][:level])
            c0, r0 = entry["origin"][level] if "origin" in entry else (0, 0)
            for k in np.flatnonzero(self.lengths[first : first + columns * rows]):
                place = map(int, (self.offsets[first + k], self.lengths[first + k]))
                found.append((level, c0 + int(k % columns), r0 + int(k // columns), *place))
        return sorted(found, key=lambda tile: tile[3])


class TileWriter:
    """One composition: new layers go into a new pack, the sheet's other layers are carried over, the index lands last."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"tiles-{time.time_ns()}.bin"
        self.pack = open(self.path, "w+b")
        self.layers, self.tables, self.known, self.end = {}, {}, {}, 0

    def __enter__(self):
        return self

    def __exit__(self, failed, *_):
        if failed:
            self.pack.close()
            self.path.unlink(missing_ok=True)
        else:
            self.close()

    def layer(self, path, grid, **facts):
        """A layer to be written, by the (columns, rows) of each level; origin, if given, is each level's first (column, row)."""
        self.layers[path] = {"grid": [list(level) for level in grid], **facts}
        self.tables[path] = (np.zeros(cells_of(grid), dtype="<f8"), np.zeros(cells_of(grid), dtype="<u4"))

    def add(self, path, level, c, r, blob):
        """One tile; tiles with the same bytes are kept once."""
        digest = hashlib.sha1(blob).digest()
        if digest not in self.known:
            self.pack.seek(self.end)
            self.pack.write(blob)
            self.known[digest] = (self.end, len(blob))
            self.end += len(blob)
        offsets, lengths = self.tables[path]
        k = cell(self.layers[path], level, c, r)
        offsets[k], lengths[k] = self.known[digest]

    def place(self, path, level, c, r):
        offsets, lengths = self.tables[path]
        k = cell(self.layers[path], level, c, r)
        return None if k is None or not lengths[k] else (int(offsets[k]), int(lengths[k]))

    def flush(self):
        self.pack.flush()

    def carry(self, old, path, levels=None):
        """A layer of another store, byte for byte."""
        entry = old.layers[path]
        grid = entry["grid"][:levels]
        facts = {k: v for k, v in entry.items() if k not in ("grid", "first", "origin")}
        self.layer(path, grid, **({"origin": entry["origin"][:levels]} if "origin" in entry else {}), **facts)
        with open(old.pack, "rb") as pack:
            for level, c, r, offset, length in old.tiles(path):
                if level < len(grid):
                    pack.seek(offset)
                    self.add(path, level, c, r, pack.read(length))

    def close(self):
        old = TileStore(self.folder) if packed(self.folder) else None
        for path in [] if old is None else [path for path in old.layers if path not in self.layers]:
            self.carry(old, path)
        self.pack.close()
        header, first = {"pack": self.path.name, "layers": {}}, 0
        for path, entry in self.layers.items():
            header["layers"][path] = entry | {"first": first}
            first += cells_of(entry["grid"])
        header["cells"] = first
        text = json.dumps(header).encode()
        index = MAGIC + struct.pack("<I", len(text)) + text
        index += b"\0" * (-len(index) % 8)
        index += b"".join(self.tables[path][0].tobytes() for path in self.layers)
        index += b"".join(self.tables[path][1].tobytes() for path in self.layers)
        part = self.folder / (INDEX + ".part")
        part.write_bytes(index)
        part.rename(self.folder / INDEX)
        if old is not None and old.pack != self.path:
            old.pack.unlink(missing_ok=True)
