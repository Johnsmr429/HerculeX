"""Registry of remote zarr v2 arrays, resolver answers and small documents: cache/chunks/arrays.sqlite."""

import contextlib
import hashlib
import json
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass
from math import prod

from scans.chunks import bucket
from storage.data_roots import CT_CHUNKS as STORE

REGISTRY = STORE / "arrays.sqlite"
SCHEMA = """
CREATE TABLE IF NOT EXISTS arrays (
    root TEXT NOT NULL, level INTEGER NOT NULL, url TEXT NOT NULL UNIQUE,
    sample TEXT NOT NULL, name TEXT NOT NULL,
    shape TEXT NOT NULL, chunks TEXT NOT NULL, dtype TEXT NOT NULL, fill_value INTEGER NOT NULL,
    compressor TEXT, separator TEXT NOT NULL, scale TEXT NOT NULL, added REAL NOT NULL,
    PRIMARY KEY (root, level), UNIQUE (sample, name, level));
CREATE TABLE IF NOT EXISTS resolved (
    sample TEXT NOT NULL, scan TEXT NOT NULL, kind TEXT NOT NULL, url TEXT NOT NULL, level INTEGER,
    source TEXT NOT NULL, candidates TEXT NOT NULL, resolved REAL NOT NULL,
    PRIMARY KEY (sample, scan, kind));
CREATE TABLE IF NOT EXISTS documents (url TEXT PRIMARY KEY, body BLOB NOT NULL, fetched REAL NOT NULL);
"""
GENERIC_FOLDERS = {"volumes", "surfaces", "predictions", "representations"}


@dataclass(frozen=True)
class ArrayInfo:
    root: str
    level: int
    url: str
    sample: str
    name: str
    shape: tuple
    chunks: tuple
    dtype: str
    fill_value: int
    compressor: dict | None
    separator: str
    scale: tuple

    @property
    def chunk_bytes(self):
        return prod(self.chunks)

    @property
    def grid(self):
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    def chunk_url(self, z, y, x):
        return f"{self.url}/{z}{self.separator}{y}{self.separator}{x}"


def connect():
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(REGISTRY, timeout=60, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=60000")
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'documents'").fetchone():
        db.executescript(SCHEMA)
    return db


@contextlib.contextmanager
def registry():
    db = connect()
    try:
        yield db
    finally:
        db.close()


def sample_and_name(root):
    """Sample and a readable store name from the array root URL (bucket folder names, run folder for groups)."""
    parts = urllib.parse.urlsplit(root).path.strip("/").split("/")
    if parts[0] == "samples" and len(parts) > 2:
        parts = parts[1:]
    sample, folder = parts[0], parts[-1]
    stem = folder.removesuffix(".zarr").removesuffix(".ome")
    if stem.startswith(f"{sample}_") and len(parts) > 2 and parts[-2] not in GENERIC_FOLDERS:
        stem = f"{parts[-2]}-{stem.removeprefix(f'{sample}_')}"
    return sample, stem


def level_scale(attributes, level):
    for scales in attributes.get("multiscales", []):
        for dataset in scales.get("datasets", []):
            if str(dataset.get("path")) == str(level):
                for transform in dataset.get("coordinateTransformations", []):
                    if transform.get("type") == "scale":
                        return tuple(float(v) for v in transform["scale"][-3:])
    return (float(2**level),) * 3


def from_row(row):
    root, level, url, sample, name, shape, chunks, dtype, fill, compressor, separator, scale = row
    return ArrayInfo(
        root,
        level,
        url,
        sample,
        name,
        tuple(json.loads(shape)),
        tuple(json.loads(chunks)),
        dtype,
        fill,
        json.loads(compressor) if compressor else None,
        separator,
        tuple(json.loads(scale)),
    )


COLUMNS = "root, level, url, sample, name, shape, chunks, dtype, fill_value, compressor, separator, scale"


def array_info(root, level):
    """Shape, chunks and codec of one level, read from .zarray once and kept in the registry."""
    root = root.rstrip("/")
    with registry() as db:
        row = db.execute(f"SELECT {COLUMNS} FROM arrays WHERE root = ? AND level = ?", (root, level)).fetchone()
    if row:
        return from_row(row)
    url = f"{root}/{level}"
    zarray = bucket.get_json(f"{url}/.zarray")
    if zarray.get("zarr_format") != 2 or zarray["dtype"] not in ("|u1", "<u1", "u1"):
        raise ValueError(f"{url}: only uint8 zarr v2 arrays are supported, got {zarray}")
    if zarray.get("order", "C") != "C" or zarray.get("filters"):
        raise ValueError(f"{url}: order C without filters is required, got {zarray}")
    try:
        attributes = bucket.get_json(f"{root}/.zattrs")
    except FileNotFoundError:
        attributes = {}
    sample, name = sample_and_name(root)
    values = [
        root,
        level,
        url,
        sample,
        name,
        json.dumps(zarray["shape"]),
        json.dumps(zarray["chunks"]),
        "uint8",
        int(zarray.get("fill_value") or 0),
        json.dumps(zarray["compressor"]) if zarray.get("compressor") else None,
        zarray.get("dimension_separator", "."),
        json.dumps(level_scale(attributes, level)),
    ]
    with registry() as db:
        taken = db.execute(
            "SELECT root FROM arrays WHERE sample = ? AND name = ? AND level = ? AND root != ?",
            (sample, name, level, root),
        ).fetchone()
        if taken:
            values[4] = f"{name}-{hashlib.sha1(root.encode()).hexdigest()[:6]}"
        db.execute(
            f"INSERT OR IGNORE INTO arrays ({COLUMNS}, added) VALUES ({', '.join('?' * 13)})", (*values, time.time())
        )
        row = db.execute(f"SELECT {COLUMNS} FROM arrays WHERE root = ? AND level = ?", (root, level)).fetchone()
    return from_row(row)


def find_array(sample, name, level):
    with registry() as db:
        row = db.execute(
            f"SELECT {COLUMNS} FROM arrays WHERE sample = ? AND name = ? AND level = ?", (sample, name, level)
        ).fetchone()
        if row:
            return from_row(row)
        other = db.execute("SELECT root FROM arrays WHERE sample = ? AND name = ? LIMIT 1", (sample, name)).fetchone()
    if other:
        return array_info(other[0], level)
    raise KeyError(f"no registered array {sample}/{name}; open it once by URL")


def all_arrays():
    with registry() as db:
        return [from_row(r) for r in db.execute(f"SELECT {COLUMNS} FROM arrays ORDER BY sample, name, level")]


def resolved(sample, scan):
    with registry() as db:
        rows = db.execute("SELECT kind, url, level FROM resolved WHERE sample = ? AND scan = ?", (sample, scan))
        return {kind: (url, level) for kind, url, level in rows}


def save_resolved(sample, scan, answers, source):
    """answers: kind -> (url, level or None, candidates)."""
    now = time.time()
    with registry() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM resolved WHERE sample = ? AND scan = ?", (sample, scan))
        db.executemany(
            "INSERT INTO resolved VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(sample, scan, k, u, lv, source, json.dumps(c), now) for k, (u, lv, c) in answers.items()],
        )
        db.execute("COMMIT")
