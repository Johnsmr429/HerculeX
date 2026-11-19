"""Decompressed chunks of one array level: an append-only pack file and its SQLite index (one writer, many readers)."""

import contextlib
import fcntl
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np

ABSENT = -1
UNKNOWN = -2
PACK = "chunks.pack"
INDEX = "index.sqlite"
LOCK = "writer.lock"
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    z INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL, offset INTEGER NOT NULL,
    PRIMARY KEY (z, y, x)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(path):
    db = sqlite3.connect(path, timeout=60, isolation_level=None, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA busy_timeout=60000")
    return db


class ChunkStore:
    def __init__(self, folder, chunk_bytes, url):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chunk_bytes = int(chunk_bytes)
        self.lock = threading.Lock()
        self.db = connect(self.folder / INDEX)
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone():
            self.db.executescript(SCHEMA)
        meta = dict(self.db.execute("SELECT key, value FROM meta"))
        if not meta:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('url', ?), ('chunk_bytes', ?)", (url, str(chunk_bytes)))
            meta = dict(self.db.execute("SELECT key, value FROM meta"))
        if meta["url"] != url or int(meta["chunk_bytes"]) != self.chunk_bytes:
            raise ValueError(f"{self.folder} belongs to {meta['url']} ({meta['chunk_bytes']} B chunks), not {url}")
        self.fd = -1

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        self.db.close()

    def pack_fd(self):
        with self.lock:
            if self.fd < 0 and (self.folder / PACK).exists():
                self.fd = os.open(self.folder / PACK, os.O_RDONLY)
            return self.fd

    def offsets(self, first, last):
        """Dense offsets of chunk keys first <= key < last (per axis); UNKNOWN where nothing is recorded."""
        first, last = [int(v) for v in first], [int(v) for v in last]
        grid = np.full([b - a for a, b in zip(first, last)], UNKNOWN, np.int64)
        if grid.size == 0:
            return grid
        with self.lock:
            rows = self.db.execute(
                "SELECT z, y, x, offset FROM chunks WHERE z >= ? AND z < ? AND y >= ? AND y < ? AND x >= ? AND x < ?",
                (first[0], last[0], first[1], last[1], first[2], last[2]),
            ).fetchall()
        if rows:
            found = np.array(rows, np.int64)
            grid[found[:, 0] - first[0], found[:, 1] - first[1], found[:, 2] - first[2]] = found[:, 3]
        return grid

    def lookup(self, keys):
        """Offsets of arbitrary (M, 3) chunk keys; UNKNOWN where nothing is recorded."""
        keys = np.asarray(keys, np.int64).reshape(-1, 3)
        result = np.full(len(keys), UNKNOWN, np.int64)
        if not len(keys):
            return result
        first, last = keys.min(0), keys.max(0) + 1
        if np.prod(last - first) <= max(4 * len(keys), 1 << 16):
            return self.offsets(first, last)[tuple((keys - first).T)]
        with self.lock:
            self.db.execute("CREATE TEMP TABLE IF NOT EXISTS wanted (i INTEGER, z INTEGER, y INTEGER, x INTEGER)")
            self.db.execute("BEGIN")
            try:
                self.db.executemany(
                    "INSERT INTO wanted VALUES (?, ?, ?, ?)", ((i, *k) for i, k in enumerate(keys.tolist()))
                )
                rows = self.db.execute("SELECT w.i, c.offset FROM wanted w JOIN chunks c USING (z, y, x)").fetchall()
            finally:
                self.db.execute("ROLLBACK")
        if rows:
            found = np.array(rows, np.int64)
            result[found[:, 0]] = found[:, 1]
        return result

    def usage(self):
        with self.lock:
            stored, absent = self.db.execute(
                "SELECT COALESCE(SUM(offset >= 0), 0), COALESCE(SUM(offset < 0), 0) FROM chunks"
            ).fetchone()
        files = {p.name: p.stat().st_size for p in self.folder.iterdir() if p.is_file()}
        return {"stored": stored, "absent": absent, "pack_bytes": files.get(PACK, 0), "bytes": sum(files.values())}

    @contextlib.contextmanager
    def writer(self, wait=True):
        lock = os.open(self.folder / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            writer = ChunkWriter(self)
            try:
                yield writer
            finally:
                writer.close()
        finally:
            os.close(lock)


class ChunkWriter:
    """Appends chunks to the pack, then records them; a killed run leaves only an unrecorded tail, cut on reopen."""

    def __init__(self, store):
        self.store = store
        self.db = connect(store.folder / INDEX)
        self.fd = os.open(store.folder / PACK, os.O_RDWR | os.O_CREAT, 0o644)
        size = os.fstat(self.fd).st_size
        past_end = "FROM chunks WHERE offset >= 0 AND offset + ? > ?"
        lost = self.db.execute(f"SELECT COUNT(*) {past_end}", (store.chunk_bytes, size)).fetchone()
        if lost[0]:
            self.db.execute(f"DELETE {past_end}", (store.chunk_bytes, size))
        self.end = self.db.execute(
            "SELECT COALESCE(MAX(offset), ?) + ? FROM chunks WHERE offset >= 0", (-store.chunk_bytes, store.chunk_bytes)
        ).fetchone()[0]
        if size > self.end:
            os.ftruncate(self.fd, self.end)
        self.repaired = {"dropped_rows": lost[0], "cut_bytes": max(0, size - self.end)}

    def close(self):
        os.close(self.fd)
        self.db.close()

    def put(self, present, data, absent=()):
        """Store decompressed chunks data[i] for keys present[i] and mark keys absent as ABSENT."""
        present = np.asarray(present, np.int64).reshape(-1, 3)
        absent = np.asarray(absent, np.int64).reshape(-1, 3)
        if len(present):
            view = memoryview(np.ascontiguousarray(data).reshape(-1)).cast("B")
            if len(view) != len(present) * self.store.chunk_bytes:
                raise ValueError("data size does not match the number of chunks")
            written = 0
            while written < len(view):
                written += os.pwrite(self.fd, view[written:], self.end + written)
            os.fsync(self.fd)
        rows = [(*map(int, k), self.end + i * self.store.chunk_bytes) for i, k in enumerate(present)]
        rows += [(*map(int, k), ABSENT) for k in absent]
        if rows:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", rows)
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
        self.end += len(present) * self.store.chunk_bytes
