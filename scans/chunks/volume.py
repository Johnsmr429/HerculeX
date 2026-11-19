"""One level of a remote uint8 zarr v2 array, cached chunk by chunk in the outputs' cache/chunks."""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numcodecs
import numpy as np

from scans.chunks import array_registry
from scans.chunks.chunk_store import ABSENT, UNKNOWN, ChunkStore
from scans.chunks.fetch_urls import UrlFetch
from scans.chunks.read_box import chunks_near_points, read_box
from storage.data_roots import CT_CHUNKS as STORE

MIN_FREE_BYTES = 60 * 10**9
DECODED_BATCH_BYTES = 128 << 20
DECODE_THREADS = 4
READ_THREADS = 6


class DiskFull(OSError):
    pass


def guard_disk():
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MIN_FREE_BYTES:
        raise DiskFull(f"{free / 1e9:.1f} GB free on the store's disc, fetching needs {MIN_FREE_BYTES / 1e9:.0f} GB")


def default_connections(chunk_bytes):
    return 64 if chunk_bytes >= 1 << 20 else 256


def parse_box(text):
    """'z0:z1,y0:y1,x0:x1' -> (z0, z1, y0, y1, x0, x1)."""
    return tuple(int(v) for part in text.split(",") for v in part.split(":"))


def as_box(box):
    box = parse_box(box) if isinstance(box, str) else tuple(int(v) for v in box)
    if len(box) != 6 or box[0] >= box[1] or box[2] >= box[3] or box[4] >= box[5]:
        raise ValueError(f"box must be (z0, z1, y0, y1, x0, x1) with each start below its end, got {box}")
    return box


def cause_text(status):
    return f"curl error {-status}" if status < 0 else f"HTTP {status}"


@dataclass
class FetchReport:
    requested: int = 0
    cached: int = 0
    fetched: int = 0
    absent: int = 0
    failed: int = 0
    received_bytes: int = 0
    stored_bytes: int = 0
    retries: int = 0
    retry_cause: int = 0
    connects: int = 0
    connections: int = 0
    seconds: float = 0.0
    network_seconds: float = 0.0
    aborted: bool = False

    @property
    def chunks_per_second(self):
        return (self.fetched + self.absent + self.failed) / max(self.network_seconds, 1e-9)

    @property
    def megabytes_per_second(self):
        return self.received_bytes / 1e6 / max(self.network_seconds, 1e-9)

    def line(self):
        return (
            f"{self.requested} chunks: {self.cached} cached, {self.fetched} fetched, {self.absent} absent, "
            f"{self.failed} failed; {self.received_bytes / 1e6:.1f} MB received in {self.network_seconds:.1f} s "
            f"({self.chunks_per_second:.0f} chunks/s, {self.megabytes_per_second:.1f} MB/s, "
            f"{self.connections} connections, {self.connects} connects, {self.retries} retries"
            + (f", last retry: {cause_text(self.retry_cause)})" if self.retries else ")")
            + (" ABORTED" if self.aborted else "")
        )


class Decoder:
    def __init__(self, compressor, chunk_bytes):
        self.codec = numcodecs.get_codec(compressor) if compressor else None
        self.blosc = bool(compressor) and compressor["id"] == "blosc"
        self.chunk_bytes = chunk_bytes

    def into(self, body, out):
        if self.blosc:
            if len(body) < 16 or int.from_bytes(bytes(body[4:8]), "little") != self.chunk_bytes:
                raise ValueError("blosc chunk of the wrong size")
            self.codec.decode(body, out=out)
            return
        decoded = np.frombuffer(self.codec.decode(body), np.uint8)
        if decoded.size != self.chunk_bytes:
            raise ValueError("chunk of the wrong size")
        out[:] = decoded


class Volume:
    """Volume(url or (sample, name), level): box reads in this level's voxels, zyx order, fill where nothing is known."""

    def __init__(self, source, level=0):
        if isinstance(source, str):
            self.info = array_registry.array_info(source, level)
        else:
            self.info = array_registry.find_array(*source, level)
        self.folder = STORE / self.info.sample / self.info.name / str(self.info.level)
        self.store = ChunkStore(self.folder, self.info.chunk_bytes, self.info.url)
        self.decoder = Decoder(self.info.compressor, self.info.chunk_bytes)

    @classmethod
    def for_scan(cls, sample, scan, kind, level=None):
        from scans.chunks.resolve_scan import array_root

        root, default_level = array_root(sample, scan, kind)
        return cls(root, default_level if level is None else level)

    def __repr__(self):
        return f"Volume({self.info.sample}/{self.info.name}/{self.info.level} shape={self.shape} chunks={self.chunks})"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.store.close()

    @property
    def shape(self):
        return self.info.shape

    @property
    def chunks(self):
        return self.info.chunks

    @property
    def grid(self):
        return self.info.grid

    @property
    def scale(self):
        return self.info.scale

    @property
    def url(self):
        return self.info.url

    def chunk_range(self, box):
        """First and last+1 chunk keys per axis touched by box, clipped to the array."""
        box = as_box(box)
        first = [max(0, box[2 * a] // c) for a, c in enumerate(self.chunks)]
        last = [min(g, (box[2 * a + 1] - 1) // c + 1) for a, (c, g) in enumerate(zip(self.chunks, self.grid))]
        return first, last

    def chunk_keys(self, box):
        first, last = self.chunk_range(box)
        if any(a >= b for a, b in zip(first, last)):
            return np.zeros((0, 3), np.int64)
        axes = np.meshgrid(*[np.arange(a, b) for a, b in zip(first, last)], indexing="ij")
        return np.stack([a.reshape(-1) for a in axes], 1).astype(np.int64)

    def missing(self, box):
        """(M, 3) keys of chunks in box that were never fetched."""
        first, last = self.chunk_range(box)
        if any(a >= b for a, b in zip(first, last)):
            return np.zeros((0, 3), np.int64)
        return np.argwhere(self.store.offsets(first, last) == UNKNOWN) + np.asarray(first, np.int64)

    def chunks_near(self, points, margin):
        """(M, 3) keys of chunks within margin voxels of (N, 3) zyx points given in this level's voxels."""
        return chunks_near_points(points, margin, self.chunks, self.grid)

    def fetch(self, box, connections=None, progress=None):
        return self.fetch_chunks(self.chunk_keys(box), connections, progress)

    def fetch_points(self, points, margin, connections=None, progress=None):
        return self.fetch_chunks(self.chunks_near(points, margin), connections, progress)

    def fetch_chunks(self, keys, connections=None, progress=None):
        """Fetch the chunks among keys that the store does not know yet; resumable, one writer at a time."""
        started = time.monotonic()
        keys = np.asarray(keys, np.int64).reshape(-1, 3)
        inside = np.all((keys >= 0) & (keys < np.asarray(self.grid)), 1)
        keys = np.unique(keys[inside], axis=0)
        report = FetchReport(requested=len(keys))
        if len(keys):
            guard_disk()
            with self.store.writer() as writer:
                todo = keys[self.store.lookup(keys) == UNKNOWN]
                report.cached = len(keys) - len(todo)
                if len(todo):
                    self.download(
                        todo, writer, connections or default_connections(self.info.chunk_bytes), report, progress
                    )
        report.seconds = time.monotonic() - started
        return report

    def download(self, todo, writer, connections, report, progress):
        size = self.info.chunk_bytes
        max_bodies = max(1, DECODED_BATCH_BYTES // size)
        decoded = np.empty((max_bodies, size), np.uint8) if self.decoder.codec else None
        started = shown = time.monotonic()
        fetch = UrlFetch(self.info.url, todo, self.info.separator, connections, max_body=2 * size + (1 << 16))
        report.connections = fetch.connections
        with fetch, ThreadPoolExecutor(DECODE_THREADS) as pool:
            try:
                for bodies, buffer in fetch.batches(
                    capacity=max(64 << 20, 2 * size + (1 << 20)), max_bodies=max_bodies
                ):
                    status = bodies["status"]
                    gone = status == 404
                    keys, data, bad = self.decode(bodies[status == 200], buffer, decoded, pool)
                    writer.put(todo[keys], data, todo[bodies["index"][gone]])
                    report.fetched += len(keys)
                    report.absent += int(gone.sum())
                    report.failed += int((~gone & (status != 200)).sum()) + bad
                    report.received_bytes += int(bodies["length"].sum())
                    report.stored_bytes += len(keys) * size
                    report.network_seconds = time.monotonic() - started
                    if progress and time.monotonic() - shown > 5:
                        shown = time.monotonic()
                        done = report.fetched + report.absent + report.failed
                        progress(f"{done}/{len(todo)} chunks, {report.megabytes_per_second:.1f} MB/s")
                    guard_disk()
            finally:
                stats = fetch.stats()
                report.retries = stats["retries"]
                report.retry_cause = stats["retry_cause"]
                report.connects = stats["connects"]
                report.aborted = bool(stats["aborted"])
                report.network_seconds = time.monotonic() - started

    def decode(self, bodies, buffer, decoded, pool):
        """Indices into todo, (n, chunk_bytes) data and the count of corrupt bodies, for the 200 answers of a batch."""
        size = self.info.chunk_bytes
        index, offset, length = bodies["index"], bodies["offset"], bodies["length"]
        if self.decoder.codec is None:
            good = length == size
            if good.all() and (not len(offset) or offset[-1] == size * (len(offset) - 1)):
                return index, buffer[: len(index) * size].reshape(-1, size), 0
            return index[good], np.stack([buffer[o : o + size] for o in offset[good]]), int((~good).sum())

        def decode_some(rows):
            failed = []
            for j in rows:
                try:
                    self.decoder.into(buffer[offset[j] : offset[j] + length[j]], decoded[j])
                except (ValueError, RuntimeError):
                    failed.append(j)
            return failed

        failed = sorted(
            f for part in pool.map(decode_some, np.array_split(np.arange(len(bodies)), DECODE_THREADS)) for f in part
        )
        good = np.ones(len(bodies), bool)
        good[failed] = False
        data = decoded[: len(bodies)] if good.all() else decoded[: len(bodies)][good]
        return index[good], data, len(failed)

    def read_with_unknown(self, box):
        """(uint8 [z, y, x], bool mask over the box's chunk grid of chunks never fetched)."""
        box = as_box(box)
        first = np.array([box[2 * a] // c for a, c in enumerate(self.chunks)], np.int64)
        last = np.array([(box[2 * a + 1] - 1) // c + 1 for a, c in enumerate(self.chunks)], np.int64)
        offsets = np.full(last - first, ABSENT, np.int64)
        lo, hi = np.maximum(first, 0), np.minimum(last, self.grid)
        if np.all(lo < hi):
            inner = tuple(slice(a, b) for a, b in zip(lo - first, hi - first))
            offsets[inner] = self.store.offsets(lo.tolist(), hi.tolist())
        unknown = offsets == UNKNOWN
        offsets[unknown] = ABSENT
        stored = int((offsets >= 0).sum())
        fd = self.store.pack_fd() if stored else -1
        threads = 1 if stored < 4 else min(READ_THREADS, stored)
        return read_box(fd, self.chunks, box, offsets, self.info.fill_value, threads, self.shape), unknown

    def read(self, box, fetch=False, connections=None):
        """uint8 [z, y, x]; fill where chunks are absent or unknown; fetch=True first fetches what is missing."""
        if fetch:
            self.fetch(box, connections)
        out, unknown = self.read_with_unknown(box)
        if fetch and unknown.any():
            raise OSError(f"{int(unknown.sum())} chunks of box {as_box(box)} could not be fetched")
        return out
