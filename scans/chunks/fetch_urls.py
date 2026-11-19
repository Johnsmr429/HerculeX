"""Many chunk GETs through the C libcurl-multi fetcher: bodies come back in memory batches."""

import ctypes
import os
import resource
from pathlib import Path

import numpy as np

from runner import compiled_kernels as native

HERE = Path(__file__).resolve().parent
BODY = np.dtype([("index", "<i8"), ("offset", "<i8"), ("length", "<i8"), ("status", "<i4"), ("attempts", "<i4")])
STATS = (
    "delivered",
    "absent",
    "failed",
    "bytes",
    "retries",
    "active",
    "queued",
    "connects",
    "needed",
    "aborted",
    "retry_cause",
)
FINISHED, TOO_BIG = -1, -2
SPARE_FILES = 64


def library():
    lib = native.load("chunk-fetch", [HERE / "fetch_urls.c"], libraries=("-lcurl",))
    if not hasattr(lib, "declared"):
        p, i32, i64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
        lib.lg_fetch_keys_start.argtypes = [ctypes.c_char_p, ctypes.c_char_p, p, i64, i32, i32, i32, i64, i64, p]
        lib.lg_fetch_keys_start.restype = p
        lib.lg_fetch_take.argtypes = [p, p, i64, p, i64, i32]
        lib.lg_fetch_take.restype = i64
        lib.lg_fetch_stats.argtypes = [p, p]
        lib.lg_fetch_stop.argtypes = [p]
        lib.lg_fetch_close.argtypes = [p]
        lib.declared = True
    return lib


def usable_connections(wanted):
    """Raise this process's open-file soft limit (macOS starts at 256) so `wanted` more sockets fit, else fewer."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return wanted
    reserved = len(os.listdir("/dev/fd")) + SPARE_FILES
    if soft < reserved + wanted:
        target = reserved + wanted if hard == resource.RLIM_INFINITY else min(reserved + wanted, hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    return max(1, min(wanted, soft - reserved))


class UrlFetch:
    """GET base/z{sep}y{sep}x for every (M, 3) key over kept-alive connections, optionally one byte range (first,
    last) a key; iterate batches() for the bodies."""

    def __init__(
        self,
        base,
        keys,
        separator=".",
        connections=64,
        timeout_ms=120_000,
        retries=6,
        memory_cap=192 << 20,
        max_body=None,
        ranges=None,
    ):
        self.lib = library()
        self.keys = np.ascontiguousarray(keys, np.int64).reshape(-1, 3)
        self.count = len(self.keys)
        self.ranges = None if ranges is None else np.ascontiguousarray(ranges, np.int64).reshape(self.count, 2)
        self.connections = usable_connections(connections)
        self.handle = self.lib.lg_fetch_keys_start(
            base.rstrip("/").encode(),
            separator.encode(),
            native.pointer(self.keys),
            self.count,
            self.connections,
            timeout_ms,
            retries,
            memory_cap,
            max_body or memory_cap,
            None if self.ranges is None else native.pointer(self.ranges),
        )
        if not self.handle:
            raise OSError(f"the chunk fetcher could not start for {base}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.handle:
            self.lib.lg_fetch_close(self.handle)
            self.handle = None

    def stop(self):
        if self.handle:
            self.lib.lg_fetch_stop(self.handle)

    def stats(self):
        values = np.zeros(len(STATS), np.int64)
        if self.handle:
            self.lib.lg_fetch_stats(self.handle, native.pointer(values))
        return dict(zip(STATS, values.tolist()))

    def batches(self, capacity=64 << 20, max_bodies=4096, wait_ms=250):
        """Yield (bodies, buffer): BODY records (index into keys) and the bytes they point into; both are reused."""
        buffer = np.empty(capacity, np.uint8)
        bodies = np.empty(max_bodies, BODY)
        while True:
            n = self.lib.lg_fetch_take(
                self.handle, native.pointer(buffer), len(buffer), native.pointer(bodies), max_bodies, wait_ms
            )
            if n == FINISHED:
                return
            if n == TOO_BIG:
                buffer = np.empty(max(2 * len(buffer), self.stats()["needed"]), np.uint8)
            elif n > 0:
                yield bodies[:n], buffer
