"""Box reads from a chunk pack (pread, threaded) and chunk marking around points, in C."""

import ctypes
from pathlib import Path

import numpy as np

from runner import compiled_kernels as native

HERE = Path(__file__).resolve().parent


def library():
    lib = native.load("chunk-read", [HERE / "read_box.c", HERE / "chunks_near_points.c"])
    if not hasattr(lib, "declared"):
        p = ctypes.c_void_p
        lib.lg_read_box.argtypes = [ctypes.c_int, p, p, p, ctypes.c_uint8, p, ctypes.c_int32]
        lib.lg_read_box.restype = ctypes.c_int
        lib.lg_chunks_near_points.argtypes = [p, ctypes.c_int64, ctypes.c_double, p, p, p]
        lib.lg_chunks_near_points.restype = None
        lib.declared = True
    return lib


def read_box(fd, chunks, box, offsets, fill=0, threads=4, shape=None):
    """uint8 [z, y, x] of box from chunks at offsets (grid over the box, negative = fill); fill beyond shape."""
    out = np.empty((box[1] - box[0], box[3] - box[2], box[5] - box[4]), np.uint8)
    chunks = np.asarray(chunks, np.int64)
    box = np.asarray(box, np.int64)
    offsets = np.ascontiguousarray(offsets, np.int64)
    code = library().lg_read_box(
        fd, native.pointer(chunks), native.pointer(box), native.pointer(offsets), fill, native.pointer(out), threads
    )
    if code:
        raise OSError(f"chunk pack read failed for box {box.tolist()}")
    for axis, size in enumerate(shape or ()):
        if box[2 * axis + 1] > size:
            out[(slice(None),) * axis + (slice(max(0, size - box[2 * axis]), None),)] = fill
    return out


def chunks_near_points(points, margin, chunks, grid):
    """(M, 3) keys of chunks within margin voxels of any zyx point (NaN and out-of-array points skipped)."""
    points = np.ascontiguousarray(points, np.float32).reshape(-1, 3)
    chunks = np.asarray(chunks, np.int64)
    grid = np.asarray(grid, np.int64)
    marks = np.zeros(int(np.prod(grid)), np.uint8)
    library().lg_chunks_near_points(
        native.pointer(points),
        len(points),
        float(margin),
        native.pointer(chunks),
        native.pointer(grid),
        native.pointer(marks),
    )
    return np.stack(np.unravel_index(np.flatnonzero(marks), tuple(grid)), 1).astype(np.int64)
