"""The z-plane range of each brick a layer stack needs."""

import ctypes
from pathlib import Path

import numpy as np

from runner.compiled_kernels import load
from sampling import column_sampling as sampler

HERE = Path(__file__).parent
SAFE_VOXELS = 2
_TYPED = {}


def library():
    lib = load("touched-planes", [HERE / "touched_planes.c"], headers=[HERE / "column_kernels.c"])
    if id(lib) not in _TYPED:
        i32, i64, f32, ptr = ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p
        lib.hx_touched_ranges.argtypes, lib.hx_touched_ranges.restype = (
            [ptr, ptr, i64, i64, ptr, ptr, ptr, ptr, i64, i64, i32, f32, i32, ptr],
            None,
        )
        _TYPED[id(lib)] = True
    return lib


def mark_ranges(ranges, store, xyz, normal, stride):
    """Extend each brick's inclusive plane range with one surface's reads."""
    _, h, w = xyz.shape
    grid = np.ascontiguousarray(np.concatenate([(xyz[:1] > 0).astype(np.float32), xyz, normal]), dtype=np.float32)
    (row, row_part), (col, col_part) = sampler.column_cells(h, stride), sampler.column_cells(w, stride)
    geometry = np.array([*store.grid, *store.brick], dtype=np.int32)
    library().hx_touched_ranges(
        geometry.ctypes.data,
        grid.ctypes.data,
        h,
        w,
        row.ctypes.data,
        col.ctypes.data,
        row_part.ctypes.data,
        col_part.ctypes.data,
        len(row),
        len(col),
        sampler.LAYERS,
        -(sampler.LAYERS // 2),
        SAFE_VOXELS,
        ranges.ctypes.data,
    )
