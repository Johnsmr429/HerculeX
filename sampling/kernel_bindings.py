"""Typed call into column_kernels.c."""

import ctypes
from pathlib import Path

from runner.compiled_kernels import load

HERE = Path(__file__).parent
_TYPED = {}


def library():
    lib = load("columns", [HERE / "column_kernels.c"])
    if id(lib) not in _TYPED:
        i32, i64, f32, ptr = ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p
        lib.hx_sample_columns.argtypes, lib.hx_sample_columns.restype = (
            [ptr, ptr, ptr, ptr, ptr, i64, i32, f32, ptr],
            None,
        )
        lib.hx_sample_grid.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            i64,
            i64,
            ptr,
            ptr,
            ptr,
            ptr,
            i64,
            i64,
            i64,
            i64,
            i32,
            i32,
            f32,
            ptr,
            ptr,
            ptr,
        ]
        lib.hx_sample_grid.restype = i64
        _TYPED[id(lib)] = True
    return lib
