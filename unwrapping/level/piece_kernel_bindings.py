"""Typed calls into the piece kernels: sheet graph, rays, patches and their merging, labels, the network simplex."""

import ctypes
from pathlib import Path

import numpy as np

from runner.compiled_kernels import load

HERE = Path(__file__).parent
EDGE = np.dtype([("a", "<i4"), ("b", "<i4"), ("weight", "<f4"), ("step", "<i4")])
PAIR = np.dtype([("a", "<i4"), ("b", "<i4"), ("rank", "<i4")])
_TYPES = {}


def library():
    lib = load(
        "unwrapper",
        [
            HERE / "piece_kernels.c",
            HERE / "network_simplex.c",
            HERE / "patch_agglomeration.c",
        ],
    )
    if id(lib) in _TYPES:
        return lib
    i32, i64, f32, ptr = ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p
    signatures = {
        "hx_sheet_edges": ([ptr, ptr, i32, i32, i32, i32, ptr, ptr, f32, f32, f32, f32, ptr, i64], i64),
        "hx_forest_init": ([i64, ptr, ptr, ptr, ptr, ptr], None),
        "hx_group_edges": ([ptr, ptr, i64, ptr, i32, ptr, ptr, ptr], None),
        "hx_labels": ([i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr], None),
        "hx_network_simplex": ([i32, i64, ptr, ptr, ptr, ptr, ptr, i64], i64),
        "hx_radial_pairs": ([ptr, ptr, i32, i32, i32, i32, i32, i32, ptr, f32, f32, f32, i32, ptr, i64], i64),
        "hx_mutex_grow": ([ptr, ptr, i64, ptr, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr], None),
        "hx_agglomerate": ([i64, ptr, i64, ptr, ptr, i64, ptr, i64, f32, f32, f32, f32, ptr, ptr], None),
        "hx_cell_table": ([ptr, i64, ptr, i64], None),
        "hx_merge_edges": ([ptr, ptr, i32, ptr], None),
        "hx_sort_edges": ([ptr, i64, ptr], None),
        "hx_turned_steps": ([ptr, i64, ptr, ptr], None),
        "hx_same_label": ([ptr, i64, ptr, ptr], None),
        "hx_same_label_wide": ([ptr, i64, ptr, ptr], None),
        "hx_potential_agrees": ([ptr, i64, ptr, ptr], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    _TYPES[id(lib)] = True
    return lib
