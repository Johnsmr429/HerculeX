"""Typed calls into the sheet-node kernels."""

import ctypes
from pathlib import Path

import numpy as np

from runner.compiled_kernels import load

HERE = Path(__file__).parent
CELL, MARGIN, MAXN, AIR_LEVEL = 8, 8, 8, 40
NODE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "i1"),
        ("ny", "i1"),
        ("nz", "i1"),
        ("prominence", "u1"),
        ("width", "u1"),
        ("coherence", "u1"),
        ("pad", "<u2"),
    ]
)
STATS = np.dtype([("mean", "u1"), ("coherence", "u1"), ("wavelength", "u1"), ("contrast", "u1")])
_GPU = {}


def cpu_library(cell=CELL, margin=MARGIN):
    lib = load(
        f"sheet-nodes-c{cell}",
        [HERE / "node_kernels.c"],
        ("-O3", "-ffast-math", "-mcpu=apple-m1", f"-DCELL={cell}", f"-DMARGIN={margin}"),
    )
    i32, ptr = ctypes.c_int32, ctypes.c_void_p
    lib.hx_cell_row.argtypes = [ptr, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr]
    lib.hx_cell_row.restype = None
    return lib


def gpu_library():
    if "lib" in _GPU:
        return _GPU["lib"]
    lib = load(
        "sheet-nodes-gpu",
        [HERE / "gpu_nodes_host.m"],
        ("-O2", "-fobjc-arc", "-framework", "Metal", "-framework", "Foundation"),
    )
    i32, u32, ptr = ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p
    lib.hx_gpu_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, i32]
    lib.hx_gpu_open.restype = i32
    lib.hx_gpu_cells.argtypes = [ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, u32, ptr, ptr]
    lib.hx_gpu_cells.restype = i32
    message = ctypes.create_string_buffer(4096)
    source = f"#define CELL {CELL}\n#define MARGIN {MARGIN}\n".encode() + (HERE / "gpu_nodes.metal").read_bytes()
    if lib.hx_gpu_open(source, message, len(message)):
        raise RuntimeError(message.value.decode())
    _GPU["lib"] = lib
    return lib
