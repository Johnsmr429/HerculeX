"""Typed calls into fibre_kernels.c, each equal to the scipy or numpy steps it stands for."""

import ctypes
from pathlib import Path

import numpy as np

from runner.compiled_kernels import C_FLAGS, load

HERE = Path(__file__).parent
FUSED_BELOW_RADIUS, PLAIN_BLOCK = 10, 8
TRUNCATE = 4.0
_TYPED = {}


def library():
    lib = load("fibres", [HERE / "fibre_kernels.c"], (*C_FLAGS, "-ffp-contract=off"))
    if id(lib) not in _TYPED:
        i32, i64, f64, ptr = ctypes.c_int32, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
        lib.hx_gaussian_axis.argtypes, lib.hx_gaussian_axis.restype = [ptr, ptr, ptr, i32, i32, ptr, i32, i32], None
        lib.hx_face_depth.argtypes, lib.hx_face_depth.restype = [ptr, i64, i64, i64, i64, ptr], None
        lib.hx_layer_at.argtypes, lib.hx_layer_at.restype = [ptr, i64, i64, i64, i64, i64, i64, ptr, f64, ptr], None
        _TYPED[id(lib)] = True
    return lib


def kernel(sigma, radius):
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / (sigma * sigma) * x**2)
    return phi / phi.sum()


def gaussian(image, sigmas):
    """scipy.ndimage.gaussian_filter of a float32 array."""
    sigmas = (sigmas,) * image.ndim if np.isscalar(sigmas) else sigmas
    lib = library()
    source, target = np.ascontiguousarray(image, dtype=np.float32), np.empty(image.shape, dtype=np.float32)
    spare = np.empty_like(target) if source is image else source
    dims = np.array(image.shape, dtype=np.int64)
    for axis, sigma in enumerate(sigmas):
        radius = int(TRUNCATE * sigma + 0.5)
        weights = kernel(sigma, radius)
        plain = radius // PLAIN_BLOCK * PLAIN_BLOCK if radius >= FUSED_BELOW_RADIUS else 0
        lib.hx_gaussian_axis(
            source.ctypes.data,
            target.ctypes.data,
            dims.ctypes.data,
            image.ndim,
            axis,
            weights.ctypes.data,
            radius,
            plain,
        )
        source, target = target, spare if axis == 0 else source
    return source


def face_depth(smoothed, start):
    """Per column, the sub-layer depth of the steepest fall along float32 layers, which may be a reversed view."""
    layers, height, width = smoothed.shape
    assert smoothed.dtype == np.float32 and smoothed.strides[1:] == (4 * width, 4)
    depth = np.empty((height, width), dtype=np.float64)
    library().hx_face_depth(
        smoothed.ctypes.data, smoothed.strides[0] // 4, layers, height * width, start, depth.ctypes.data
    )
    return depth


def layer_at(block, depth, offset=0):
    """A layer at depth + offset per column of a uint8 block of any strides, linear between layers."""
    layers, height, width = block.shape
    assert block.dtype == np.uint8 and depth.dtype == np.float64 and depth.flags.c_contiguous
    out = np.empty((height, width), dtype=np.float32)
    library().hx_layer_at(
        block.ctypes.data, *block.strides, layers, height, width, depth.ctypes.data, float(offset), out.ctypes.data
    )
    return out
