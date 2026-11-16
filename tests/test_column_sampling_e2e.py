"""Layer stacks read across brick edges and holes match a direct read of the volume."""

import numpy as np

from sampling.kernel_bindings import library
from storage.brick_store import BrickStore

BRICK = 16
GRID = 3


def sampled(tmp_path, volume, missing, position, normal, layers=9):
    store = BrickStore(tmp_path / "bricks", volume.shape, (BRICK,) * 3)
    for index in np.ndindex(GRID, GRID, GRID):
        z, y, x = (i * BRICK for i in index)
        store.put(
            index,
            None
            if index == missing
            else np.ascontiguousarray(volume[z : z + BRICK, y : y + BRICK, x : x + BRICK]).tobytes(),
        )
    store.save()
    data, offsets = store._open(), store.offsets.copy()
    geometry = np.array([*store.grid, *store.brick], dtype=np.int32)
    out = np.empty((layers, position.shape[1]), dtype=np.uint8)
    library().hx_sample_columns(
        data.ctypes.data,
        offsets.ctypes.data,
        geometry.ctypes.data,
        position.ctypes.data,
        normal.ctypes.data,
        position.shape[1],
        layers,
        -(layers // 2),
        out.ctypes.data,
    )
    return out


def truth(volume, missing, position, normal, layers=9):
    side = volume.shape[0]
    known = volume.astype(np.float64)
    z, y, x = (i * BRICK for i in missing)
    known[z : z + BRICK, y : y + BRICK, x : x + BRICK] = 0
    out = np.zeros((layers, position.shape[1]))
    for layer in range(layers):
        p = position + (layer - layers // 2) * normal
        base = np.floor(p).astype(int)
        f = p - base
        for a, b, e in np.ndindex(2, 2, 2):
            ix, iy, iz = base[0] + e, base[1] + b, base[2] + a
            inside = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < side) & (iy < side) & (iz < side)
            weight = (f[2] if a else 1 - f[2]) * (f[1] if b else 1 - f[1]) * (f[0] if e else 1 - f[0])
            out[layer] += np.where(
                inside, weight * known[iz.clip(0, side - 1), iy.clip(0, side - 1), ix.clip(0, side - 1)], 0
            )
    return out


def test_columns_read_across_bricks_edges_and_holes(tmp_path):
    rng = np.random.default_rng(5)
    side = BRICK * GRID
    volume = rng.integers(1, 255, size=(side, side, side), dtype=np.uint8)
    position = rng.uniform(1, side - 1, size=(3, 4000)).astype(np.float32)
    position[:, :500] = rng.integers(1, GRID, size=(3, 500)) * BRICK - rng.uniform(0, 1, size=(3, 500))
    normal = rng.normal(size=(3, 4000)).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=0)
    missing = (1, 1, 1)
    read = sampled(tmp_path, volume, missing, position, normal)
    assert (
        np.abs(
            read.astype(np.float64) - truth(volume, missing, position.astype(np.float64), normal.astype(np.float64))
        ).max()
        <= 1
    )
    position[0, 7] = -1
    assert not sampled(tmp_path / "second", volume, missing, position, normal)[:, 7].any()
