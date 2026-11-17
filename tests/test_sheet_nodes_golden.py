"""The GPU sheet-node kernel (Metal, or CUDA off macOS) must find the same nodes as the C kernel on a spiral of known sheets."""

import numpy as np

from storage.brick_store import BrickStore
from storage.node_store import read_cells, read_nodes
from unwrapping.detection.detect_nodes import detect_store

SIZE, HEIGHT, PITCH, CENTRE = 512, 128, 9.0, 255.5


def spiral_volume(rng, chirality=1):
    """Archimedean sheets r = PITCH * (t + phi / 2 pi), grey 120 on 50."""
    y, x = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    r = np.hypot(x - CENTRE, y - CENTRE)
    turns = r / PITCH - np.arctan2(y - CENTRE, x - CENTRE) * chirality / (2 * np.pi)
    plane = 50 + 70 * np.exp(-0.5 * ((np.abs(turns - np.round(turns)) * PITCH) / 1.4) ** 2)
    plane[(r < 60) | (r > 240)] = 0
    volume = np.repeat(plane[None], HEIGHT, axis=0) + rng.normal(0, 3, (HEIGHT, SIZE, SIZE)) * (plane > 0)
    return np.clip(volume, 0, 255).astype(np.uint8)


def write_store(volume, folder):
    store = BrickStore(folder, volume.shape, (128, 128, 128), source="synthetic")
    for z, y, x in np.ndindex(*store.offsets.shape):
        block = np.zeros((128, 128, 128), dtype=np.uint8)
        part = volume[z * 128 : (z + 1) * 128, y * 128 : (y + 1) * 128, x * 128 : (x + 1) * 128]
        block[: part.shape[0], : part.shape[1], : part.shape[2]] = part
        store.put((z, y, x), block.tobytes())
    store.save()


def test_gpu_matches_c(tmp_path):
    write_store(spiral_volume(np.random.default_rng(3)), tmp_path / "level")
    detect_store(tmp_path / "level", tmp_path / "gpu", {}, device="gpu")
    detect_store(tmp_path / "level", tmp_path / "cpu", {}, device="cpu")
    gpu, cpu = read_nodes(tmp_path / "gpu"), read_nodes(tmp_path / "cpu")
    cells_gpu, cells_cpu = read_cells(tmp_path / "gpu"), read_cells(tmp_path / "cpu")
    assert abs(len(gpu) - len(cpu)) <= 0.001 * len(cpu)
    shared = np.intersect1d(cells_gpu, cells_cpu)
    assert len(shared) >= 0.999 * len(np.unique(cells_cpu))
    one_gpu = np.unique(cells_gpu, return_index=True)[1]
    one_cpu = np.unique(cells_cpu, return_index=True)[1]
    gap = np.abs(gpu["x"][one_gpu][: len(shared)] - cpu["x"][one_cpu][: len(shared)])
    assert np.percentile(gap, 50) < 0.01
