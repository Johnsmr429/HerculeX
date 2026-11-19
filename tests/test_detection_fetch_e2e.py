"""Sheet detection's fetch loop on a fake scan: every brick is fetched once, though chunks share a brick row."""

import numpy as np
import pytest

from storage.brick_store import ABSENT, BrickStore
from unwrapping.detection import detect_nodes as detector

SHAPE, BRICK = (128 * 23, 256, 256), (128, 128, 128)


def brick_bytes(index):
    """The brick a fake server holds at an index; one in every corner (z, 0, 0) is masked out."""
    if index[1] == index[2] == 0 and index[0] % 2:
        return None
    return np.full(BRICK, (31 * index[0] + 7 * index[1] + index[2]) % 251, np.uint8).tobytes()


@pytest.fixture
def scan(tmp_path, monkeypatch):
    fetched, seen = [], {}

    def fetch_level(path, level, out, *, z_bricks, **_):
        store = BrickStore(out, SHAPE, BRICK, source=f"{path}/{level}")
        for index in map(tuple, store.missing()):
            if z_bricks[0] <= index[0] < z_bricks[1]:
                fetched.append(index)
                store.put(index, brick_bytes(index))
        store.save()
        return store

    def detect_store(level, out, record, cell_planes=None):
        store = BrickStore(level)
        first, last = record["rows"]
        for index in np.ndindex(last - first, *store.grid[1:]):
            index = (first + index[0], *index[1:])
            brick = store.read_brick(index)
            assert (brick is None and brick_bytes(index) is None) or brick.tobytes() == brick_bytes(index)
            assert store.offsets[index] != ABSENT or brick_bytes(index) is None
        seen[out.name] = [p.name for p in level.parent.iterdir()]
        out.mkdir(parents=True)
        return record | {"nodes": 0, "seconds": 0}

    row = {
        "sample": "PHercTest",
        "volume": "v1",
        "voxel_um": 9.0,
        "products": {"ome-zarr": "p"},
        "roots": {"ome-zarr": "r"},
    }
    planned = [
        {
            "row": row,
            "level": 1,
            "shape": SHAPE,
            "rows": (first, last),
            "planes": (low, high),
            "out": tmp_path / "nodes" / f"rows-{first:04d}-{last:04d}",
        }
        for first, last, low, high in detector.chunks(SHAPE[0])
    ]
    monkeypatch.setattr(detector, "DETECTION_WORK", tmp_path / "work" / "detection-work")
    monkeypatch.setattr(detector, "NODE_ROUTE", tmp_path)
    monkeypatch.setattr(detector, "FLOOR_GIB", 0)
    monkeypatch.setattr(detector, "fetch_level", fetch_level)
    monkeypatch.setattr(detector, "detect_store", detect_store)
    monkeypatch.setattr(detector, "jobs", lambda *_: iter(planned))
    return planned, fetched, seen


def test_every_brick_is_fetched_once_and_each_store_is_cleared(scan):
    planned, fetched, seen = scan
    assert len(planned) == 2 and planned[0]["rows"][1] - 1 == planned[1]["rows"][0]
    assert detector.one_pass({"PHercTest"}, False) == 0
    assert sorted(fetched) == sorted(set(fetched)) == [tuple(i) for i in np.ndindex(23, 2, 2)]
    assert all(job["out"].is_dir() for job in planned)
    assert not any(detector.DETECTION_WORK.rglob("bricks.bin"))
    first, second = (detector.work_folder(job).name for job in planned)
    assert first in seen[planned[1]["out"].name] or second in seen[planned[1]["out"].name]


def test_a_chunk_after_a_failed_one_still_fetches_its_whole_rows(scan, monkeypatch):
    planned, fetched, _ = scan
    original = detector.detect_chunk
    monkeypatch.setattr(detector, "detect_chunk", lambda job, f: 1 if job is planned[0] else original(job, f))
    assert detector.one_pass({"PHercTest"}, False) == 1
    assert detector.work_folder(planned[0]).is_dir() and not detector.work_folder(planned[1]).is_dir()
    assert sorted(set(fetched)) == [tuple(i) for i in np.ndindex(23, 2, 2)]
