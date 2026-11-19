"""A changed spot surface can sample new planes of an already cached CT brick."""

import numpy as np

from spots import surfaces
from storage.brick_store import BrickStore


def test_cached_bricks_are_complete_across_surface_changes_and_legacy_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(surfaces, "work_of", lambda _: tmp_path)
    folder = tmp_path / "ct"
    truth = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    legacy = np.zeros_like(truth)
    legacy[1] = truth[1]
    store = BrickStore(folder, shape=(4, 4, 8), brick=(4, 4, 4))
    store.put((0, 0, 0), legacy.tobytes())
    store.save()
    monkeypatch.setattr(surfaces.sampler, "scan_store", lambda *args: (BrickStore(folder), "synthetic"))
    planes = [1, 1]

    def touched(ranges, *_):
        ranges[0, 0, 0] = planes

    monkeypatch.setattr(surfaces, "mark_ranges", touched)
    requested = []

    def fetch(url, items, deliver):
        assert url == "synthetic"
        for item in items:
            assert len(item) == 2, "a reusable CT cache must request the entire brick"
            index, _ = item
            requested.append(index)
            deliver(index, truth.tobytes())
        return []

    monkeypatch.setattr(surfaces.sampler, "fetched_all", fetch)
    meshes = [(None, None, {"stride": 1})]
    first = surfaces.ct_store("PHercTest", "scan", "spot", meshes)
    np.testing.assert_array_equal(first.read_brick((0, 0, 0)), truth)
    assert requested == [(0, 0, 0)]
    planes[:] = [3, 3]
    second = surfaces.ct_store("PHercTest", "scan", "spot", meshes)
    np.testing.assert_array_equal(second.read((3, 4), (0, 4), (0, 4)), truth[3:4])
    assert requested == [(0, 0, 0)], "the second surface should reuse its complete brick"
    complete = np.load(folder / "complete.npy")
    assert complete.tolist() == [[[True, False]]]
