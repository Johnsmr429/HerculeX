"""The disc report's reclaimable folders on a data root of its own: each kind found, and what is still in use left."""

import json
import os
import time

import pytest

from runner import disk, stage_queue


@pytest.fixture
def roots(tmp_path, monkeypatch):
    names = {
        "NODE_ROUTE": tmp_path / "nodes",
        "LEVEL_ROUTE": tmp_path / "level",
        "LIBRARY": tmp_path / "library",
        "DETECTION_WORK": tmp_path / "detection-work",
        "SCRATCH": tmp_path / "scratch",
    }
    for name, path in names.items():
        path.mkdir()
        monkeypatch.setattr(disk, name, path)
    monkeypatch.setattr(disk, "FACES", "level-0")
    return names


def aged(path, days=2):
    then = time.time() - days * disk.DAY
    os.utime(path, (then, then))
    return path


def test_each_kind_of_leftover_is_found_and_work_in_use_is_left(roots):
    nodes = roots["NODE_ROUTE"] / "S" / "v" / "nodes"
    killed = nodes / "rows-0000-0012.building"
    killed.mkdir(parents=True)
    aged(killed, 1)
    claimed = nodes / "rows-0011-0023.building"
    claimed.mkdir()
    aged(claimed, 1)
    (nodes / ".rows-0011-0023.claim").touch()
    done = nodes / "rows-0022-0030"
    done.mkdir()
    stage_queue.finish(done, {"method": "m"})

    pieces = roots["LEVEL_ROUTE"] / "S" / "v" / "pieces" / "rows-0000-0012"
    faces = roots["LEVEL_ROUTE"] / "S" / "v" / "faces" / "level-0" / "rows-0000-0012"
    faces.mkdir(parents=True)
    read, unread = (roots["LIBRARY"] / "cache" / "bricks" / "S" / name for name in ("read", "unread"))
    for store, names in ((read, ["piece-0001", "piece-0002"]), (unread, ["piece-0001", "piece-0003"])):
        store.mkdir(parents=True)
        (store / "planned.json").write_text(json.dumps([str(pieces / n) for n in names]))
    for name in ("piece-0001", "piece-0002"):
        (faces / f"{name}.png").touch()

    work = roots["DETECTION_WORK"] / "S" / "v"
    (work / "L1-0022-0030").mkdir(parents=True)
    (work / "L1-0000-0012").mkdir()

    stale, fresh, cache = (roots["SCRATCH"] / n for n in ("old-run", "new-run", "pytest-of-someone"))
    for folder in (stale, fresh, cache):
        folder.mkdir()
    aged(stale)
    aged(cache)

    found = {(kind, path.name) for kind, path, _ in disk.reclaimable()}
    assert found == {
        ("unfinished unit", "rows-0000-0012.building"),
        ("read CT store", "read"),
        ("detection store", "L1-0022-0030"),
        ("old scratch", "old-run"),
    }
