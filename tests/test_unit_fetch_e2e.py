"""The ink read's CT fetched unit by unit as the pieces stage writes them, on a fake scan and a simulated clock."""

import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import pytest

from runner import stage_queue
from sampling import piece_images
from storage.brick_store import UNKNOWN, BrickStore
from unwrapping.level import run_levels

ROWS = ("rows-0000-0012", "rows-0011-0023", "rows-0022-0030")


def finish_unit(unit, pieces, method=run_levels.METHOD):
    unit.mkdir(parents=True)
    for piece in pieces:
        (unit / f"piece-{piece:04d}").mkdir()
    record = {"method": method, "isometry": {}, "pieces": [{"piece": p, "written": p != 3} for p in pieces]}
    stage_queue.finish(unit, record)


@pytest.fixture
def scan(tmp_path, monkeypatch):
    from ink.reading import unit_fetch

    level, nodes, bricks = tmp_path / "level", tmp_path / "grown", tmp_path / "bricks"
    for name in ROWS:
        (nodes / "PHercT" / "v1" / "nodes" / name).mkdir(parents=True)
        stage_queue.finish(nodes / "PHercT" / "v1" / "nodes" / name, {"method": "n"})
    monkeypatch.setattr(unit_fetch, "LEVEL_ROUTE", level)
    monkeypatch.setattr(unit_fetch, "NODE_ROUTE", nodes)
    monkeypatch.setattr(unit_fetch, "ct_bricks", lambda sample, store: bricks / sample / store)
    monkeypatch.setattr(unit_fetch, "POLL_SECONDS", 60)
    return unit_fetch, level / "PHercT" / "v1"


class Clock:
    """Simulated time; each sleep may finish the next unit or write the layout."""

    def __init__(self, events):
        self.now, self.events = 1000.0, list(events)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.events:
            self.events.pop(0)()


class Fetches:
    """A stand-in for fetch_bricks that records the stores asked for, and can fail one or run out of room."""

    def __init__(self, failing=(), no_room=()):
        self.asked, self.failing, self.no_room = [], set(failing), set(no_room)

    def __call__(self, sample, volume, jobs, store, report=None, still=None):
        if still is not None and not still():
            return False
        self.asked.append((store.name, [job[0].name for job in jobs]))
        if store.name in self.failing:
            raise OSError("3 bricks failed")
        if store.name in self.no_room:
            raise sys.modules["ink.reading.unit_fetch"].NoRoom("1 GiB free")
        store.mkdir(parents=True, exist_ok=True)
        return True


def marked(path, later=5.0):
    """A file written now, dated `later` seconds after now, so it is newer than what was just written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    then = time.time() + later
    os.utime(path, (then, then))


def test_units_are_fetched_as_they_finish_until_every_unit_is_ready(scan):
    unit_fetch, base = scan
    fetches = Fetches()

    def rebuilt():
        shutil.rmtree(base / "pieces" / ROWS[2])
        finish_unit(base / "pieces" / ROWS[2], [5])

    faces = base / "faces" / "level-0" / ROWS[1]
    faces.mkdir(parents=True)
    (faces / "piece-0001.png").touch()
    finish_unit(base / "pieces" / ROWS[0], [1, 2, 3])
    clock = Clock(
        [
            lambda: None,
            lambda: finish_unit(base / "pieces" / ROWS[1], [1, 2]),
            lambda: finish_unit(base / "pieces" / ROWS[2], [5], method="older code"),
            rebuilt,
        ]
    )
    assert unit_fetch.follow("PHercT", "v1", fetch=fetches, clock=clock, sleep=clock.sleep) == set(ROWS)
    assert fetches.asked == [
        (ROWS[0], ["piece-0001", "piece-0002"]),
        (ROWS[1], ["piece-0001", "piece-0002"]),
        (ROWS[2], ["piece-0005"]),
    ]
    assert not clock.events


def test_the_read_takes_over_after_its_layout_or_start_even_for_a_fetch_started_again(scan):
    unit_fetch, base = scan
    finish_unit(base / "pieces" / ROWS[0], [1])
    fetches = Fetches()

    def laid_out():
        marked(base / "sheet" / "layout.json")
        finish_unit(base / "pieces" / ROWS[1], [1])
        marked(base / "sheet" / "layout.json", 10.0)

    clock = Clock([laid_out])
    assert unit_fetch.follow("PHercT", "v1", fetch=fetches, clock=clock, sleep=clock.sleep) == {ROWS[0]}
    again = Fetches()
    marked(base / "sheet" / unit_fetch.READING, 20.0)
    assert unit_fetch.follow("PHercT", "v1", fetch=again, clock=clock, sleep=clock.sleep) == set()
    assert again.asked == []


def test_an_old_layout_does_not_stop_a_fetch_for_new_pieces(scan):
    unit_fetch, base = scan
    marked(base / "sheet" / "layout.json", -3600)
    finish_unit(base / "pieces" / ROWS[0], [1])
    finish_unit(base / "pieces" / ROWS[1], [1])
    finish_unit(base / "pieces" / ROWS[2], [1])
    fetches = Fetches()
    clock = Clock([])
    assert unit_fetch.follow("PHercT", "v1", fetch=fetches, clock=clock, sleep=clock.sleep) == set(ROWS)


def test_a_failed_store_is_left_to_the_read_and_a_short_disc_stops_the_fetch(scan):
    unit_fetch, base = scan
    for row in ROWS:
        finish_unit(base / "pieces" / row, [1])
    clock = Clock([])
    failing = Fetches(failing={ROWS[0]})
    assert unit_fetch.follow("PHercT", "v1", fetch=failing, clock=clock, sleep=clock.sleep) == set(ROWS)
    short = Fetches(no_room={ROWS[1]})
    assert unit_fetch.follow("PHercT", "v1", fetch=short, clock=clock, sleep=clock.sleep) == {ROWS[0]}
    assert [name for name, _ in short.asked] == [ROWS[0], ROWS[1]]


def test_a_read_that_starts_while_a_store_waits_for_its_lock_keeps_it(scan):
    unit_fetch, base = scan
    for row in ROWS:
        finish_unit(base / "pieces" / row, [1])
    fetches = Fetches()

    def read_starts_first(sample, volume, jobs, store, report=None, still=None):
        if store.name == ROWS[1]:
            marked(base / "sheet" / unit_fetch.READING, 30.0)
        return fetches(sample, volume, jobs, store, report, still)

    clock = Clock([])
    assert unit_fetch.follow("PHercT", "v1", fetch=read_starts_first, clock=clock, sleep=clock.sleep) == {ROWS[0]}
    assert [name for name, _ in fetches.asked] == [ROWS[0]]


def test_the_fetch_stops_at_its_budget_and_when_nothing_finishes(scan, monkeypatch):
    unit_fetch, base = scan

    def fetch(sample, volume, jobs, store, report=None, still=None):
        store.mkdir(parents=True, exist_ok=True)
        (store / "bricks.bin").write_bytes(b"\0" * 2048)
        return True

    finish_unit(base / "pieces" / ROWS[0], [1])
    finish_unit(base / "pieces" / ROWS[1], [1])
    monkeypatch.setattr(unit_fetch, "BUDGET_GIB", 1024 / 2**30)
    clock = Clock([])
    assert unit_fetch.follow("PHercT", "v1", fetch=fetch, clock=clock, sleep=clock.sleep) == {ROWS[0]}
    monkeypatch.setattr(unit_fetch, "BUDGET_GIB", 100)
    monkeypatch.setattr(unit_fetch, "IDLE_HOURS", 0.05)
    clock = Clock([])
    assert unit_fetch.follow("PHercT", "v1", fetch=fetch, clock=clock, sleep=clock.sleep) == set(ROWS[:2])
    assert clock.now - 1000.0 >= 0.05 * 3600


def test_only_the_scan_the_pipeline_reads_is_fetched(scan, monkeypatch):
    unit_fetch, _ = scan
    for name in ROWS[:1]:
        other = unit_fetch.NODE_ROUTE / "PHercT" / "v2" / "nodes" / name
        other.mkdir(parents=True)
        stage_queue.finish(other, {"method": "n"})
    from scans.catalog import scan_choice

    monkeypatch.setattr(scan_choice, "chosen_scans", lambda samples: [{"sample": "PHercT", "volume": "v1"}])
    assert unit_fetch.scans_of("PHercT") == ["v1"]
    assert unit_fetch.scans_of("PHercT", "v2") == ["v2"]


def test_two_fetches_of_one_store_take_turns_and_a_rebuilt_unit_starts_it_again(tmp_path, monkeypatch):
    moved, made = [], []

    def scan_store(sample, volume, folder, level):
        made.append(folder.exists())
        return BrickStore(folder, (8, 8, 8), (4, 4, 4)), "http://nowhere"

    def ranges_of(store, jobs):
        chosen = np.argwhere(store.offsets == UNKNOWN)
        return chosen, np.zeros(len(chosen), dtype=int), np.full(len(chosen), 3)

    def fetch_all(url, items, deliver, report=None):
        time.sleep(0.2)
        for key, *_ in items:
            moved.append(key)
            deliver(key, bytes(64))
        return []

    monkeypatch.setattr(piece_images.sampler, "scan_store", scan_store)
    monkeypatch.setattr(piece_images, "ranges_of", ranges_of)
    monkeypatch.setattr(piece_images, "fetch_all", fetch_all)
    unit = tmp_path / "pieces" / "rows-0000-0012"
    finish_unit(unit, [1])
    jobs, store = [(unit / "piece-0001", None, None)], tmp_path / "store"
    both = [threading.Thread(target=piece_images.fetch_bricks, args=("s", "v", jobs, store)) for _ in range(2)]
    for thread in both:
        thread.start()
    for thread in both:
        thread.join()
    assert len(moved) == 8 and made == [False, True]
    record = json.loads((unit / "record.json").read_text()) | {"rebuilt": True}
    (unit / "record.json").write_text(json.dumps(record))
    piece_images.fetch_bricks("s", "v", jobs, store)
    assert len(moved) == 16 and made == [False, True, False]
