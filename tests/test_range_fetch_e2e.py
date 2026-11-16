"""The fetcher against a small server of its own."""

import http.server
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sampling import column_sampling as sampler
from sampling import piece_images
from scans.transfer.parallel_fetch import fetch_all
from storage.brick_store import UNKNOWN, BrickStore

BODY = bytes(range(256)) * 16


class Objects(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if "missing" in self.path:
            self.send_error(404)
            return
        wanted = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        body, status = (
            (BODY[int(wanted[1]) : int(wanted[2]) + 1], 206) if wanted and "plain" not in self.path else (BODY, 200)
        )
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


@pytest.fixture
def server():
    running = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Objects)
    running.protocol_version = "HTTP/1.1"
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{running.server_address[1]}/bucket"
    running.shutdown()


def test_a_range_a_whole_object_and_a_missing_one(server):
    got = {}
    items = [("span", "a/0", (16, 47)), ("whole", "a/1"), ("gone", "missing/2"), ("plain", "plain/3", (16, 47))]
    assert fetch_all(server, items, got.__setitem__, connections=2) == []
    assert got["span"] == BODY[16:48] and got["whole"] == BODY and got["gone"] is None
    assert got["plain"] == BODY


def test_a_fetch_takes_every_brick_its_stacks_read_and_no_more(monkeypatch):
    def marks(ranges, store, xyz, normal, stride):
        ranges[0, 1, 0] = 1, 2
        ranges[1, 1, 1] = 0, 0

    monkeypatch.setattr(piece_images.sampler, "load_surface", lambda folder: (None, {"stride": 1}))
    monkeypatch.setattr(piece_images, "surface_of", lambda xyz, meta, axis: (None, None))
    monkeypatch.setattr(piece_images, "mark_ranges", marks)
    store = SimpleNamespace(grid=(2, 2, 2), brick=(4, 4, 4), offsets=np.full((2, 2, 2), UNKNOWN))
    store.offsets[1, 1, 1] = 0
    indices, first, last = piece_images.ranges_of(store, [("piece", None, None)])
    assert indices.tolist() == [[0, 1, 0]] and first.tolist() == [1] and last.tolist() == [2]


def test_a_store_asked_for_pieces_it_was_not_planned_for_starts_again(monkeypatch, tmp_path):
    made = []

    def scan_store(sample, volume, folder, level):
        made.append(folder.exists())
        return BrickStore(folder, (8, 8, 8), (4, 4, 4)), "http://nowhere"

    monkeypatch.setattr(piece_images.sampler, "scan_store", scan_store)
    monkeypatch.setattr(
        piece_images,
        "ranges_of",
        lambda store, jobs: (np.zeros((0, 3), dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)),
    )
    monkeypatch.setattr(piece_images, "fetch_all", lambda *a, **k: [])
    folder = tmp_path / "store"
    piece_images.fetch_bricks("s", "v", [("a", None, None), ("b", None, None)], folder)
    piece_images.fetch_bricks("s", "v", [("a", None, None)], folder)
    piece_images.fetch_bricks("s", "v", [("c", None, None)], folder)
    assert made == [False, True, False]


def test_a_fetch_the_read_waits_on_goes_down_to_the_floor_and_one_ahead_does_not(monkeypatch, tmp_path):
    store = SimpleNamespace(save=lambda: None, folder=tmp_path)
    monkeypatch.setattr(
        sampler.shutil,
        "disk_usage",
        lambda folder: SimpleNamespace(free=(sampler.FLOOR_BYTES + sampler.ROOM_BYTES) // 2),
    )

    def waited(seconds):
        raise TimeoutError

    monkeypatch.setattr(sampler.time, "sleep", waited)
    with pytest.raises(TimeoutError):
        sampler.keep_room(store)
    sampler.reader_waiting.set()
    try:
        sampler.keep_room(store)
    finally:
        sampler.reader_waiting.clear()


def test_bricks_that_failed_are_tried_once_more(monkeypatch):
    asked = []

    def fetch(base, items, deliver, report=None):
        asked.append([item[0] for item in items])
        return [(items[0][0], "giving up")] if len(asked) == 1 else []

    monkeypatch.setattr(sampler, "RETRY_PAUSE_SECONDS", 0)
    assert sampler.fetched_all("http://x", [("a", "a"), ("b", "b")], None, fetch=fetch) == []
    assert asked == [["a", "b"], ["a"]]
