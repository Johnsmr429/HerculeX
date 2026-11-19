"""The fetcher against a small server of its own."""

import http.server
import json
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
        if "missing" in self.path or self.path.rstrip("/").endswith("/9/9/9"):
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


def test_a_store_with_bricks_never_fetched_is_refused_by_the_read(tmp_path):
    from ink.reading import piece_reading

    store = BrickStore(tmp_path / "store", (8, 8, 8), (4, 4, 4))
    store.put((0, 0, 0), bytes(64))
    store.save()
    with pytest.raises(RuntimeError, match="7 bricks never fetched"):
        piece_reading.complete_store(store.folder)
    store.offsets[store.offsets == UNKNOWN] = -2
    store.save()
    assert piece_reading.complete_store(store.folder).grid == (2, 2, 2)


def test_bricks_come_through_the_native_client_as_through_the_python_one(server, monkeypatch):
    from scans.transfer import parallel_fetch

    ranged = [((0, 0, k), f"0/0/{k}", (16 * k, 16 * k + 15)) for k in range(60)] + [((9, 9, 9), "9/9/9", (0, 9))]
    whole = [((1, 2, k), f"1/2/{k}") for k in range(5)] + [((9, 9, 9), "9/9/9")]
    native = []
    real = parallel_fetch.fetch_bricks_native
    monkeypatch.setattr(parallel_fetch, "fetch_bricks_native", lambda *a: native.append(1) or real(*a))
    results = {}
    for fetcher in ("curl", "python"):
        monkeypatch.setattr(parallel_fetch, "FETCHER", fetcher)
        for name, items in (("ranged", ranged), ("whole", whole)):
            got = {}
            failures = parallel_fetch.fetch_all(server, items, lambda key, body, got=got: got.__setitem__(key, body), 8)
            results[fetcher, name] = (got, failures)
    assert native == [1, 1]
    assert results["curl", "ranged"] == results["python", "ranged"]
    assert results["curl", "whole"] == results["python", "whole"]
    got, failures = results["curl", "ranged"]
    assert not failures and got[(0, 0, 5)] == BODY[80:96] and got[(9, 9, 9)] is None
    assert results["curl", "whole"][0][(1, 2, 0)] == BODY
    mixed = [((0, 0, 0), "0/0/0"), ((0, 0, 1), "0/0/1", (0, 3))]
    assert parallel_fetch.bricks_of(mixed) is None and parallel_fetch.bricks_of([("k", "other/path")]) is None


def test_bricks_a_native_fetch_never_reached_fail_and_are_not_absent():
    import socket

    from scans.transfer import parallel_fetch

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    items = [((0, 0, k), f"0/0/{k}", (0, 15)) for k in range(300)]
    delivered = []
    failures = parallel_fetch.fetch_all(f"http://127.0.0.1:{port}/b", items, lambda k, b: delivered.append(k), 8)
    assert not delivered and sorted(key for key, _ in failures) == [key for key, *_ in items]


def stores(monkeypatch, failing=()):
    """fetch_bricks against a fake scan: which stores were made new, and a fetch that fails while `failing` holds."""
    made = []

    def scan_store(sample, volume, folder, level):
        made.append(folder.exists())
        return BrickStore(folder, (8, 8, 8), (4, 4, 4)), "http://nowhere"

    def ranges_of(store, jobs):
        names = {str(job[0]) for job in jobs}
        chosen = [i for i in ([0, 0, 0], [1, 1, 1]) if store.offsets[tuple(i)] == UNKNOWN and ("a" in names or i[0])]
        chosen = np.array(chosen, dtype=int).reshape(-1, 3)
        return chosen, np.zeros(len(chosen), dtype=int), np.full(len(chosen), 3)

    def fetch_all(url, items, deliver, report=None):
        if failing and failing[0]:
            return [(items[0][0], "down")]
        for key, *_ in items:
            deliver(key, bytes(64))
        return []

    monkeypatch.setattr(piece_images.sampler, "scan_store", scan_store)
    monkeypatch.setattr(piece_images, "ranges_of", ranges_of)
    monkeypatch.setattr(
        piece_images.sampler, "fetched_all", lambda url, items, deliver, report, fetch: fetch_all(url, items, deliver)
    )
    return made


def test_a_store_filled_for_part_of_its_plan_is_kept_for_that_part_only(tmp_path, monkeypatch):
    down = [True]
    made = stores(monkeypatch, down)
    store = tmp_path / "store"
    with pytest.raises(OSError, match="bricks failed"):
        piece_images.fetch_bricks("s", "v", [("a", None, None), ("b", None, None)], store)
    down[0] = False
    piece_images.fetch_bricks("s", "v", [("b", None, None)], store)
    assert json.loads((store / "planned.json").read_text()) == ["b"]
    piece_images.fetch_bricks("s", "v", [("a", None, None)], store)
    assert made == [False, True, False]


def test_a_store_of_another_scan_or_depth_is_fetched_again(tmp_path, monkeypatch):
    made = stores(monkeypatch)
    store, jobs = tmp_path / "store", [("a", None, None)]
    piece_images.fetch_bricks("s", "v", jobs, store)
    piece_images.fetch_bricks("s", "v", jobs, store)
    piece_images.fetch_bricks("s", "w", jobs, store)
    monkeypatch.setattr(sampler, "LAYERS", sampler.LAYERS + 8)
    piece_images.fetch_bricks("s", "w", jobs, store)
    assert made == [False, True, False, False]
