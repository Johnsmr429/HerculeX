"""The labeling and blind review workflows through the local site server."""

import functools
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from website import site_server


@pytest.fixture
def request_site(tmp_path, monkeypatch):
    for name in ("SHEETS", "LEVEL_ROUTE", "HAND_MARKS", "CIRCLE_REVIEW"):
        monkeypatch.setattr(site_server, name, tmp_path / name.lower())
    monkeypatch.setitem(site_server.MOUNTS, "/sheets/", site_server.SHEETS)
    handler = functools.partial(site_server.Site, directory=site_server.HERE)

    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            answer = connection.getresponse()
            return answer.status, answer.read()
        finally:
            connection.close()

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield request
        finally:
            server.shutdown()


def test_retired_blind_review_is_unavailable(request_site):
    for path in ("/website/blind_pairs.html", "/blind_pairs/", "/blind_pairs/pairs.json", "/blind_pairs/key.json"):
        assert request_site("GET", path)[0] == 404


def test_archived_circle_links_open_the_depth_images(request_site):
    folder = site_server.CIRCLE_REVIEW / "depths" / "1b68c2ba-new"
    folder.mkdir(parents=True)
    (folder / "3-depth-+0.png").write_bytes(b"depth image")
    (folder / "spot.json").write_text("metadata")
    archived = "/submission/PHerc0211/work/depths/1b68c2ba-new/"

    assert request_site("GET", archived + "3-depth-+0.png") == (200, b"depth image")
    assert request_site("GET", archived + "spot.json")[0] == 404


def test_labels_keep_their_layout_and_stay_in_their_own_route(request_site):
    scans = {
        "level/scroll/volume": site_server.LEVEL_ROUTE / "scroll/volume/sheet/layout.json",
        "part/scroll-rows-student": site_server.SHEETS / "scroll-rows-student/placed.json",
    }
    for layout in scans.values():
        layout.parent.mkdir(parents=True)
        layout.write_text('{"pieces": []}')

    def post(body):
        return request_site("POST", "/marks", json.dumps(body), {"Content-Type": "application/json"})

    stroke = {"width_mm": 0.4, "layer": "scan", "points": [[4.25, 8.5], [5.0, 9.0]]}
    for scan, kind in zip(scans, ("ink", "blank", "spot", "ink")):
        status, body = post({**stroke, "kind": kind, "scan": scan})
        assert status == 200
        saved = json.loads(body)
        assert saved["kind"] == kind and saved["points"] == stroke["points"]
        mark_folder, _ = site_server.folders_of(scan)
        assert (mark_folder / "layouts" / f"{saved['layout']}.json").read_bytes() == scans[scan].read_bytes()
        assert post({**stroke, "kind": kind, "scan": scan, "points": [[float("nan"), 0]]})[0] == 400
        assert json.loads(request_site("GET", "/marks?scan=" + scan)[1]) == [saved]
        assert post({"scan": scan, "remove": saved["id"]})[0] == 200
        assert request_site("GET", "/marks?scan=" + scan) == (200, b"[]")
    taught = site_server.LEVEL_ROUTE / "scroll/volume/sheet/placed-taught.json"
    taught.write_text('{"pieces": [{"flip_cols": true}]}')
    saved = json.loads(post({**stroke, "kind": "ink", "layer": "taught ink", "scan": "level/scroll/volume"})[1])
    kept = site_server.folders_of("level/scroll/volume")[0] / "layouts" / f"{saved['layout']}.json"
    assert kept.read_bytes() == taught.read_bytes()


def test_the_start_line_is_counted_when_asked(request_site, monkeypatch):
    counts = iter(
        [
            {"scrolls": 2, "read_scrolls": 1, "laid_cm2": 9.0, "ink_cm2": 4.0},
            {"scrolls": 2, "read_scrolls": 2, "laid_cm2": 9.0, "ink_cm2": 9.0},
        ]
    )
    monkeypatch.setattr(site_server, "start_status", lambda: next(counts))
    assert json.loads(request_site("GET", "/status")[1])["ink_cm2"] == 4.0
    assert json.loads(request_site("GET", "/status")[1])["read_scrolls"] == 2

    def unmounted():
        raise OSError("HerculeX data folder is unavailable")

    monkeypatch.setattr(site_server, "start_status", unmounted)
    assert request_site("GET", "/status")[0] == 503
