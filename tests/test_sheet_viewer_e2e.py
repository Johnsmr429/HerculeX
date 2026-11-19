"""End to end: the WebGL sheet viewer reads a tile store through the site server's tile map and decoded batches."""

import base64
import fcntl
import functools
import http.client
import io
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from storage.tile_store import TileStore, TileWriter
from website import sheet_tiles, site_server

TILE = 512
COLUMNS, ROWS = 4, 2
PIXEL_MM = 0.01
SHEET_VALUE = 120
CHROME = Path(os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))


def level_value(c, r):
    return 30 + 50 * c + 40 * r


def ink_value(c, r):
    return 200 if c < 2 else 100


def png(pixels, filters=None):
    """An 8-bit grey PNG, by PIL or with the given filter type on each row in turn."""
    if filters is None:
        out = io.BytesIO()
        Image.fromarray(pixels).save(out, format="PNG")
        return out.getvalue()
    h, w = pixels.shape
    rows, above = [], np.zeros(w, dtype=np.int32)
    for y in range(h):
        row, kind = pixels[y].astype(np.int32), filters[y % len(filters)]
        left = np.concatenate([[0], row[:-1]])
        corner = np.concatenate([[0], above[:-1]])
        if kind == 1:
            row = row - left
        elif kind == 2:
            row = row - above
        elif kind == 3:
            row = row - (left + above) // 2
        elif kind == 4:
            pa, pb, pc = np.abs(above - corner), np.abs(left - corner), np.abs(left + above - 2 * corner)
            row = row - np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, above, corner))
        rows.append(bytes([kind]) + (row % 256).astype(np.uint8).tobytes())
        above = pixels[y].astype(np.int32)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def write_layer(writer, name, value):
    """One layer of COLUMNS x ROWS flat tiles in three levels, each tile at value(c, r), a 0 tile left empty."""
    whole = np.zeros((ROWS * TILE, COLUMNS * TILE), dtype=np.uint8)
    for c in range(COLUMNS):
        for r in range(ROWS):
            whole[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = value(c, r)
    writer.layer(name, [(COLUMNS, ROWS), (2, 1), (1, 1)], raw=True, low=0.0, high=255.0)
    for level in range(3):
        step = 2**level
        image = whole[::step, ::step]
        for c in range(-(-image.shape[1] // TILE)):
            for r in range(-(-image.shape[0] // TILE)):
                cut = np.zeros((TILE, TILE), dtype=np.uint8)
                part = image[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
                cut[: part.shape[0], : part.shape[1]] = part
                if cut.any():
                    writer.add(name, level, c, r, png(cut, [0, 1, 2, 3, 4] if level == 0 else None))


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A synthetic sheet of 4 x 2 flat tiles in three levels, one cell left empty, served by the site server."""
    yield from served(
        tmp_path, monkeypatch, {"layer0": ("+0 µm", lambda c, r: 0 if (c, r) == (3, 1) else level_value(c, r))}
    )


@pytest.fixture
def ink_site(tmp_path, monkeypatch):
    """A synthetic sheet whose two left columns hold ink above the published floor and two right columns ink below it,
    over a flat scan."""
    yield from served(
        tmp_path,
        monkeypatch,
        {"sheet": ("brightest across the sheet", lambda c, r: SHEET_VALUE), "taught": ("ink", ink_value)},
        tint="purple",
    )


def served(tmp_path, monkeypatch, layers, tint=""):
    """The site server over one sheet page whose layers are each a name and a value per tile."""
    sheets = tmp_path / "sheets"
    folder = sheets / "sheet-test"
    with TileWriter(folder) as writer:
        for path, (_, value) in layers.items():
            write_layer(writer, path, value)
    shown = [
        {"path": path, "mm": PIXEL_MM, "levels": 3, "low": 0.0, "high": 255.0, "raw": True, "name": name}
        for path, (name, _) in layers.items()
    ]
    area = {"x": 0, "y": 0, "w": COLUMNS * TILE * PIXEL_MM, "h": ROWS * TILE * PIXEL_MM}
    sheet_tiles.page(folder, "test sheet", shown, area, "../index.html", "level/scroll/volume", tint=tint)
    layout = tmp_path / "level" / "scroll" / "volume" / "sheet" / "layout.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{"pieces": []}')
    monkeypatch.setattr(site_server, "LEVEL_ROUTE", tmp_path / "level")
    monkeypatch.setattr(site_server, "HAND_MARKS", tmp_path / "marks")
    monkeypatch.setattr(site_server, "SHEETS", sheets)
    monkeypatch.setitem(site_server.MOUNTS, "/sheets/", sheets)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(site_server.Site, directory=site_server.HERE))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def request(method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, path, None if body is None else json.dumps(body), headers)
            answer = connection.getresponse()
            return answer.status, answer.read()
        finally:
            connection.close()

    try:
        yield {
            "request": request,
            "folder": folder,
            "port": server.server_port,
            "log": tmp_path / "marks" / "level_route" / "scroll" / "volume" / "marks.jsonl",
        }
    finally:
        server.shutdown()
        server.server_close()


def test_the_tile_map_marks_stored_cells_and_batches_come_back_decoded(site):
    request, store = site["request"], TileStore(site["folder"])
    status, body = request("GET", "/tilemap?dir=/sheets/sheet-test/")
    size = struct.unpack_from("<I", body)[0]
    head = json.loads(body[4 : 4 + size])
    bits = np.unpackbits(np.frombuffer(body[4 + size :], dtype=np.uint8), bitorder="little")[: head["cells"]]
    assert status == 200 and head["layers"]["layer0"]["grid"] == [[4, 2], [2, 1], [1, 1]]
    assert (bits == (store.lengths > 0)).all() and not bits[COLUMNS + 3]

    asked = [["layer0", 0, c, r] for r in range(ROWS) for c in range(COLUMNS)] + [["layer0", 1, 1, 0]]
    status, body = request("POST", "/tiles", {"dir": "/sheets/sheet-test/", "stamp": head["stamp"], "tiles": asked})
    tiles = np.frombuffer(body, dtype=np.uint8).reshape(len(asked), TILE, TILE)
    assert status == 200
    for (_, level, c, r), tile in zip(asked, tiles):
        stored = store.image("layer0", level, c, r)
        assert (tile == (0 if stored is None else stored)).all()
    assert (tiles[0] == level_value(0, 0)).all() and not tiles[COLUMNS + 3].any()

    stale = {"dir": "/sheets/sheet-test/", "stamp": "0", "tiles": asked[:1]}
    assert request("POST", "/tiles", stale)[0] == 409
    assert request("POST", "/tiles", {**stale, "stamp": head["stamp"], "tiles": asked * 10})[0] == 400
    assert request("POST", "/tiles", {**stale, "dir": "/sheets/missing/"})[0] == 404
    assert request("GET", "/tilemap?dir=/sheets/missing/")[0] == 404


def test_the_c_unfilter_decodes_every_png_row_filter_as_pil_does():
    rng = np.random.default_rng(7)
    smooth = np.cumsum(rng.integers(-3, 4, (TILE, TILE)), axis=1).astype(np.uint8)
    for pixels in (rng.integers(0, 256, (TILE, TILE), dtype=np.uint8), smooth):
        for filters in ([0], [1], [2], [3], [4], [4, 3, 2, 1, 0], None):
            blob = png(pixels, filters)
            assert bytes(site_server.raw_tile(blob, TILE)) == pixels.tobytes()
    small = png(smooth[:100, :200])
    assert site_server.grey_png(small, TILE) is None
    decoded = np.frombuffer(bytes(site_server.raw_tile(small, TILE)), dtype=np.uint8).reshape(TILE, TILE)
    assert (decoded[:100, :200] == smooth[:100, :200]).all() and not decoded[100:].any()


class Chrome:
    """Headless Chrome driven over the DevTools pipe, with no packages beyond the standard library."""

    def __init__(self, width, height):
        commands, reply = os.pipe(), os.pipe()
        child_in, child_out = fcntl.fcntl(commands[0], fcntl.F_DUPFD, 10), fcntl.fcntl(reply[1], fcntl.F_DUPFD, 10)
        os.close(commands[0])
        os.close(reply[1])
        self.profile = Path(os.environ.get("TMPDIR", "/tmp")) / f"hx-viewer-test-{os.getpid()}"
        self.process = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                f'exec "$0" "$@" 3<&{child_in} 4>&{child_out}',
                str(CHROME),
                "--headless=new",
                "--remote-debugging-pipe",
                f"--user-data-dir={self.profile}",
                "--no-first-run",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=1",
                "about:blank",
            ],
            pass_fds=(child_in, child_out),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(child_in)
        os.close(child_out)
        self.write, self.read, self.buffer, self.id = commands[1], reply[0], b"", 0
        target = self.call("Target.createTarget", url="about:blank")["targetId"]
        self.session = self.call("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)

    def call(self, method, session=None, **params):
        self.id += 1
        message = {"id": self.id, "method": method, "params": params}
        if session or getattr(self, "session", None):
            message["sessionId"] = session or self.session
        os.write(self.write, json.dumps(message).encode() + b"\0")
        while True:
            while b"\0" not in self.buffer:
                self.buffer += os.read(self.read, 1 << 16)
            line, self.buffer = self.buffer.split(b"\0", 1)
            answer = json.loads(line)
            if answer.get("id") == self.id:
                if "error" in answer:
                    raise RuntimeError(answer["error"])
                return answer["result"]

    def evaluate(self, expression):
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get("value")

    def close(self):
        self.process.kill()
        self.process.wait()
        shutil.rmtree(self.profile, ignore_errors=True)


def opened(chrome, port):
    """The viewer on the test sheet at one voxel per pixel, its tiles loaded; returns its view."""
    chrome.call("Page.navigate", url=f"http://127.0.0.1:{port}/sheets/sheet-test/index.html")
    state = None
    for _ in range(200):
        time.sleep(0.05)
        raw = chrome.evaluate("window.viewer ? JSON.stringify(viewer.view()) : null")
        if raw and (state := json.loads(raw))["resident"] and not state["busy"] and not state["queued"]:
            break
    assert state and state["zoom"] > 0
    chrome.evaluate("viewer.voxel()")
    time.sleep(0.5)
    for _ in range(100):
        state = json.loads(chrome.evaluate("JSON.stringify(viewer.view())"))
        if not state["busy"] and not state["queued"]:
            break
        time.sleep(0.05)
    return state


def screen(chrome):
    shot = chrome.call("Page.captureScreenshot", format="png")["data"]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(shot))).convert("RGB")).astype(int)


def press(chrome, key):
    for kind in ("keyDown", "keyUp"):
        chrome.call("Input.dispatchKeyEvent", type=kind, key=key, code=f"Key{key.upper()}")


def mouse(chrome, kind, x, y, buttons=1):
    chrome.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", buttons=buttons, clickCount=1)


@pytest.mark.skipif(not CHROME.exists(), reason=f"needs Chrome at {CHROME} (set CHROME)")
def test_a_stroke_is_saved_drawn_and_erased(site):
    chrome = Chrome(1024, 600)
    try:
        opened(chrome, site["port"])
        for _ in range(50):
            if chrome.evaluate("!document.getElementById('marking').hidden"):
                break
            time.sleep(0.05)
        press(chrome, "g")
        mouse(chrome, "mousePressed", 200, 300)
        for x in range(210, 420, 10):
            mouse(chrome, "mouseMoved", x, 300)
        mouse(chrome, "mouseReleased", 420, 300, 0)
        for _ in range(50):
            if site["log"].exists() and site["log"].read_text().count("\n") == 1:
                break
            time.sleep(0.05)
        saved = json.loads(site["log"].read_text())
        assert saved["kind"] == "ink" and len(saved["points"]) >= 10 and saved["width_mm"] == 0.4
        time.sleep(0.2)
        red, beside = screen(chrome)[300, 310], screen(chrome)[400, 310]
        assert red[0] > red[1] + 60 and abs(beside[0] - beside[1]) <= 1
        press(chrome, "g")
        press(chrome, "x")
        mouse(chrome, "mousePressed", 310, 300)
        mouse(chrome, "mouseReleased", 310, 300, 0)
        for _ in range(50):
            if site["log"].read_text().count("\n") == 2:
                break
            time.sleep(0.05)
        assert json.loads(site["log"].read_text().splitlines()[1])["remove"] == saved["id"]
        time.sleep(0.2)
        erased = screen(chrome)[300, 310]
        assert abs(erased[0] - erased[1]) <= 1
    finally:
        chrome.close()


@pytest.mark.skipif(not CHROME.exists(), reason=f"needs Chrome at {CHROME} (set CHROME)")
def test_the_viewer_draws_every_tile_at_its_stored_value(site):
    chrome = Chrome(1024, 600)
    try:
        state = opened(chrome, site["port"])
        pixels = screen(chrome)[:, :, 0]
        zoom, ox, oy = state["zoom"], state["ox"], state["oy"]
        checked = 0
        for c in range(COLUMNS):
            for r in range(ROWS):
                x = round(((c + 0.5) * TILE * PIXEL_MM - ox) * zoom)
                y = round(((r + 0.5) * TILE * PIXEL_MM - oy) * zoom)
                if 60 <= y < pixels.shape[0] - 30 and 0 <= x < pixels.shape[1]:
                    expected = 0 if (c, r) == (3, 1) else level_value(c, r)
                    assert abs(int(pixels[y, x]) - expected) <= 1, (c, r, int(pixels[y, x]), expected)
                    checked += 1
        assert checked >= 2
        assert chrome.evaluate("document.getElementById('label').textContent").endswith("test sheet · +0 µm")
    finally:
        chrome.close()


@pytest.mark.skipif(not CHROME.exists(), reason=f"needs Chrome at {CHROME} (set CHROME)")
def test_the_ink_and_scan_sliders_redraw_the_purple_ink(ink_site):
    chrome = Chrome(1024, 600)
    try:
        state = opened(chrome, ink_site["port"])
        zoom, ox, oy = state["zoom"], state["ox"], state["oy"]
        y = min(540, max(80, round((ROWS * TILE * PIXEL_MM / 2 - oy) * zoom)))
        columns = {}
        for c in range(COLUMNS):
            x = round(((c + 0.5) * TILE * PIXEL_MM - ox) * zoom)
            if 20 <= x < 1004:
                columns.setdefault(ink_value(c, 0) > sheet_tiles.WEAKEST_INK, x)
        assert set(columns) == {True, False}
        chrome.evaluate("window.failures = []; addEventListener('error', (e) => failures.push(e.message))")

        def slide(name, value):
            chrome.evaluate(
                f"(f => {{ f.value = {value}; f.dispatchEvent(new Event('input')); }})(document.getElementById('{name}'))"
            )
            time.sleep(0.3)
            return screen(chrome)

        before = screen(chrome)
        ink, under = before[y, columns[True]], before[y, columns[False]]
        assert ink[2] > ink[1] + 60, ink
        faded = slide("floor", 220)[y, columns[True]]
        assert abs(faded[2] - faded[1]) <= 3 and faded[1] < ink[1], faded
        brighter = slide("scan", 100)[y, columns[False]]
        assert brighter[1] > under[1] + 50, (under, brighter)
        assert chrome.evaluate("failures") == []
        look = chrome.evaluate("JSON.parse(localStorage.inkLook)")
        assert (look["floor"], look["scan"]) == (220, 1)
    finally:
        chrome.close()
