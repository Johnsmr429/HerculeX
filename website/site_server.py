"""The local site at http://localhost:8765/sheets/index.html and the hand-mark store."""

import ctypes
import functools
import hashlib
import http.server
import io
import json
import math
import os
import re
import struct
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from runner.compiled_kernels import load
from storage.data_roots import (
    BROWSER_ASSETS,
    CIRCLE_REVIEW,
    HAND_MARKS,
    HERE,
    LEVEL_ROUTE,
    SHEETS,
    SITE_ATLAS,
    stage_folder,
)
from storage.tile_store import INDEX, TileStore
from website.site_pages import start_status

PORT = 8765
NAME = re.compile(r"[A-Za-z0-9._-]+")
RANGE = re.compile(r"bytes=(\d+)-(\d+)")
CIRCLE_IMAGE = re.compile(r"paper/[\w-]+\.jpg|depths/[\w-]+/[\w+-]+\.png")
KINDS = ("ink", "blank", "spot")
MOST_POINTS = 20000
MOST_TILES = 64
PNG = b"\x89PNG\r\n\x1a\n"
TILE_THREADS = max(2, min(6, (os.cpu_count() or 4) // 2))
_writing = threading.Lock()
_stores, _stores_lock = {}, threading.Lock()
_decoders = ThreadPoolExecutor(TILE_THREADS, thread_name_prefix="tile-decode")
_building = threading.Lock()


@functools.cache
def unfilter():
    """The C row unfilter of 8-bit grey PNG, built once."""
    with _building:
        calls = load("tile-decode", [Path(__file__).with_name("tile_decode.c")])
    call = calls.hx_png_unfilter
    call.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    call.restype = ctypes.c_int
    return call


def grey_png(blob, size):
    """An 8-bit grey, non-interlaced size x size PNG decoded without PIL, None for any other PNG."""
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", blob[16:29])
    if (width, height, depth, colour, interlace) != (size, size, 8, 0, 0):
        return None
    parts, at = [], 8
    while at + 8 <= len(blob):
        length, kind = struct.unpack_from(">I4s", blob, at)
        if kind == b"IDAT":
            parts.append(blob[at + 8 : at + 8 + length])
        elif kind == b"IEND":
            break
        at += 12 + length
    filtered = zlib.decompress(b"".join(parts))
    if len(filtered) != size * (size + 1):
        return None
    out = bytearray(size * size)
    if unfilter()(filtered, ctypes.addressof(ctypes.c_char.from_buffer(out)), size, size):
        return None
    return out


def store_at(folder):
    """The tile store of a sheet folder with its index's stamp, reloaded when publication replaces the index."""
    stamp = str((folder / INDEX).stat().st_mtime_ns)
    with _stores_lock:
        held = _stores.get(folder)
        if held is None or held[0] != stamp:
            store = TileStore(folder)
            present = np.packbits(store.lengths > 0, bitorder="little").tobytes()
            held = _stores[folder] = (stamp, store, present)
    return held


def tile_map(folder):
    """A sheet's layers and one bit per tile cell, set where the store holds a tile: the viewer's whole index."""
    stamp, store, present = store_at(folder)
    header = json.dumps({"stamp": stamp, "cells": int(store.lengths.size), "layers": store.layers}).encode()
    return struct.pack("<I", len(header)) + header + present


def raw_tile(blob, size):
    """A stored tile decoded to size x size grey bytes, row by row."""
    if blob[:8] == PNG and (found := grey_png(blob, size)) is not None:
        return found
    image = Image.open(io.BytesIO(blob))
    if image.mode != "L":
        image = image.convert("L")
    if image.size != (size, size):
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(image.crop((0, 0, min(size, image.width), min(size, image.height))), (0, 0))
        image = canvas
    return image.tobytes()


def tile_bytes(folder, stamp, tiles, size):
    """Each asked tile (layer, level, column, row) decoded, in order; empty cells are zeros."""
    held, store, _ = store_at(folder)
    if held != stamp:
        raise LookupError("the tile store was republished")
    places = []
    for layer, level, c, r in tiles:
        if layer not in store.layers:
            raise ValueError(f"no layer {layer}")
        places.append(store.place(layer, int(level), int(c), int(r)))
    with open(store.pack, "rb") as pack:
        blobs = [None if p is None else os.pread(pack.fileno(), p[1], p[0]) for p in places]
    stored = [blob for blob in blobs if blob is not None]
    decoded, empty = iter(_decoders.map(raw_tile, stored, [size] * len(stored))), bytes(size * size)
    return [empty if blob is None else next(decoded) for blob in blobs]


def folders_of(scan):
    """A scan's marks folder in the hand marks and the folder of the layout they lie on."""
    parts = str(scan).split("/")
    if not 2 <= len(parts) <= 3 or not all(NAME.fullmatch(p) and p not in (".", "..") for p in parts):
        raise ValueError(f"no such scan: {scan}")
    if len(parts) == 2 and parts[0] == "part":
        marks, folder, layout = HAND_MARKS / "parts" / parts[1], SHEETS / parts[1], "placed.json"
    elif len(parts) == 3 and parts[0] == "level":
        marks, folder, layout = (
            HAND_MARKS / "level_route" / parts[1] / parts[2],
            stage_folder(LEVEL_ROUTE / parts[1] / parts[2], "sheet"),
            "layout.json",
        )
    else:
        raise ValueError(f"no such scan: {scan}")
    if not (folder / layout).exists():
        raise ValueError(f"no such scan: {scan}")
    return marks, folder


def layout_digest(marks, folder, layer=""):
    """The digest of the placement a stroke lies on, kept beside the marks; taught layers lie on their own."""
    names = (
        ("placed-taught.json", "placed.json", "layout.json") if "taught" in layer else ("placed.json", "layout.json")
    )
    shown = next(folder / name for name in names if (folder / name).exists())
    data = shown.read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:12]
    kept = marks / "layouts" / f"{digest}.json"
    if not kept.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        pending = kept.with_suffix(".json.part")
        pending.write_bytes(data)
        pending.replace(kept)
    return digest


def standing(log):
    """Every mark of a log not removed since, oldest first."""
    marks = {}
    for line in log.read_text().splitlines() if log.exists() else ():
        entry = json.loads(line)
        if "remove" in entry:
            marks.pop(entry["remove"], None)
        else:
            marks[entry["id"]] = entry
    return list(marks.values())


def marks_of(scan):
    """Every mark of a scan still standing, oldest first."""
    log = folders_of(scan)[0] / "marks.jsonl"
    with _writing:
        return standing(log)


def stroke_of(body):
    """A checked stroke in sheet millimetres."""
    points = body["points"]
    width = float(body["width_mm"])
    if body["kind"] not in KINDS or not 0.01 <= width <= 50 or not 1 <= len(points) <= MOST_POINTS:
        raise ValueError("not a stroke")
    points = [[float(x), float(y)] for x, y in points]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError("non-finite stroke coordinates")
    return {
        "kind": body["kind"],
        "width_mm": width,
        "layer": str(body.get("layer", ""))[:40],
        "points": points,
    }


def saved(body):
    """One stroke or one removal appended to the scan's marks."""
    marks, folder = folders_of(body["scan"])
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with _writing:
        if "remove" in body:
            entry["remove"] = str(body["remove"])
        else:
            stroke = stroke_of(body)
            entry |= stroke | {"id": uuid.uuid4().hex, "layout": layout_digest(marks, folder, stroke["layer"])}
        marks.mkdir(parents=True, exist_ok=True)
        with (marks / "marks.jsonl").open("a") as log:
            log.write(json.dumps(entry) + "\n")
    return entry


MOUNTS = {"/sheets/": SHEETS, "/atlas/": SITE_ATLAS, "/website/": BROWSER_ASSETS}


class Site(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if "?v=" in self.path else "no-cache")
        super().end_headers()

    def translate_path(self, path):
        """Serve the site and the depth and paper images linked from the PHerc0211 circles page."""
        parts = Path(super().translate_path(path)).relative_to(self.directory).parts
        if parts[:3] == ("submission", "PHerc0211", "work") and CIRCLE_IMAGE.fullmatch("/".join(parts[3:])):
            return str(CIRCLE_REVIEW.joinpath(*parts[3:]))
        if not parts or (root := MOUNTS.get(f"/{parts[0]}/")) is None:
            return None
        return str(root.joinpath(*parts[1:]))

    def send_head(self):
        if self.path.split("?", 1)[0] == "/":
            self.send_response(302)
            self.send_header("Location", "/sheets/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if self.translate_path(self.path) is None:
            return self.send_error(404)
        return super().send_head()

    def log_message(self, *_):
        pass

    def answer(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_part(self, asked):
        """A byte range of a file, as a viewer reads one tile of a sheet from its pack."""
        translated, found = self.translate_path(self.path), RANGE.fullmatch(asked.strip())
        if translated is None or not found or not (path := Path(translated)).is_file():
            return self.send_error(404)
        first, last, size = int(found[1]), int(found[2]), path.stat().st_size
        if first > last or last >= size:
            return self.send_error(416)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        with open(path, "rb") as pack:
            pack.seek(first)
            self.wfile.write(pack.read(last - first + 1))
        return None

    def sheet_folder(self, url_path):
        """The served folder of a sheet's tile store, from its site path."""
        translated = self.translate_path(url_path)
        if translated is None or not (Path(translated) / INDEX).is_file():
            raise FileNotFoundError(url_path)
        return Path(translated)

    def send_bytes(self, data, headers=()):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def send_tiles(self):
        """Decoded tiles of one sheet, asked as {"dir", "stamp", "size", "tiles": [[layer, level, c, r], ...]}."""
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            size, tiles = int(body.get("size", 512)), body["tiles"]
            if not 1 <= len(tiles) <= MOST_TILES or not 16 <= size <= 2048:
                raise ValueError("bad tile request")
            parts = tile_bytes(self.sheet_folder(body["dir"]), str(body["stamp"]), tiles, size)
        except LookupError as error:
            return self.answer(409, {"error": str(error)})
        except FileNotFoundError:
            return self.answer(404, {"error": "no tile store"})
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            return self.answer(400, {"error": str(error)})
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(sum(map(len, parts))))
        self.end_headers()
        for part in parts:
            self.wfile.write(part)
        return None

    def do_GET(self):
        if self.headers.get("Range"):
            return self.send_part(self.headers["Range"])
        if self.path.startswith("/tilemap?dir="):
            try:
                return self.send_bytes(tile_map(self.sheet_folder(self.path.removeprefix("/tilemap?dir="))))
            except FileNotFoundError:
                return self.send_error(404)
        if self.path == "/status":
            try:
                return self.answer(200, start_status())
            except (KeyError, OSError, ValueError) as error:
                return self.answer(503, {"error": str(error)})
        if not self.path.startswith("/marks?scan="):
            return super().do_GET()
        try:
            self.answer(200, marks_of(self.path.removeprefix("/marks?scan=")))
        except ValueError as error:
            self.answer(404, {"error": str(error)})

    def do_POST(self):
        if self.path == "/tiles" and self.headers.get("Content-Type") == "application/json":
            return self.send_tiles()
        if self.path != "/marks" or self.headers.get("Content-Type") != "application/json":
            return self.answer(403, {"error": "marks only"})
        try:
            entry = saved(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            return self.answer(400, {"error": str(error)})
        except OSError:
            return self.answer(500, {"error": "could not write marks"})
        return self.answer(200, entry)


def serve(port=PORT):
    http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Site, directory=HERE)).serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else PORT)
