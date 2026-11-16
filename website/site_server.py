"""The local site at http://localhost:8765/sheets/index.html and the hand-mark store."""

import functools
import hashlib
import http.server
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path

from storage.data_roots import (
    BROWSER_ASSETS,
    CIRCLE_REVIEW,
    GROWN_ROUTE,
    HAND_MARKS,
    HERE,
    LEVEL_ROUTE,
    SHEETS,
    SITE_ATLAS,
    stage_folder,
)

PORT = 8765
NAME = re.compile(r"[A-Za-z0-9._-]+")
RANGE = re.compile(r"bytes=(\d+)-(\d+)")
CIRCLE_IMAGE = re.compile(r"paper/[\w-]+\.jpg|depths/[\w-]+/[\w+-]+\.png")
KINDS = ("ink", "blank", "spot")
MOST_POINTS = 20000
_writing = threading.Lock()


def folders_of(scan):
    """A scan's marks folder in the hand marks and the folder of the layout they lie on."""
    parts = str(scan).split("/")
    if not 2 <= len(parts) <= 3 or not all(NAME.fullmatch(p) and p not in (".", "..") for p in parts):
        raise ValueError(f"no such scan: {scan}")
    if len(parts) == 2 and parts[0] == "two-unwraps":
        marks, folder, layout = HAND_MARKS / "two_unwraps" / parts[1], SHEETS / f"two-unwraps-{parts[1]}", "placed.json"
    elif len(parts) == 2 and parts[0] == "part":
        marks, folder, layout = HAND_MARKS / "parts" / parts[1], SHEETS / parts[1], "placed.json"
    elif len(parts) == 3 and parts[0] == "level":
        marks, folder, layout = (
            HAND_MARKS / "level_route" / parts[1] / parts[2],
            stage_folder(LEVEL_ROUTE / parts[1] / parts[2], "sheet"),
            "layout.json",
        )
    elif len(parts) == 2:
        marks, folder, layout = (
            HAND_MARKS / "grown_route" / parts[0] / parts[1],
            stage_folder(GROWN_ROUTE / parts[0] / parts[1], "sheet"),
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

    def do_GET(self):
        if self.headers.get("Range"):
            return self.send_part(self.headers["Range"])
        if not self.path.startswith("/marks?scan="):
            return super().do_GET()
        try:
            self.answer(200, marks_of(self.path.removeprefix("/marks?scan=")))
        except ValueError as error:
            self.answer(404, {"error": str(error)})

    def do_POST(self):
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
    serve()
