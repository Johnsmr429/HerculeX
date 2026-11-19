"""Timed benchmarks of the pipeline's hot paths on fixed inputs, recorded so a slowdown shows as a number.

./run.sh runner.speed [NAME ...] [--record] [--compare]

Each benchmark runs on the same input every time: a frozen sheet control, or synthetic data seeded the same way
(the exact solve, the level fit, the brick store, the page step, both fetch clients). It
prints the median of its repeats; --record appends it to records/speed.jsonl with the commit, the load and the board's
running jobs; --compare flags any benchmark more than 15% slower than the best of its last five records. Run them on
the board (`--memory 4 --cores 8`), ideally alone, since other jobs make every number slower.
"""

import argparse
import http.server
import json
import os
import resource
import statistics
import subprocess
import threading
import time

import numpy as np

from storage.data_roots import HERE, SCRATCH, SHEET_CONTROLS

RECORDS = HERE / "records" / "speed.jsonl"
SLOWER = 1.15
CONTROL = "PHerc1667-20231117161658-rows-0008-0020"
BODY = 2 << 20


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def solver():
    """The exact level solve on a random graph of 200,000 patches and 1,000,000 terms."""
    from unwrapping.level import kernel_bindings as kb

    rng = np.random.default_rng(0)
    count, terms = 200_000, 1_000_000
    p = rng.integers(0, count, terms).astype(np.int32)
    q = ((p + rng.integers(1, 50, terms)) % count).astype(np.int32)
    k = rng.integers(-2, 3, terms).astype(np.int32)
    up, dn = rng.integers(1, 64, (2, terms)).astype(np.int32)
    start = time.perf_counter()
    kb.minimize(count, p, q, k, up, dn, np.zeros(count, dtype=np.int32))
    return {"seconds": time.perf_counter() - start, "terms": terms}


def level():
    """Production prepare and grow_pieces on a frozen sheet control, with the level route's own phase times."""
    from unwrapping.level import level_pieces

    start = time.perf_counter()
    nodes, theta, _first, _grid, pairs, record, _knots = level_pieces.prepare([SHEET_CONTROLS / CONTROL], None)
    prepared = time.perf_counter()
    level_pieces.grow_pieces(nodes, theta, pairs, record)
    done = time.perf_counter()
    phases = record["level_route"]["seconds"] | {"prepare": round(prepared - start, 1)}
    return {"seconds": done - start, "nodes": len(nodes), "phases": phases}


def bricks():
    """A brick store written with 512 bricks of 128³ (1 GiB) and read back in a random order."""
    from storage.brick_store import BrickStore

    folder = SCRATCH / "speed-bricks"
    subprocess.run(["rm", "-rf", str(folder)], check=True)
    store = BrickStore(folder, (1024, 1024, 1024), (128, 128, 128))
    raw = np.random.default_rng(0).integers(0, 255, 128**3, dtype=np.uint8).tobytes()
    start = time.perf_counter()
    for index in np.ndindex(8, 8, 8):
        store.put(index, raw)
    store.save()
    wrote = time.perf_counter()
    order = np.random.default_rng(1).permutation(512)
    reader = BrickStore(folder)
    total = sum(int(reader.read_brick(np.unravel_index(i, (8, 8, 8)))[0, 0, 0]) for i in order)
    read = time.perf_counter()
    subprocess.run(["rm", "-rf", str(folder)], check=True)
    return {"seconds": read - start, "write_s": round(wrote - start, 3), "read_s": round(read - wrote, 3), "sum": total}


class Bodies(http.server.BaseHTTPRequestHandler):
    payload = bytes(BODY)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(BODY))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *arguments):
        pass


def fetch(fetcher):
    """400 bricks of 2 MiB from a server on this Mac through one fetch client: its own ceiling and CPU cost."""
    from scans.transfer import parallel_fetch

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Bodies)
    server.protocol_version = "HTTP/1.1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    items = [((0, y, x), f"0/{y}/{x}") for y in range(20) for x in range(20)]
    moved = [0]
    parallel_fetch.FETCHER = fetcher
    cpu, start = cpu_seconds(), time.perf_counter()
    parallel_fetch.fetch_all(
        f"http://127.0.0.1:{server.server_address[1]}/b",
        items,
        lambda _, b: moved.__setitem__(0, moved[0] + len(b)),
        32,
    )
    seconds = time.perf_counter() - start
    server.shutdown()
    return {"seconds": seconds, "mb_s": round(moved[0] / 1e6 / seconds, 1), "cpu_s": round(cpu_seconds() - cpu, 2)}


def page():
    """A synthetic sheet of 120 pieces, 5 to 21 mm wide over 480 mm, composed into three layers of tiles."""
    from PIL import Image
    from scipy import ndimage as ni

    from website import sheet_tiles

    root = SCRATCH / "speed-page"
    subprocess.run(["rm", "-rf", str(root)], check=True)
    faces = root / "scroll" / "scan" / "papyrus-faces" / "level-0" / "unit"
    faces.mkdir(parents=True)
    rng, pieces, pixel_mm = np.random.default_rng(7), [], 0.01
    for n in range(120):
        h, w = int(rng.integers(500, 2100)), int(rng.integers(500, 2100))
        smooth = ni.zoom(ni.gaussian_filter(rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)), 2), 8, order=1)[:h, :w]
        scan = np.clip(110 + 45 * smooth + rng.normal(0, 10, (h, w)), 1, 255).astype(np.uint8)
        for suffix, image in (("", scan), ("-layer0", np.roll(scan, 3, axis=1)), ("-taught", (smooth > 0.3) * 200)):
            Image.fromarray(image.astype(np.uint8)).save(faces / f"piece-{n:04d}{suffix}.png")
        pieces.append(
            {"unit": "unit", "piece": n, "area": float(h * w), "um": pixel_mm, "stride": 2, "group": n // 4}
            | {"x_mm": float(rng.uniform(0.5, 480)), "y_mm": float(rng.uniform(0.5, 60))}
            | {"width_mm": w * pixel_mm, "height_mm": h * pixel_mm, "flip_rows": False, "flip_cols": False}
        )
    layout = root / "scroll" / "scan" / "papyrus-sheet" / "layout.json"
    layout.parent.mkdir(parents=True)
    layout.write_text(json.dumps({"pieces": pieces}))
    saved = sheet_tiles.ROUTE_OUTPUTS, sheet_tiles.SHEETS
    sheet_tiles.ROUTE_OUTPUTS, sheet_tiles.SHEETS = root, root / "sheets"
    try:
        start = time.perf_counter()
        sheet_tiles.run("scroll", "scan", pixel_mm, "page", ("taught", "layer0", "sheet"))
        seconds = time.perf_counter() - start
    finally:
        sheet_tiles.ROUTE_OUTPUTS, sheet_tiles.SHEETS = saved
        subprocess.run(["rm", "-rf", str(root)], check=True)
    return {"seconds": seconds, "pieces": len(pieces), "processes": sheet_tiles.PROCESSES}


BENCHMARKS = {
    "solver": (solver, 3),
    "level": (level, 1),
    "bricks": (bricks, 3),
    "page": (page, 1),
    "fetch-curl": (lambda: fetch("curl"), 3),
    "fetch-python": (lambda: fetch("python"), 3),
}


def context():
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=False
    ).stdout
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "."], cwd=HERE, capture_output=True, text=True, check=False
    )
    from runner import job_board

    running = [j["name"] for j in job_board.all_jobs(job_board.connect()) if j["state"] in job_board.ACTIVE]
    return {"commit": head.strip(), "dirty": bool(dirty.stdout.strip()), "load": round(os.getloadavg()[0], 1)} | {
        "beside": [name for name in running if not name.startswith("qol-speed")]
    }


def history(name):
    if not RECORDS.is_file():
        return []
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines() if line.strip()]
    return [row for row in rows if row["name"] == name]


def main():
    parser = argparse.ArgumentParser(prog="runner.speed", description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", choices=[[], *BENCHMARKS], help="default: every benchmark")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    about, slower = context(), []
    for name in args.names or BENCHMARKS:
        run, repeats = BENCHMARKS[name]
        results = [run() for _ in range(repeats)]
        median = statistics.median(r["seconds"] for r in results)
        details = {k: v for k, v in results[len(results) // 2].items() if k != "seconds"}
        best = min((row["seconds"] for row in history(name)[-5:]), default=None)
        verdict = "" if best is None else f"  best of last five {best:.2f} s"
        if best is not None and median > SLOWER * best:
            verdict += f"  SLOWER by {100 * (median / best - 1):.0f}%"
            slower.append(name)
        print(f"{name:14s}{median:8.2f} s  {json.dumps(details)}{verdict}", flush=True)
        if args.record:
            row = {"name": name, "seconds": round(median, 3), "repeats": repeats, "at": time.time()}
            with RECORDS.open("a") as out:
                out.write(json.dumps(row | details | about) + "\n")
    if about["beside"]:
        print(f"ran beside {', '.join(about['beside'])}: compare with care")
    if args.compare and slower:
        raise SystemExit(f"slower than recorded: {', '.join(slower)}")


if __name__ == "__main__":
    main()
