"""The one route from a scan to the website, stage by stage or all at once."""

import argparse
import importlib
import signal
import subprocess
import sys
import time

from runner import stage_queue
from storage.data_roots import (
    DATA,
    HERE,
    LIBRARY,
    NODE_ROUTE,
    OUTPUTS,
    ROUTE_OUTPUTS,
    stage_entries,
    stage_folder,
)
from storage.route_database import connect, index
from website.survey_status import snapshot

STAGES = {
    "sheets": "unwrapping.detection.detect_nodes",
    "pieces": "unwrapping.level.run_levels",
    "sheet": "runner.sheet_stage",
}
FETCH = "ink.reading.unit_fetch"
ROUTE = list(STAGES)
UPSTREAM = {"pieces": "sheets", "sheet": "pieces", "fetch": "pieces"}
IDLE_SECONDS = 15


def run_stage(stage, samples, band=False, *, scan=None, require_ready=False, since=None):
    if stage == "fetch":
        importlib.import_module(FETCH).run(samples, band, volume=scan, since=since)
        return
    if stage == "pieces" and require_ready:
        pattern = "band-*" if band else "*"
        detected = {
            path.parts[-4]
            for path in stage_entries(NODE_ROUTE, "nodes", pattern)
            if (scan is None or path.parts[-3] == scan) and stage_queue.done(path)
        }
        missing = set(samples) - detected
        if missing or not detected:
            selection = f" for {', '.join(sorted(missing))}" if missing else ""
            raise RuntimeError(f"no finished sheet-node units{selection}. Run the sheets stage first.")
    module = importlib.import_module(STAGES.get(stage, STAGES["sheet"]))
    if stage in ("layout", "ink", "sheet"):
        module.run(samples, volume=scan, mode=stage, require_ready=require_ready)
    else:
        module.run(samples, band=band, volume=scan)


def finished_marker(stage):
    return ROUTE_OUTPUTS / f"{stage}.finished"


def follow(stage, samples, band=False):
    """Run a stage until a pass that began after the stage before it finished."""
    before = UPSTREAM.get(stage) if UPSTREAM.get(stage) in ROUTE else None
    upstream = finished_marker(before) if before else None
    marker = finished_marker(stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    needs_all_nodes = stage == "pieces" and not band
    began = time.time()
    while True:
        upstream_done = upstream is None or upstream.exists()
        if upstream_done or not needs_all_nodes:
            run_stage(stage, samples, band, since=began)
        if upstream_done:
            break
        for _ in range(IDLE_SECONDS):
            if upstream.exists():
                break
            time.sleep(1)
    marker.write_text("finished\n")


def run_all(samples, band=False):
    """Each stage in its own process, so the link, the GPU and the CPU cores work at once on different units."""
    with stage_queue.stage_lock(ROUTE_OUTPUTS / "pipeline.lock"):
        for stage in ROUTE:
            finished_marker(stage).unlink(missing_ok=True)
        flag = ["--band"] if band else []
        workers = {}
        try:
            for stage in [*ROUTE, "fetch"]:
                workers[stage] = subprocess.Popen(
                    [sys.executable, "-m", "runner.pipeline", "follow", stage, *samples, *flag], cwd=HERE
                )
            while workers:
                for stage, worker in tuple(workers.items()):
                    code = worker.poll()
                    if code is None:
                        continue
                    del workers[stage]
                    if code:
                        print(f"{stage} failed ({code}). Rerun the same command to resume.", file=sys.stderr)
                        return code if code > 0 else 128 - code
                if workers:
                    time.sleep(0.2)
        finally:
            for worker in workers.values():
                worker.terminate()
            for worker in workers.values():
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    return 0


def status(samples):
    index()
    rows = (
        connect()
        .execute(
            "SELECT sample, volume, stage, COUNT(*), SUM(seconds) FROM units WHERE unit NOT IN ('joints', 'frame') "
            "GROUP BY sample, volume, stage ORDER BY sample, volume"
        )
        .fetchall()
    )
    table = {}
    for sample, volume, stage, count, seconds in rows:
        if not samples or sample in samples:
            table.setdefault((sample, volume), {})[stage] = (count, seconds)
    stages = ("sheet-nodes", "pieces")
    print(f"Route: {ROUTE_OUTPUTS.name}")
    print(f"{'scroll':14s}{'scan':16s}" + "".join(f"{s:>16s}" for s in stages) + f"{'cm2 laid':>12s}{'cm2 read':>12s}")
    for (sample, volume), row in table.items():
        layout = stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "layout.json"
        coverage = snapshot(sample, volume) if layout.is_file() else None
        print(
            f"{sample:14s}{volume:16s}"
            + "".join(f"{row.get(s, (0, 0))[0]:>8d} {row.get(s, (0, 0))[1] / 3600:6.2f}h" for s in stages)
            + (f"{coverage['laid_cm2']:12.0f}{coverage['ink_cm2']:12.0f}" if coverage else "")
        )
    if not table:
        print("No completed units for this selection.")


def interrupted(number, _frame):
    raise SystemExit(128 + number)


def main():
    parser = argparse.ArgumentParser(prog="runner.pipeline", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list scan IDs and finished stages").add_argument("samples", nargs="*")
    commands.add_parser("index", help="rebuild the route index")
    run = commands.add_parser("run", help="run or resume a stage")
    run.add_argument("stage", choices=("all", *STAGES, "fetch", "layout", "ink"))
    run.add_argument("samples", nargs="*", metavar="SCROLL")
    run.add_argument("--band", action="store_true", help="process a central band")
    run.add_argument("--scan", metavar="ID", help="choose one scan for this stage")
    follower = commands.add_parser("follow")
    follower.add_argument("stage", choices=(*STAGES, "fetch"))
    follower.add_argument("samples", nargs="*")
    follower.add_argument("--band", action="store_true")
    arguments = sys.argv[1:]
    if arguments[:1] in (["run"], ["follow"]) and "--band" in arguments:
        arguments = [a for a in arguments if a != "--band"] + ["--band"]
    args = parser.parse_args(arguments)
    if args.command == "run" and args.stage in ("layout", "ink") and not args.samples:
        parser.error(f"run {args.stage} needs a scroll")
    if args.command == "run" and args.scan and (len(args.samples) != 1 or args.stage == "all"):
        parser.error("--scan needs one scroll and one pipeline stage")
    if args.command == "run" and args.band and args.stage not in ("all", "sheets", "pieces"):
        parser.error("--band applies to all, sheets and pieces")
    for folder in (LIBRARY, OUTPUTS):
        if not folder.is_dir() and folder.is_relative_to(DATA):
            folder.mkdir(parents=True, exist_ok=True)
        if not folder.is_dir():
            parser.exit(1, f"HerculeX data folder is unavailable: {folder} (mounted? see storage/data_roots.py)\n")
    if args.command == "follow" or (args.command == "run" and args.stage in ("all", "sheet", "layout", "ink")):
        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "status":
            status(set(args.samples))
        elif args.command == "index":
            print(index())
        elif args.command == "follow":
            follow(args.stage, args.samples, args.band)
        elif args.stage == "all":
            return run_all(args.samples, args.band)
        else:
            run_stage(args.stage, args.samples, args.band, scan=args.scan, require_ready=True)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"HerculeX: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
