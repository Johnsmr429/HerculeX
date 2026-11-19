"""Where the space on the three discs goes, by folder and by owner, and what can be removed without losing work.

./run.sh runner.disk [--depth N]        a report; nothing is removed
./run.sh runner.disk reclaimable        only what can go, with the command that removes each

Reclaimable means one of: a unit folder still named .building that no process is writing (a killed run; the stage
writes it again), a CT store of the ink read whose pieces all have their ink maps (the read drops it itself unless
HERCULEX_KEEP_BRICKS is set), a sheet-detection work store whose chunk has its nodes, or scratch files older than a
day. Removal stays a person's decision: the report prints the paths and sizes, never deletes.
"""

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runner import stage_queue
from storage.data_roots import DETECTION_WORK, FACES, HERE, LEVEL_ROUTE, LIBRARY, NODE_ROUTE, OUTPUTS, SCRATCH

DAY = 86400
SHOWN = 8


def sizes(paths):
    """Bytes under each path, by du, in parallel."""

    def one(path):
        found = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=False).stdout.split()
        return int(found[0]) * 1024 if found else 0

    with ThreadPoolExecutor(8) as pool:
        return dict(zip(paths, pool.map(one, paths)))


def gib(n):
    return f"{n / 2**30:7.1f} GiB"


def children(folder):
    try:
        return sorted(p for p in folder.iterdir() if not p.name.startswith(".") or p.name == ".building")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def tree(folder, depth, indent=""):
    """Lines of a folder's largest children down to `depth`."""
    parts = children(folder)
    measured = sizes([p for p in parts if not p.is_symlink()])
    lines = []
    for path, size in sorted(measured.items(), key=lambda kv: -kv[1])[:SHOWN]:
        lines.append(f"{indent}{gib(size)}  {path.name}")
        if depth > 1 and path.is_dir() and size > 2**30:
            lines += tree(path, depth - 1, indent + "    ")
    rest = sorted(measured.values(), reverse=True)[SHOWN:]
    if rest:
        lines.append(f"{indent}{gib(sum(rest))}  ({len(rest)} more)")
    return lines


def unfinished_units():
    """Unit folders still named .building, with no claim held on them: a killed run left them."""
    found = {}
    for route in (NODE_ROUTE, LEVEL_ROUTE):
        for path in route.glob("*/*/*/*.building"):
            claim = path.with_name(f".{path.name.removesuffix('.building')}.claim")
            if not claim.exists() and time.time() - path.stat().st_mtime > DAY / 24:
                found.setdefault(path.resolve(), path.resolve())
    return list(found.values())


def read_stores():
    """CT stores of the ink read whose unit's pieces all have their ink maps."""
    found = []
    for store in (LIBRARY / "cache" / "bricks").glob("*/*"):
        planned = store / "planned.json"
        if not planned.is_file():
            continue
        pieces = [Path(p) for p in json.loads(planned.read_text())]
        faces = [p.parents[2] / "faces" / FACES / p.parent.name / f"{p.name}.png" for p in pieces]
        if pieces and all(face.is_file() for face in faces):
            found.append(store)
    return found


def detection_stores():
    """Work stores of sheet detection whose chunk already has its nodes."""
    found = []
    for store in DETECTION_WORK.glob("*/*/L*-*-*"):
        sample, volume, (_, first, last) = store.parts[-3], store.parts[-2], store.name.split("-")
        nodes = NODE_ROUTE / sample / volume / "nodes" / f"rows-{first}-{last}"
        if stage_queue.done(nodes):
            found.append(store)
    return found


def old_scratch():
    """Folders in the scratch untouched for a day; pytest and compiler caches look after themselves."""
    return [
        p
        for p in children(SCRATCH)
        if p.is_dir()
        and not p.name.startswith(("pytest-of-", "torchinductor"))
        and time.time() - p.stat().st_mtime > DAY
    ]


def reclaimable():
    """(kind, path, bytes) of everything that can go."""
    groups = {
        "unfinished unit": unfinished_units(),
        "read CT store": read_stores(),
        "detection store": detection_stores(),
        "old scratch": old_scratch(),
    }
    measured = sizes([p for paths in groups.values() for p in paths])
    return [(kind, path, measured[path]) for kind, paths in groups.items() for path in paths]


def report(depth):
    for name, root in (("Mac", HERE), ("library", LIBRARY), ("outputs", OUTPUTS)):
        usage = shutil.disk_usage(root)
        print(f"{name}: {gib(usage.free).strip()} free of {gib(usage.total).strip()} ({root})")
    for name, root in (("outputs", OUTPUTS), ("library", LIBRARY), ("runtime", HERE / "runtime")):
        print(f"\n{name}")
        print("\n".join(tree(root, depth, "  ")))
    studies = OUTPUTS / "studies"
    if studies.is_dir():
        print("\nstudies by owner")
        print("\n".join(tree(studies, 1, "  ")))
    found = reclaimable()
    print(
        f"\nreclaimable: {gib(sum(n for *_, n in found)).strip()} in {len(found)} folders; see `runner.disk reclaimable`"
    )


def main():
    parser = argparse.ArgumentParser(prog="runner.disk", description=__doc__.split("\n\n")[0])
    parser.add_argument("what", nargs="?", choices=("report", "reclaimable"), default="report")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    if args.what == "report":
        report(args.depth)
        return
    for kind, path, size in sorted(reclaimable(), key=lambda row: -row[2]):
        print(f"{gib(size)}  {kind:16s}  rm -r '{path}'")


if __name__ == "__main__":
    main()
