"""CLI: fetch a box (or the chunks around points) of a scan's array into the store; resolve, usage, evict."""

import argparse
import fcntl
import functools
import os
import shutil
import sys

import numpy as np

from scans.chunks import array_registry
from scans.chunks.chunk_store import INDEX, LOCK, ChunkStore
from scans.chunks.resolve_scan import KINDS, resolve
from scans.chunks.volume import Volume, parse_box
from storage.data_roots import CT_CHUNKS as STORE

USAGE = """scans.chunks.fetch SAMPLE SCAN KIND [--level L] (--box z0:z1,y0:y1,x0:x1 | --points FILE.npy --margin M)
scans.chunks.fetch resolve SAMPLE SCAN [--refresh]
scans.chunks.fetch usage
scans.chunks.fetch evict SAMPLE/NAME[/LEVEL]"""


def fetch(argv):
    parser = argparse.ArgumentParser(prog="scans.chunks.fetch", usage=USAGE)
    parser.add_argument("sample")
    parser.add_argument("scan")
    parser.add_argument("kind", choices=KINDS, help="the array: ct")
    parser.add_argument("--level", type=int, help="pyramid level, default 0")
    parser.add_argument("--box", type=parse_box, help="z0:z1,y0:y1,x0:x1 in this level's voxels")
    parser.add_argument("--points", help=".npy of (N, 3) zyx points in this level's voxels")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--connections", type=int)
    args = parser.parse_args(argv)
    if (args.box is None) == (args.points is None):
        parser.error("give exactly one of --box or --points")
    with Volume.for_scan(args.sample, args.scan, args.kind, args.level) as volume:
        print(f"{volume} {volume.url}", flush=True)
        progress = functools.partial(print, flush=True)
        if args.box is not None:
            report = volume.fetch(args.box, args.connections, progress)
        else:
            report = volume.fetch_points(np.load(args.points), args.margin, args.connections, progress)
    print(report.line(), flush=True)
    return 1 if report.failed or report.aborted else 0


def show_resolve(argv):
    parser = argparse.ArgumentParser(prog="scans.chunks.fetch resolve")
    parser.add_argument("sample")
    parser.add_argument("scan")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    for kind, (url, level) in sorted(resolve(args.sample, args.scan, args.refresh).items()):
        print(f"{kind:10s} {'' if level is None else level:>2} {url}")
    return 0


def level_folders(root=STORE):
    return sorted(p.parent for p in root.glob(f"*/*/*/{INDEX}"))


def usage(argv):
    if argv:
        raise SystemExit("usage: scans.chunks.fetch usage")
    known = {(a.sample, a.name, str(a.level)): a for a in array_registry.all_arrays()}
    total = 0
    print(f"{'array':80s} {'stored':>8s} {'absent':>8s} {'GB':>8s}")
    for folder in level_folders():
        key = folder.relative_to(STORE).parts
        info = known.get(key)
        store = ChunkStore(folder, info.chunk_bytes, info.url) if info else None
        if store is None:
            print(f"{'/'.join(key):80s} (not in the registry)")
            continue
        row = store.usage()
        store.close()
        total += row["bytes"]
        print(f"{'/'.join(key):80s} {row['stored']:8d} {row['absent']:8d} {row['bytes'] / 1e9:8.3f}")
    free = shutil.disk_usage(STORE).free if STORE.exists() else 0
    print(f"total {total / 1e9:.3f} GB in {STORE}; {free / 1e9:.0f} GB free on the disc")
    return 0


def evict(argv):
    if len(argv) != 1:
        raise SystemExit("usage: scans.chunks.fetch evict SAMPLE/NAME[/LEVEL]")
    target = (STORE / argv[0]).resolve()
    parts = target.relative_to(STORE.resolve()).parts if target.is_relative_to(STORE.resolve()) else ()
    if len(parts) not in (2, 3) or not target.is_dir():
        raise SystemExit(f"{argv[0]} is not an array (SAMPLE/NAME) or array level (SAMPLE/NAME/LEVEL) in {STORE}")
    if parts[:2] not in {(a.sample, a.name) for a in array_registry.all_arrays()}:
        raise SystemExit(f"{argv[0]} is not an array of the volume registry; not evicted")
    folders = [target] if len(parts) == 3 else sorted(target.iterdir())
    if not all(f.is_dir() and f.name.isdigit() and (f / INDEX).exists() for f in folders):
        raise SystemExit(f"{target} holds files the volume store did not write; not evicted")
    locks = []
    try:
        for folder in folders:
            lock = os.open(folder / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
            locks.append(lock)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"{folder} is being written by another process; not evicted") from None
        freed = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        shutil.rmtree(target)
    finally:
        for lock in locks:
            os.close(lock)
    for parent in target.parents:
        if parent == STORE.resolve() or any(parent.iterdir()):
            break
        parent.rmdir()
    print(f"evicted {target} ({freed / 1e9:.3f} GB)")
    return 0


def main(argv):
    commands = {"usage": usage, "evict": evict, "resolve": show_resolve}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    return fetch(argv)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("interrupted; what was stored is kept, run the same command again to resume", flush=True)
        sys.exit(130)
