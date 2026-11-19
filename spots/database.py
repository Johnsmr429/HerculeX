"""The circled-spot results database: one Parquet file per table under SPOT_CANDIDATES/db.

Tables and their keys:

    spots      spot_id                                  every circle in the marks, with where it lies (catalogue.py)
    surfaces   spot_id, surface                         each surface built for a spot: original, snapped to m7, ...
    reads      spot_id, surface, model, side, depth     each ink map of a surface: one model, one side, one depth
    verdicts   spot_id, surface, judged_at              what a reader of the images saw, appended, never replaced

Writing a row whose key is already there replaces that row; verdicts only ever grow. A circle erased in the viewer
leaves spots, surfaces and reads at the next catalogue build (forget). Writes take a lock and go through a temporary
file, so a reader never sees half a table.
"""

import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.parquet as pq

from storage.data_roots import SPOT_CANDIDATES

DB = SPOT_CANDIDATES / "db"
KEYS = {
    "spots": ("spot_id",),
    "surfaces": ("spot_id", "surface"),
    "reads": ("spot_id", "surface", "model", "side", "depth"),
    "verdicts": ("spot_id", "surface", "judged_at"),
}


def path_of(table):
    if table not in KEYS:
        raise ValueError(f"no table {table}")
    return DB / f"{table}.parquet"


@contextmanager
def locked():
    DB.mkdir(parents=True, exist_ok=True)
    with open(DB / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def rows(table):
    """Every row of a table as dictionaries, or none before its first write."""
    path = path_of(table)
    return pq.read_table(path).to_pylist() if path.exists() else []


def key_of(table, row):
    return tuple(row.get(k) for k in KEYS[table])


def stored(table, all_rows):
    """A whole table written in place of the old one; call with the lock held."""
    columns = list(dict.fromkeys(k for r in all_rows for k in r))
    data = pa.Table.from_pylist([{c: r.get(c) for c in columns} for r in all_rows])
    path = path_of(table)
    part = path.with_suffix(".parquet.part")
    pq.write_table(data, part, compression="zstd")
    os.replace(part, path)


def write(table, new_rows):
    """Rows added to a table, replacing any with the same key."""
    if not new_rows:
        return
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_rows = [{**r, "written_at": r.get("written_at", stamp)} for r in new_rows]
    with locked():
        fresh = {key_of(table, r) for r in new_rows}
        stored(table, [r for r in rows(table) if key_of(table, r) not in fresh] + new_rows)


def forget(spot_ids):
    """Erased spots dropped from spots, surfaces and reads; verdicts keep what a reader saw."""
    gone = set(spot_ids)
    if not gone:
        return
    with locked():
        for table in ("spots", "surfaces", "reads"):
            old = rows(table)
            if any(r["spot_id"] in gone for r in old):
                stored(table, [r for r in old if r["spot_id"] not in gone])


def summary():
    """How many rows each table holds."""
    return {t: len(rows(t)) for t in KEYS}


def verdict(argv):
    """One reader's judgement of a spot's surface, appended."""
    import argparse

    parser = argparse.ArgumentParser(prog="./run.sh spots.database verdict")
    parser.add_argument("spot_id")
    parser.add_argument("surface")
    parser.add_argument("--by", required=True, help="who looked")
    parser.add_argument("--letters", type=int, default=0, help="letters that can be named")
    parser.add_argument("--readable", choices=("no", "traces", "letters", "passage"), required=True)
    parser.add_argument("--stack", default="ensemble")
    parser.add_argument("--side", default="inward")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--reading", default="", help="the letters as read")
    parser.add_argument("--note", default="")
    a = parser.parse_args(argv)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row = vars(a) | {"judged_at": stamp}
    write("verdicts", [row])
    print(json.dumps(row))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verdict":
        verdict(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in KEYS:
        for r in rows(sys.argv[1]):
            print(json.dumps(r, default=str))
    else:
        print(json.dumps(summary()))
