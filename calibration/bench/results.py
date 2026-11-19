"""Benchmark runs and their metrics in one SQLite file, and the report of the winding network against v1 per control."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
import time

from storage.data_roots import HERE as CODE
from storage.data_roots import SURFACE_BENCH as BENCH

DB = BENCH / "results.sqlite"
LANE = ("ml/winding", "calibration/bench", "scans/chunks", "storage/data_roots.py")
SCHEMA = """
create table if not exists runs (
    id integer primary key,
    unit text unique,
    created text,
    control text,
    split text,
    system text,
    method text,
    kind text,
    commit_id text,
    parameters text,
    source text,
    detail text
);
create table if not exists metrics (
    run integer references runs(id) on delete cascade,
    key text,
    value real,
    primary key (run, key)
);
create index if not exists runs_by_control on runs (control, system, kind, method);
"""
COLUMNS = (
    ("L-W", "network", "winding", None, "share_frames", "network W: share of truth on its frame's modal offset"),
    ("L-Wcov", "network", "winding", None, "coverage", "network W: share of the box's truth with a valid field"),
    ("L-adj", "network", "winding", None, "adjacent_exact", "network W: radially adjacent truth wraps one apart in W"),
    ("L-cm2", "network", "surface", None, "correct_cm2", "network sheets: cm2 on their dominant wrap"),
    ("L-wrong", "network", "surface", None, "wrong_rate", "network sheets: wrong-wrap area rate"),
    ("L-big", "network", "surface", None, "largest_correct_cm2", "network sheets: largest connected correct cm2"),
    ("gF", "v1", "winding", "grown-field", "share_frames", "v1 grown field kept whole, at nearest node, whole band"),
    ("lP-wr", "v1", "surface", "level-pieces", "wrong_rate", "v1 level pieces (ink calibration): wrong-wrap rate"),
    ("lP-big", "v1", "surface", "level-pieces", "largest_correct_cm2", "v1 level pieces: largest correct cm2"),
    ("gP-wr", "v1", "surface", "grown-pieces", "wrong_rate", "v1 grown pieces (ink calibration): wrong rate"),
    ("gP-big", "v1", "surface", "grown-pieces", "largest_correct_cm2", "v1 grown pieces: largest correct cm2"),
    ("pP-wr", "v1", "surface", "grown-production", "wrong_rate", "v1 production grown pieces: wrong rate"),
    ("pP-big", "v1", "surface", "grown-production", "largest_correct_cm2", "v1 production grown: largest cm2"),
)
RECORDED = (
    ("ws-wr", "wrap-study-patches", "usable.wrong_rate", "wrap study: wrong rate in pieces >=1 cm2"),
    ("ws-cm2", "wrap-study-patches", "usable.correct_cm2", "wrap study: truth cm2 judged right in pieces >=1 cm2"),
    ("ws-big", "wrap-study-patches", "largest_cm2", "wrap study: largest piece cm2 (claimed, not checked)"),
    ("rr-lvl", "ruled-roll-level-route", "modal_share", "ruled-roll level route: one-offset share (slab)"),
)
SLABS = (
    ("roll-frame-planes-", "ruled-roll-p50-frame-planes-"),
    ("level-field", "ruled-roll-level-frame"),
    ("grown-field", "ruled-roll-grown-field"),
)
FOOTINGS = """Footing. Table 1 is one footing: the bench's truth (names where wrap names exist, else v1's groups) and the
bench's metrics, for the winding network and for v1's stored outputs alike; only these columns compare the two. Its winding
numbers count every truth vertex in the box, pressed ones included; its surface numbers are area on truth, the modal
wrap per 4-connected part. Table 2 puts slab and whole-band fields on v1's own footing (groups truth, resolved
vertices of the largest group, as ideas/2026-11-12-ruled-roll quick/error_map) beside v1's recorded number: they
reproduce v1, they are not comparable with table 1. Table 3 is v1's records as written: the wrap study re-grew its
own pieces and counted truth vertices judged per piece and group with pressed truth left out; none of it compares
with any bench column."""


def connect():
    BENCH.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=60)
    db.execute("pragma foreign_keys = on")
    db.executescript(SCHEMA)
    return db


def commit_id():
    def git(*words):
        done = subprocess.run(["git", "-C", str(CODE), *words], capture_output=True, text=True, check=False)
        return done.stdout.strip()

    head = git("rev-parse", "--short", "HEAD") or "unknown"
    return head + ("+dirty" if git("status", "--porcelain", "--", *LANE) else "")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def record(control, system, method, kind, metrics, parameters, source=None, detail=None, split=None):
    """Store one run (replacing an earlier run of the same unit) and return its id."""
    from calibration.bench.truth import split as split_of

    unit = f"{system}|{method}|{kind}|{control}|{digest(parameters)}"
    with connect() as db:
        db.execute("delete from runs where unit = ?", (unit,))
        cursor = db.execute(
            "insert into runs (unit, created, control, split, system, method, kind, commit_id, parameters, source, "
            "detail) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                unit,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                control,
                split or split_of(control),
                system,
                method,
                kind,
                commit_id(),
                json.dumps(parameters, sort_keys=True, default=str),
                source,
                None if detail is None else json.dumps(detail, default=str),
            ),
        )
        run = cursor.lastrowid
        db.execute("delete from metrics where run = ?", (run,))
        db.executemany(
            "insert into metrics (run, key, value) values (?, ?, ?)",
            [(run, k, float(v)) for k, v in metrics.items() if v is not None],
        )
    return run


def flatten(value, prefix=""):
    """Numbers of a nested dict as dotted keys."""
    out = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            out |= flatten(item, f"{name}.")
        elif isinstance(item, bool | int | float):
            out[name] = float(item)
    return out


def latest(db, control, system, kind, method=None):
    """Metrics of the newest run matching, or None; a method ending in * matches by prefix."""
    query = "select id, method from runs where control = ? and system = ? and kind = ?"
    values = [control, system, kind]
    if method and method.endswith("*"):
        query += " and method like ? and method not like ?"
        values += [method[:-1] + "%", "%@%"]
    elif method:
        query += " and method = ?"
        values.append(method)
    row = db.execute(query + " order by created desc, id desc limit 1", values).fetchone()
    if row is None:
        return None, None
    return dict(db.execute("select key, value from metrics where run = ?", (row[0],)).fetchall()), row[1]


def cell(value):
    return "-" if value is None else f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    return ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in [header, *rows]]


def truth_of_run(db, control, system, kind, method):
    row = db.execute(
        "select parameters from runs where control = ? and system = ? and kind = ? and method = ? "
        "order by created desc, id desc limit 1",
        (control, system, kind, method),
    ).fetchone()
    return json.loads(row[0]).get("truth", "?") if row else None


def slab_rows(db, control):
    """Per v1 field: the bench's share, v1's footing reproduced by the bench, and v1's recorded share."""
    rows = []
    for computed, recorded in SLABS:
        found = db.execute(
            "select distinct method from runs where control = ? and system = 'v1' and kind = 'winding' "
            "and method like ?",
            (control, computed + "%"),
        ).fetchall()
        for (method,) in sorted(found):
            base = method.split("@")[0]
            mine, _ = latest(db, control, "v1", "winding", method)
            on_groups = method.endswith("@groups") or truth_of_run(db, control, "v1", "winding", method) == "groups"
            target = recorded and (recorded + base[len(computed) :] if computed.endswith("-") else recorded)
            theirs, _ = latest(db, control, "v1", "recorded", target) if target else (None, None)
            v1_share = mine.get("v1_share_largest_frame_resolved") if on_groups else None
            recorded_share = theirs.get("modal_share") if theirs and on_groups else None
            rows.append(
                [
                    control,
                    method,
                    truth_of_run(db, control, "v1", "winding", method),
                    cell(mine.get("share_frames")),
                    cell(mine.get("coverage")),
                    cell(v1_share),
                    cell(recorded_share),
                    "-" if v1_share is None or recorded_share is None else f"{v1_share - recorded_share:+.4f}",
                ]
            )
    return rows


def report(split="all", network_method=None, v1_truth="default", sources=False):
    """Three tables on stated footings (see FOOTINGS); v1_truth 'groups' fills table 1's v1 columns from v1's group
    truth where wrap names exist, which no longer matches the network columns' truth."""
    from calibration.bench.truth import controls, default_source
    from calibration.bench.truth import split as split_of

    lines = [FOOTINGS, ""]
    with connect() as db:
        chosen = [c for c in controls() if split == "all" or split_of(c) == split]
        rows = []
        for control in chosen:
            cells = [control, split_of(control), default_source(control) or "-"]
            for _, system, kind, method, key, _ in COLUMNS:
                metrics = None
                if system == "v1" and v1_truth == "groups":
                    metrics, _ = latest(db, control, system, kind, f"{method}@groups")
                if metrics is None:
                    metrics, _ = latest(db, control, system, kind, network_method if system == "network" else method)
                cells.append(cell(None if metrics is None else metrics.get(key)))
            rows.append(cells)
        lines.append("Table 1. Bench footing, whole control (network and v1 comparable)")
        lines += table(["control", "split", "truth", *(c[0] for c in COLUMNS)], rows)
        lines += [f"  {c[0]:7s} {c[5]} [{c[1]} {c[2]} {c[3] or 'newest'} {c[4]}]" for c in COLUMNS]
        lines += ["", "Table 2. v1 fields: bench footing, v1's footing reproduced, v1's record (groups truth only)"]
        slabs = [r for control in chosen for r in slab_rows(db, control)]
        header = ["control", "v1 field", "truth", "share", "coverage", "v1-footing", "recorded", "diff"]
        lines += table(header, slabs)
        lines += ["", "Table 3. v1 records as written (v1's footing; not comparable with tables 1 and 2)"]
        rows = []
        for control in chosen:
            values = [latest(db, control, "v1", "recorded", method)[0] for _, method, _, _ in RECORDED]
            rows.append([control] + [cell(v and v.get(key)) for v, (_, _, key, _) in zip(values, RECORDED)])
        lines += table(["control", *(c[0] for c in RECORDED)], rows)
        lines += [f"  {c[0]:7s} {c[3]} [{c[1]} {c[2]}]" for c in RECORDED]
        if sources:
            found = db.execute("select distinct method, source from runs where source is not null").fetchall()
            lines.extend(f"source of {m}: {s}" for m, s in sorted(set(found)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shown = commands.add_parser("report", help="the winding network against v1, one row per control")
    shown.add_argument("--split", default="all", choices=("all", "dev", "holdout"))
    shown.add_argument("--network", help="network method to show (default: newest run of any method)")
    shown.add_argument("--v1-truth", default="default", choices=("default", "groups"))
    shown.add_argument("--sources", action="store_true", help="list the source of every recorded method")
    listing = commands.add_parser("runs", help="list runs")
    listing.add_argument("--control")
    detail = commands.add_parser("show", help="one run with its metrics")
    detail.add_argument("run", type=int)
    args = parser.parse_args()
    if args.command == "report":
        print(report(args.split, args.network, args.v1_truth, args.sources))
    elif args.command == "runs":
        with connect() as db:
            query = "select id, created, control, system, kind, method, commit_id from runs"
            rows = db.execute(
                query + (" where control = ?" if args.control else "") + " order by id",
                (args.control,) if args.control else (),
            ).fetchall()
        for row in rows:
            print(" | ".join(str(v) for v in row))
    else:
        with connect() as db:
            run = db.execute("select * from runs where id = ?", (args.run,)).fetchone()
            metrics = db.execute("select key, value from metrics where run = ? order by key", (args.run,)).fetchall()
        if run is None:
            raise SystemExit(f"no run {args.run}")
        names = (
            "id",
            "unit",
            "created",
            "control",
            "split",
            "system",
            "method",
            "kind",
            "commit",
            "parameters",
            "source",
        )
        for name, value in zip(names, run):
            print(f"{name}: {value}")
        for key, value in metrics:
            print(f"  {key} = {value:g}")


if __name__ == "__main__":
    main()
