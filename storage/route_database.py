"""One SQLite index of every stage's finished units, rebuilt from their records."""

import json
import sqlite3

from runner.stage_queue import RECORD
from storage.data_roots import ROUTE_DATABASE, ROUTE_OUTPUTS, stage_folder

SCHEMA = """
CREATE TABLE IF NOT EXISTS units(stage TEXT, sample TEXT, volume TEXT, unit TEXT, method TEXT, seconds REAL, record TEXT,
    PRIMARY KEY(stage, sample, volume, unit));
CREATE TABLE IF NOT EXISTS pieces(sample TEXT, volume TEXT, unit TEXT, piece INTEGER, area_mm2 REAL, nodes INTEGER, winding INTEGER,
    frame INTEGER, written INTEGER, PRIMARY KEY(sample, volume, unit, piece));
"""
STAGES = {"sheet-nodes": "*", "pieces": "*"}


def connect():
    ROUTE_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(ROUTE_DATABASE)
    database.execute("PRAGMA journal_mode=WAL")
    database.executescript(SCHEMA)
    return database


def seconds_of(record):
    value = record.get("seconds", 0.0)
    return float(sum(value.values()) if isinstance(value, dict) else value)


def index():
    """Every finished unit's record into the database."""
    database = connect()
    counts = {}
    with database:
        database.execute("DELETE FROM units")
        database.execute("DELETE FROM pieces")
        for scan in sorted(ROUTE_OUTPUTS.glob("*/*")):
            if not scan.is_dir():
                continue
            sample, volume = scan.parent.name, scan.name
            for stage, pattern in STAGES.items():
                base = stage_folder(scan, "nodes" if stage == "sheet-nodes" else stage)
                folders = [base] if pattern == "." else sorted(base.glob(pattern))
                for folder in folders:
                    path = folder / RECORD
                    if folder.name.endswith(".building") or not path.exists():
                        continue
                    record = json.loads(path.read_text())
                    unit = str(folder.relative_to(base)) if pattern != "." else stage
                    database.execute(
                        "INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (stage, sample, volume, unit, record.get("method"), seconds_of(record), json.dumps(record)),
                    )
                    counts[stage] = counts.get(stage, 0) + 1
                    if stage == "pieces" and unit not in ("joints", "frame"):
                        database.executemany(
                            "INSERT INTO pieces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                (
                                    sample,
                                    volume,
                                    unit,
                                    p["piece"],
                                    p["area_mm2"],
                                    p["nodes"],
                                    p["winding"],
                                    p["frame"],
                                    int(p.get("written", False)),
                                )
                                for p in record["pieces"]
                            ],
                        )
    return counts
