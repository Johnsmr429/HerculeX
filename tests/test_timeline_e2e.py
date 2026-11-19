"""The run timeline on a board of its own: a three-stage chain whose idle times are known."""

import json
import os
import subprocess
import sys
from pathlib import Path

from runner import job_board, timeline

HERE = Path(__file__).resolve().parent.parent
MINUTE = 60.0


def job(name, added, started, ended, after=(), **slots):
    return {
        "name": name,
        "added": added,
        "started": started,
        "ended": ended,
        "after": json.dumps(list(after)),
        "state": "done" if ended else "running",
        **{slot: int(slots.get(slot, 0)) for slot in job_board.SLOTS},
    }


def chain(t0):
    """Sheets on the GPU and link for 10 min, pieces on the CPU for 20 min after it, the read after pieces, and a
    fetch that is ready at 40 min but starts at 45 although nothing holds the link."""
    m = MINUTE
    return [
        job("s-sheets", t0, t0, t0 + 10 * m, gpu=1, link=1),
        job("s-pieces", t0, t0 + 10 * m, t0 + 30 * m, after=["s-sheets"]),
        job("s-sheet", t0, t0 + 30 * m, t0 + 40 * m, after=["s-pieces"], gpu=1, engine=1, link=1),
        job("s-fetch", t0 + 40 * m, t0 + 45 * m, t0 + 50 * m, link=1),
    ]


def test_spans_merge_and_subtract():
    assert timeline.merged([(5, 7), (0, 2), (1, 3), (9, 9)]) == [(0, 3), (5, 7)]
    assert timeline.minus([(0, 10)], [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)]
    assert timeline.length(timeline.clipped([(0, 10), (20, 30)], 5, 25)) == 10


def test_idle_time_is_split_into_ready_and_behind_a_stage():
    jobs, t0 = chain(1000.0), 1000.0
    end = t0 + 50 * MINUTE
    link = timeline.slot_report(jobs, "link", t0, end, end)
    assert timeline.length(link["held"]) == 25 * MINUTE
    assert timeline.length(link["idle_behind"]) == 20 * MINUTE
    assert timeline.length(link["idle_ready"]) == 5 * MINUTE
    engine = timeline.slot_report(jobs, "engine", t0, end, end)
    assert timeline.length(engine["idle_behind"]) == 30 * MINUTE and not engine["idle_ready"]


def test_lanes_and_sparks_draw_what_they_cover():
    assert timeline.lane([(0, 50)], 0, 100, 4) == "██··"
    assert timeline.lane([(0, 10)], 0, 100, 4) == "▌···"
    assert timeline.spark([0, 50, 100, None], 100) == " ▄█ "


def test_the_command_reports_a_chain_from_the_board(tmp_path):
    folder = tmp_path / "board"
    env = os.environ | {"HERCULEX_BOARD": str(folder), "PYTHONPATH": str(HERE)}
    code = (
        "import json, sys\n"
        "from runner import job_board\n"
        "rows = json.loads(sys.argv[1])\n"
        "db = job_board.connect()\n"
        "for row in rows:\n"
        "    db.execute(f\"INSERT INTO jobs({', '.join(row)}) VALUES ({', '.join('?' * len(row))})\", tuple(row.values()))\n"
        "for i in range(50):\n"
        "    db.execute('INSERT INTO samples(time, gpu, net_mb_s, room_gb) VALUES (?, ?, ?, ?)',\n"
        "               (rows[0]['added'] + 60 * i + 1, 90 if i < 10 else 0, 30 if i < 10 else 0, 5))\n"
    )
    subprocess.run([sys.executable, "-c", code, json.dumps(chain(1_000_000.0))], env=env, cwd=HERE, check=True)
    t0 = 1_000_000.0
    (folder / "events").mkdir()
    (folder / "events" / "s-sheet.jsonl").write_text(
        json.dumps({"kind": "wait-fetch", "start": t0 + 30 * MINUTE, "end": t0 + 33 * MINUTE})
        + "\n"
        + json.dumps({"kind": "read", "start": t0 + 33 * MINUTE, "end": t0 + 40 * MINUTE})
        + "\n{cut short"
    )
    shown = subprocess.run(
        [sys.executable, "-m", "runner.timeline", "s-*", "--width", "50", "--json", str(tmp_path / "t.json")],
        env=env,
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    link = next(line for line in shown.splitlines() if line.startswith("link") and "held" in line)
    assert "held 50%" in link and "ready 0h05m" in link and "behind a stage 0h20m" in link
    assert "GPU %" in shown and "link MB/s" in shown and "mean" in shown
    phase = {line.split()[0]: line for line in shown.splitlines() if line.startswith("  ")}
    assert phase["wait-fetch"].endswith("0h03m") and phase["read"].endswith("0h07m")
    data = json.loads((tmp_path / "t.json").read_text())
    assert [j["name"] for j in data["jobs"]] == ["s-sheets", "s-pieces", "s-sheet", "s-fetch"]


def test_spans_are_written_whole_and_only_when_asked(tmp_path, monkeypatch):
    from runner import events

    monkeypatch.delenv("HERCULEX_EVENTS", raising=False)
    with events.span("nothing"):
        pass
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HERCULEX_EVENTS", str(path))
    with events.span("fetch", store="rows-0000-0012"):
        pass
    try:
        with events.span("read"):
            raise ValueError
    except ValueError:
        pass
    rows = events.spans(path)
    assert [(r["kind"], r.get("store")) for r in rows] == [("fetch", "rows-0000-0012"), ("read", None)]
    assert all(r["end"] >= r["start"] for r in rows)
