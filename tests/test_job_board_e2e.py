"""The job board on tiny real jobs in a board of its own."""

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from runner import job_board, stage_queue
from runner.lane import CUDA

HERE = Path(__file__).resolve().parent.parent


def board_run(folder, *words, check=True):
    environment = os.environ | {
        "HERCULEX_BOARD": str(folder),
        "HERCULEX_BOARD_MARGIN_GB": "-100",
        "HERCULEX_BOARD_TICK": "0.2",
        "HERCULEX_BOARD_RETRY": "0.2",
        "PYTHONPATH": str(HERE),
    }
    return subprocess.run(
        [sys.executable, "-m", "runner.job_board", *words],
        cwd=HERE,
        env=environment,
        capture_output=True,
        text=True,
        check=check,
    )


def row(folder, name):
    with sqlite3.connect(folder / "board.sqlite") as database:
        database.row_factory = sqlite3.Row
        return dict(database.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone())


def until(test, seconds=30):
    limit = time.time() + seconds
    while time.time() < limit:
        if test():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def board(tmp_path):
    yield tmp_path
    lock = tmp_path / "dispatcher.lock"
    if lock.exists():
        until(lambda: lock.read_text().strip(), 5)
    if lock.exists() and lock.read_text().strip():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(lock.read_text()), signal.SIGKILL)


def waiting_job(**fields):
    return {
        "name": "x",
        "held": 0,
        "not_before": 0,
        "after": "[]",
        "needs": "[]",
        "folder": "/",
        "gpu": 0,
        "engine": 0,
        "link": 0,
        "cores": 1.0,
        "memory_gb": 1.0,
        "alone": 0,
        "last": 0,
        "added": 1.0,
        "priority": 0,
        "kind": "x",
        "free_gib": 0.0,
        "at_most": None,
        "state": "waiting",
    } | fields


def test_a_job_says_why_it_waits():
    state = {"gpu": 0, "swap_gb": 1.0}
    assert job_board.blocked(waiting_job(), [], state, 5.0) is None
    assert job_board.blocked(waiting_job(held=1), [], state, 5.0) == "held"
    busy = [waiting_job(name="t", memory_gb=1.0, state="running")]
    assert "needs 6.0 GB" in job_board.blocked(waiting_job(memory_gb=6.0), busy, state, 5.0)
    assert job_board.blocked(waiting_job(memory_gb=6.0), [], state, 5.0) is None
    assert (
        job_board.blocked(
            waiting_job(memory_gb=6.0), [waiting_job(name="gate", memory_gb=0.1, state="running")], state, 5.0
        )
        is None
    )
    assert "swap" in job_board.blocked(waiting_job(memory_gb=2.0), [], {"gpu": 0, "swap_gb": 5.0}, 5.0)
    assert job_board.blocked(waiting_job(memory_gb=0.5), [], {"gpu": 0, "swap_gb": 5.0}, 5.0) is None
    assert "busy off the board" in job_board.blocked(waiting_job(gpu=1), [], {"gpu": 97, "swap_gb": 1.0}, 5.0)
    assert "gpu is with t" in job_board.blocked(
        waiting_job(gpu=1), [waiting_job(name="t", gpu=1, state="running")], state, 5.0
    )
    assert "cores" in job_board.blocked(
        waiting_job(cores=4.0), [waiting_job(name="t", cores=7.0, state="running")], state, 5.0
    )
    assert "after t, running" in job_board.blocked(
        waiting_job(after='["t"]'), [waiting_job(name="t", state="running")], state, 5.0
    )
    assert "needs /no/such/path" in job_board.blocked(waiting_job(needs='["/no/such/path"]'), [], state, 5.0)
    assert "machine is with t" in job_board.blocked(
        waiting_job(), [waiting_job(name="t", alone=1, state="running")], state, 5.0
    )
    assert "to itself: t" in job_board.blocked(
        waiting_job(alone=1), [waiting_job(name="t", state="running")], state, 5.0
    )
    assert "2 processes off the board" in job_board.blocked(
        waiting_job(alone=1), [], state | {"strangers": [1, 2]}, 5.0
    )


def test_a_sharing_job_runs_beside_one_whole_job_on_the_gpu_and_link():
    state = {"gpu": 91, "swap_gb": 1.0}
    read = waiting_job(name="read", gpu=1, engine=1, link=1, state="running")
    nodes = waiting_job(name="nodes", gpu=1, link=1, share=1)
    assert job_board.blocked(nodes, [read], state, 5.0) is None
    assert "gpu is with read" in job_board.blocked(waiting_job(gpu=1, link=1), [read], state, 5.0)
    assert "gpu is with read" in job_board.blocked(
        nodes, [read, nodes | {"name": "n2", "state": "running"}], state, 5.0
    )
    assert "engine is with read" in job_board.blocked(waiting_job(engine=1, share=1), [read], state, 5.0)
    fetch = waiting_job(name="fetch", link=1, share=1)
    fetching = fetch | {"name": "f2", "state": "running"}
    detecting = nodes | {"name": "n1", "state": "running"}
    assert job_board.blocked(fetch, [read, detecting], state, 5.0) is None
    assert "link is with read" in job_board.blocked(fetch, [read, detecting, fetching], state, 5.0)
    assert "link is with n1" in job_board.blocked(waiting_job(link=1), [detecting, fetching, fetching], state, 5.0)
    assert job_board.blocked(waiting_job(gpu=1, link=1), [detecting, fetching], state, 5.0) is None
    assert job_board.blocked(waiting_job(gpu=1, link=1), [nodes | {"state": "running"}], state, 5.0) is None
    assert "busy off the board" in job_board.blocked(nodes, [], state, 5.0)


def test_a_gpu_job_short_of_cores_stops_the_lowest_restartable_job_and_it_comes_back(monkeypatch):
    database = sqlite3.connect(":memory:")
    database.executescript(job_board.SCHEMA)
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    read = waiting_job(name="read", gpu=1, cores=5.0, memory_gb=5.0, priority=90)
    pieces = [
        waiting_job(
            name=f"pieces-{n}",
            cores=4.0,
            memory_gb=1.0,
            priority=priority,
            state="running",
            restartable=1,
            adopted=0,
            started=float(n),
            pid=None,
        )
        for n, priority in ((1, 48), (2, 45), (3, 45))
    ]
    told = [(read, "needs 5 cores, 12 of 10 taken")]
    assert job_board.make_way(database, [read, *pieces], told, 5.0) == "pieces-3"
    assert pieces[2]["state"] == "stopping"
    assert job_board.make_way(database, [read, *pieces], told, 5.0) is None
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    fixed = [job | {"restartable": 0} for job in pieces[:2]]
    assert job_board.make_way(database, [read, *fixed], told, 5.0) is None
    assert job_board.make_way(database, [read, *pieces[:2]], [(read, "gpu is with x")], 5.0) is None
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    nodes = pieces[0] | {"name": "nodes", "cores": 1.0, "priority": 26, "state": "running"}
    page = waiting_job(name="page", cores=4.0, priority=110, state="running", restartable=0, adopted=0)
    one = pieces[1] | {"state": "running"}
    told = [(read, "needs 5 cores, 9 of 10 taken")]
    assert job_board.make_way(database, [read, page, one, nodes], told, 5.0) == "pieces-2"
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    gated = read | {"after": '["fetch"]'}
    fetch = waiting_job(name="fetch", link=1, share=1, after='["layout"]')
    layout = waiting_job(name="layout", cores=1.0, memory_gb=2.5, priority=66, after='["pieces"]')
    built = waiting_job(name="pieces", cores=3.0, memory_gb=5.5, priority=57)
    detecting = nodes | {"name": "detect", "memory_gb": 5.5, "priority": 35, "share": 1}
    board = [gated, fetch, layout, built, detecting, pieces[1] | {"state": "running", "priority": 55}]
    assert job_board.feeds_gpu(layout, board) and not job_board.feeds_gpu(built, board)
    told = [(layout, "needs 2.5 GB, room 0.3"), (built, "needs 5.5 GB, room 0.3")]
    assert job_board.make_way(database, board, told, 0.3) == "detect"
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    assert job_board.make_way(database, board, told[1:], 0.3) is None
    monkeypatch.setitem(job_board._made_way, "at", 0.0)
    assert job_board.make_way(database, board[:4] + board[5:], told[:1], 0.3) is None


def test_a_job_waits_on_the_board_for_disc_room(tmp_path):
    state = {"gpu": 0, "swap_gb": 1.0}
    assert "GiB free on the Mac" in job_board.blocked(waiting_job(free_gib=1e9), [], state, 5.0)
    assert job_board.blocked(waiting_job(free_gib=0.001), [], state, 5.0) is None
    (tmp_path / "survey-a").mkdir()
    (tmp_path / "survey-b").mkdir()
    pattern = str(tmp_path / "survey-*")
    assert "2 there" in job_board.blocked(waiting_job(at_most=json.dumps([2, pattern])), [], state, 5.0)
    assert job_board.blocked(waiting_job(at_most=json.dumps([3, pattern])), [], state, 5.0) is None


def test_memory_a_large_job_waits_for_is_kept_from_smaller_ones():
    state = {"gpu": 0, "swap_gb": 1.0}
    large = waiting_job(name="large", memory_gb=8.0, priority=5, added=2.0)
    small = waiting_job(name="small", memory_gb=2.0, priority=1, added=1.0)
    same = waiting_job(name="same", memory_gb=2.0, priority=5, added=3.0)
    running = waiting_job(name="running", memory_gb=1.0, state="running")
    told = {job["name"]: reason for job, reason in job_board.decisions([running, small, large, same], state, 3.0)}
    assert "needs 8.0 GB" in told["large"] and "needs 2.0 GB" in told["small"]
    assert told["same"] is None
    told = {job["name"]: reason for job, reason in job_board.decisions([running, small, large], state, 12.0)}
    assert told["large"] is None and told["small"] is None
    held = large | {"held": 1}
    assert {job["name"]: reason for job, reason in job_board.decisions([running, small, held], state, 3.0)}[
        "small"
    ] is None


def test_a_running_job_keeps_its_promise_only_until_its_peak_reaches_it():
    climbing = waiting_job(state="running", memory_gb=5.0, peak_gb=1.0, pid=0)
    past_it = waiting_job(state="running", memory_gb=5.0, peak_gb=5.2, pid=0)
    assert job_board.room_left([climbing], {"room_gb": 6.0}) == 2.0
    assert job_board.room_left([past_it], {"room_gb": 6.0}) == 6.0


def test_a_lower_job_takes_only_what_the_first_short_of_memory_will_not_need():
    state = {"gpu": 0, "swap_gb": 1.0}
    large = waiting_job(name="large", memory_gb=5.0, priority=5)
    behind = waiting_job(name="behind", memory_gb=6.0, priority=5, added=2.0)
    small = waiting_job(name="small", memory_gb=0.7, priority=1, added=3.0)
    too_big = waiting_job(name="too big", memory_gb=3.0, priority=1, added=4.0)
    running = waiting_job(name="running", memory_gb=4.0, priority=5, state="running")
    told = {
        job["name"]: reason for job, reason in job_board.decisions([running, small, too_big, large, behind], state, 3.0)
    }
    assert "needs 5.0 GB" in told["large"] and "needs 6.0 GB" in told["behind"]
    assert told["small"] is None and "needs 3.0 GB" in told["too big"]
    lower = waiting_job(name="lower", memory_gb=2.0, priority=1, state="running")
    told = {job["name"]: reason for job, reason in job_board.decisions([running, lower, small, large], state, 1.0)}
    assert "needs 0.7 GB" in told["small"]


def test_a_forecast_keeps_the_boards_order_and_leaves_out_what_is_held():
    typical = {"fetch": (4.0, 1.0), "read": (2.0, 2.0), "pieces": (6.0, 3.0)}
    fetches = [waiting_job(name=f"fetch-{n}", link=1, kind="fetch", added=n, owner="survey") for n in (1, 2)]
    reads = [
        waiting_job(
            name=f"read-{n}", engine=1, kind="read", memory_gb=2.0, added=2 + n, after=f'["fetch-{n}"]', owner="survey"
        )
        for n in (1, 2)
    ]
    held = waiting_job(name="held", held=1, added=5.0)
    behind = waiting_job(name="behind", after='["held"]', added=6.0)
    running = waiting_job(name="pieces-big", kind="pieces", memory_gb=6.0, state="running", started=940.0, owner="band")
    gate = waiting_job(name="gate", kind=job_board.GATE, memory_gb=0.05, state="running", started=0.0)
    last = waiting_job(name="last", after='["gate"]', added=7.0)
    started, ended, clock = job_board.forecast(
        [running, gate, *fetches, *reads, held, behind, last], typical, 3.0, 1000.0
    )
    assert (started["fetch-1"], ended["fetch-1"], started["fetch-2"]) == (0.0, 4.0, 4.0)
    assert (started["read-1"], ended["read-2"]) == (4.0, 10.0)
    assert ended["pieces-big"] == 11.0 and ended["gate"] == 11.0
    assert started["last"] == 11.0 and clock == 21.0
    assert "held" not in started and "behind" not in started


def test_swap_stops_the_lowest_priority_and_never_the_top_job(monkeypatch):
    stopped = []
    monkeypatch.setattr(job_board, "stop", lambda database, job, again, why: stopped.append(job["name"]))
    monkeypatch.setattr(job_board, "footprint_of", lambda job: job["memory_gb"])
    top = waiting_job(name="top", priority=100, memory_gb=12.0, state="running", restartable=1, adopted=0)
    big = waiting_job(name="big", priority=2, memory_gb=3.0, state="running", restartable=1, adopted=0)
    small = waiting_job(name="small", priority=2, memory_gb=1.0, state="running", restartable=1, adopted=0)
    fixed = waiting_job(name="fixed", priority=0, memory_gb=2.0, state="running", restartable=0, adopted=0)
    full = {"swap_gb": job_board.SWAP_STOP_GB + 1}
    job_board.relieve(None, [top, big, small, fixed], full)
    assert stopped == ["big"]
    stopped.clear()
    job_board.relieve(None, [top, fixed], full)
    job_board.relieve(None, [top], full)
    assert stopped == []


def test_swap_among_equal_priorities_stops_the_job_furthest_over_its_reservation(monkeypatch):
    stopped = []
    monkeypatch.setattr(job_board, "stop", lambda database, job, again, why: stopped.append((job["name"], why)))
    seen = {"honest": 3.0, "greedy": 43.6, "fixed": 30.0, "slight": 6.5}
    monkeypatch.setattr(job_board, "footprint_of", lambda job: seen[job["name"]])
    jobs = [
        waiting_job(name="honest", memory_gb=3.0, state="running", restartable=1, adopted=0, peak_gb=3.0),
        waiting_job(name="greedy", memory_gb=6.0, state="running", restartable=1, adopted=0, peak_gb=43.6),
        waiting_job(name="fixed", memory_gb=2.0, state="running", restartable=0, adopted=0, peak_gb=30.0),
        waiting_job(name="slight", memory_gb=6.0, state="running", restartable=1, adopted=0, peak_gb=6.5),
    ]
    full = {"swap_gb": job_board.SWAP_STOP_GB + 1}
    job_board.relieve(None, jobs, full)
    assert [name for name, _ in stopped] == ["greedy"] and "43.6 GB over its 6 GB" in stopped[0][1]
    stopped.clear()
    job_board.relieve(None, [jobs[0], jobs[2], jobs[3]], full)
    assert stopped == []


def test_a_job_that_must_try_later_frees_its_slot_and_returns(board):
    marker = board / "tries"
    code = (
        f"open({str(marker)!r}, 'a').write('x'); raise SystemExit(75 if len(open({str(marker)!r}).read()) == 1 else 0)"
    )
    board_run(board, "add", "later", "--memory", "0.1", "--link", "--", sys.executable, "-c", code)
    assert until(lambda: row(board, "later")["state"] == "done")
    assert len(marker.read_text()) == 2 and row(board, "later")["code"] == 0


def test_a_last_job_waits_for_what_can_still_run():
    state = {"gpu": 0, "swap_gb": 1.0}
    last = waiting_job(name="last", last=1, added=9.0)
    ahead = waiting_job(name="ahead", added=1.0)
    assert "1 jobs ahead, first ahead" in job_board.blocked(last, [ahead, last], state, 5.0)
    assert job_board.blocked(last, [ahead | {"state": "failed"}, last], state, 5.0) is None
    assert job_board.blocked(last, [ahead | {"held": 1}, last], state, 5.0) is None
    behind = waiting_job(name="behind", after='["ahead"]', added=2.0)
    assert job_board.blocked(last, [ahead | {"state": "failed"}, behind, last], state, 5.0) is None
    assert "first behind" in job_board.blocked(last, [ahead | {"state": "done"}, behind, last], state, 5.0)
    assert job_board.blocked(last, [waiting_job(name="later", added=20.0), last], state, 5.0) is None
    assert job_board.blocked(last, [waiting_job(name="other", last=1, added=2.0), last], state, 5.0) is None


def test_a_chain_runs_in_order_and_leaves_its_peak(board):
    held = "import time; a = bytearray(120_000_000); a[::4096] = b'x' * len(a[::4096]); time.sleep(3); print('first')"
    board_run(board, "add", "first", "--memory", "0.2", "--", sys.executable, "-c", held)
    board_run(
        board, "add", "second", "--memory", "0.1", "--after", "first", "--", sys.executable, "-c", "print('second')"
    )
    assert until(lambda: row(board, "second")["state"] == "done")
    first, second = row(board, "first"), row(board, "second")
    assert first["state"] == "done" and first["ended"] <= second["started"]
    assert 0.1 < first["peak_gb"] < 1.0
    assert "first" in board_run(board, "log", "first").stdout
    assert board_run(board, "wait", "second").returncode == 0


def test_a_forecast_answers_from_the_board(board):
    board_run(board, "add", "soon", "--held", "--", sys.executable, "-c", "pass")
    assert "queue should end" in board_run(board, "forecast").stdout
    assert "will not run" in board_run(board, "forecast", "soon").stdout


def test_a_slot_held_by_overlapping_jobs_counts_once():
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.execute("CREATE TABLE jobs(link INTEGER, started REAL, ended REAL)")
    database.executemany(
        "INSERT INTO jobs VALUES (1, ?, ?)", [(0.0, 20.0), (10.0, 30.0), (50.0, 60.0), (90.0, None), (0.0, 5.0)]
    )
    database.execute("INSERT INTO jobs VALUES (0, 30.0, 50.0)")
    assert job_board.share_held(database, "link", 10.0, 100.0) == (20.0 + 10.0 + 10.0) / 90.0


def test_settings_add_up_however_they_are_written(board):
    board_run(
        board, "add", "set", "--held", "--setting", "A=1", "B=2", "--setting", "C=3", "--", sys.executable, "-c", "pass"
    )
    assert json.loads(row(board, "set")["settings"]) == {"A": "1", "B": "2", "C": "3"}


def test_a_failure_holds_its_followers(board):
    board_run(board, "add", "broken", "--memory", "0.1", "--", sys.executable, "-c", "raise SystemExit(3)")
    board_run(board, "add", "follower", "--memory", "0.1", "--after", "broken", "--", sys.executable, "-c", "pass")
    assert until(lambda: row(board, "broken")["state"] == "failed")
    assert row(board, "broken")["code"] == 3
    time.sleep(1)
    assert row(board, "follower")["state"] == "waiting"
    assert "after broken, failed" in board_run(board, "status").stdout


def test_a_stopped_group_leaves_no_workers_and_restarts(board):
    marker, part = board / "tries", board / "unit.building"
    code = f"""
import multiprocessing, os, time
open({str(marker)!r}, 'a').write('x')
os.makedirs({str(part)!r}, exist_ok=True)
if len(open({str(marker)!r}).read()) == 1:
    worker = multiprocessing.get_context('spawn').Process(target=time.sleep, args=(120,))
    worker.start()
    open({str(board / "worker")!r}, 'w').write(str(worker.pid))
    time.sleep(120)
"""
    board_run(
        board,
        "add",
        "restarting",
        "--memory",
        "0.1",
        "--restartable",
        "--clean",
        str(part),
        "--",
        sys.executable,
        "-c",
        code,
    )
    assert until((board / "worker").exists)
    worker = int((board / "worker").read_text())
    job = row(board, "restarting")
    database = sqlite3.connect(board / "board.sqlite")
    database.execute("UPDATE jobs SET state='stopping', stopping=?, again=1 WHERE name='restarting'", (time.time(),))
    database.commit()
    os.killpg(job["pid"], signal.SIGTERM)
    assert until(lambda: row(board, "restarting")["state"] == "done")
    assert len(marker.read_text()) == 2 and row(board, "restarting")["tries"] == 2
    assert until(lambda: subprocess.run(["ps", "-p", str(worker)], capture_output=True, check=False).returncode != 0)


def test_a_job_that_ends_revives_a_dead_dispatcher(board):
    board_run(board, "add", "first", "--memory", "0.1", "--", "sleep", "2")
    assert until(lambda: row(board, "first")["state"] == "running")
    os.kill(int((board / "dispatcher.lock").read_text()), signal.SIGKILL)
    database = sqlite3.connect(board / "board.sqlite")
    database.execute(
        "INSERT INTO jobs(name, kind, command, folder, settings, memory_gb, cores, gpu, engine, link, priority, nice, background, after, needs, clean, restartable, state, added) "
        "VALUES ('second', 'sleep', '[\"sleep\", \"0\"]', ?, '{}', 0.1, 1, 0, 0, 0, 0, 0, 0, '[\"first\"]', '[]', '[]', 0, 'waiting', ?)",
        (str(HERE), time.time()),
    )
    database.commit()
    assert until(lambda: row(board, "second")["state"] == "done")


def test_cancel_and_memory_of_a_kind(board):
    board_run(board, "add", "long", "--memory", "0.1", "--kind", "sleeper", "--", "sleep", "60")
    assert until(lambda: row(board, "long")["state"] == "running")
    board_run(board, "cancel", "long")
    assert until(lambda: row(board, "long")["state"] == "stopped")
    board_run(board, "add", "held", "--memory", "0.1", "--held", "--", "sleep", "0")
    time.sleep(0.5)
    assert row(board, "held")["state"] == "waiting"
    board_run(board, "release", "held")
    assert until(lambda: row(board, "held")["state"] == "done")
    assert json.loads(row(board, "held")["command"]) == ["sleep", "0"]


def test_an_adopted_process_is_followed_and_never_signalled(board):
    other = subprocess.Popen(["sleep", "2"])
    board_run(board, "adopt", str(other.pid), "elsewhere", "--gpu")
    board_run(board, "add", "after-it", "--memory", "0.1", "--after", "elsewhere", "--", "sleep", "0")
    assert row(board, "after-it")["state"] == "waiting"
    assert other.wait() == 0
    assert until(lambda: row(board, "after-it")["state"] == "done")
    assert row(board, "elsewhere")["state"] == "done"


def test_a_claim_dies_with_its_process(tmp_path):
    unit = tmp_path / "band-0001-0013"
    code = f"""
import time
from runner.stage_queue import claimed
with claimed({str(unit)!r}) as mine:
    print('claimed' if mine else 'taken', flush=True)
    time.sleep(120)
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=HERE,
        env=os.environ | {"PYTHONPATH": str(HERE)},
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "claimed"
    with stage_queue.claimed(unit) as mine:
        assert not mine
    holder.kill()
    holder.wait()
    with stage_queue.claimed(unit) as mine:
        assert mine
    assert not list(tmp_path.glob("band-*")) and not list(tmp_path.iterdir())


def clients_text(display, compute):
    node = '+-o AGXDeviceUserClient  <class AGXDeviceUserClient>\n    {\n      "IOUserClientCreator" = "pid {pid}, {name}"\n      "AppUsage" = ({{"accumulatedGPUTime"={used},"API"="Metal"}})\n    }}\n'
    return "".join(
        node.replace("{pid}", str(pid))
        .replace("{name}", name)
        .replace("{used}", str(used))
        .replace("{{", "{")
        .replace("}}", "}")
        for pid, name, used in ((525, "WindowServer", display), (4242, "python3.11", compute))
    )


@pytest.mark.apple
def test_the_screen_drawing_is_not_someone_elses_gpu_work(monkeypatch):
    from runner.machine_apple import probes

    texts = iter([clients_text(0, 0), clients_text(4_000_000_000, 500_000_000)])
    clock = iter([100.0, 110.0])
    monkeypatch.setattr(probes, "text_of", lambda *command: next(texts))
    monkeypatch.setattr(probes, "_gpu_seen", {})
    monkeypatch.setattr(probes, "time", types.SimpleNamespace(time=lambda: next(clock), sleep=lambda seconds: None))
    assert probes.gpu_of_others(60) == 5


def test_the_sampler_measures_cpu_network_and_discs(monkeypatch):
    monkeypatch.setattr(job_board, "_counted", {})
    assert set(job_board.rates().values()) == {None}
    time.sleep(0.2)
    second = job_board.rates()
    assert 0 <= second["cpu_pct"] <= 100 and second["net_mb_s"] >= 0
    assert second["internal_mb_s"] is None if CUDA else second["internal_mb_s"] >= 0


def test_cores_in_use_follow_the_processes_cpu_ticks(monkeypatch):
    clock = iter([100.0, 102.0, 104.0, 106.0])
    monkeypatch.setattr(job_board.time, "time", lambda: next(clock))
    second = round(1e9 / job_board.TICK_NS)
    ticks, usage = {}, lambda user, system: [0, 0, user, system]
    cores, looked = job_board.cores_in_use({1: usage(0, 0)}, ticks, 0.0, 99.0)
    assert cores == 0.0 and looked == 100.0
    cores, looked = job_board.cores_in_use({1: usage(4 * second, 0), 2: usage(0, 2 * second)}, ticks, cores, looked)
    assert cores == pytest.approx(job_board.CORES_SMOOTHING * 3.0)
    cores, looked = job_board.cores_in_use({1: usage(4 * second, 0)}, ticks, cores, looked)
    assert cores == pytest.approx((1 - job_board.CORES_SMOOTHING) * job_board.CORES_SMOOTHING * 3.0)
    back, _ = job_board.cores_in_use({1: usage(4 * second, 0), 2: usage(2 * second, 0)}, ticks, cores, looked)
    assert back == pytest.approx((1 - job_board.CORES_SMOOTHING) * cores)
    monkeypatch.undo()
    state = {"gpu": 0, "swap_gb": 1.0}
    busy = waiting_job(name="busy", cores=0.5, cores_used=2.0, state="running")
    assert "cores" in job_board.blocked(waiting_job(cores=8.5), [busy], state, 5.0)
    assert job_board.blocked(waiting_job(cores=8.5), [busy | {"cores_used": 0.0}], state, 5.0) is None


def test_a_busy_job_is_seen_using_cores(board):
    spin = "import time\nend = time.time() + 10\nwhile time.time() < end: pass\n"
    board_run(board, "add", "busy", "--memory", "0.1", "--cores", "0.1", "--", sys.executable, "-c", spin)
    assert until(lambda: row(board, "busy")["cores_used"] > 0.2, 25)


def test_usage_reports_hours_by_owner(board):
    board_run(board, "add", "qol-sleep", "--owner", "qol", "--memory", "0.1", "--", "sleep", "0.3")
    assert until(lambda: row(board, "qol-sleep")["state"] == "done")
    board_run(board, "change", "qol-sleep", "--owner", "ml")
    report = board_run(board, "usage").stdout
    assert "ml" in report and "cores" in report and "share" not in report
