"""The cuda lane's unwrap takes every scroll through its stages in order, and a failed stage settles that scroll with its
reason while the others go on to their pages."""

import threading

from runner.machine_cuda import unwrap


class Place:
    gpus, reads, fits, threads = 2, 2, 3, 4
    read_cores, fit_cores = [0, 1], [2, 3, 4, 5]

    def describe(self):
        return "a test machine"


def test_scrolls_are_published_or_settled_with_their_reason(tmp_path, monkeypatch):
    units = {
        "A": ["rows-0000-0012", "rows-0011-0023"],
        "B": ["rows-0000-0012"],
        "C": ["rows-0000-0012", "rows-0011-0023"],
    }
    monkeypatch.setattr(unwrap, "LOGS", tmp_path)
    monkeypatch.setattr(unwrap.run_levels, "rows_of", lambda s: [tmp_path / s / "scan" / "nodes" / u for u in units[s]])
    monkeypatch.setattr(unwrap.run_levels, "share_nodes", lambda nodes: None)
    ran, lock = [], threading.Lock()

    def job(_self, name, _args, _cores, _env=None):
        with lock:
            ran.append(name)
        if name == "C-rows-0011-0023":
            return "failed"
        if name == "B-ink":
            return "failed"
        if name.endswith("-sheet"):
            (tmp_path / f"{name}.log").write_text(f"/sheets/sheet-{name[:-6]}-level/index.html: 7 pieces\n")
        return "ok"

    monkeypatch.setattr(unwrap.Unwrap, "job", job)
    run = unwrap.Unwrap([("A", None), ("B", None), ("C", None)], Place())
    assert run.run() == 1
    assert run.outcomes == {
        "A": "published sheet-A-level (7 pieces)",
        "B": "ink read failed",
        "C": "level fits failed: rows-0011-0023",
    }
    assert ran.index("A-sheets") < ran.index("B-sheets") < ran.index("C-sheets")
    assert {"A-curl", "C-curl"} <= set(ran) and "B-curl" not in ran
    assert ran.index("A-rows-0000-0012") < ran.index("A-pieces") < ran.index("A-ink") < ran.index("A-sheet")
    assert "C-pieces" not in ran and "C-ink" not in ran and "B-sheet" not in ran


def test_a_two_gpu_box_is_shared_as_on_the_rented_rtx_5090s(monkeypatch):
    monkeypatch.setattr(unwrap.os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)
    monkeypatch.setattr(unwrap, "quota_cores", lambda: None)
    monkeypatch.setattr(unwrap, "gpu_count", lambda: 2)
    monkeypatch.setattr(unwrap, "machine", lambda: {"room_gb": 240.0})
    place = unwrap.Machine()
    assert (place.reads, place.read_cores, place.fit_cores) == (2, list(range(16)), list(range(16, 64)))
    assert (place.fits, place.threads) == (16, 12)
    monkeypatch.setattr(unwrap, "quota_cores", lambda: 12)
    held = unwrap.Machine()
    assert (held.read_cores, held.fit_cores, held.fits) == (list(range(6)), list(range(6, 12)), 2)
