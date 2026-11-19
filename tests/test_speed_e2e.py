"""The benchmark runner: a result is recorded with its context, and one slower than its recent best is flagged."""

import json
import sys

import pytest

from runner import speed


def test_a_slower_run_is_flagged_against_the_best_of_the_last_five(tmp_path, monkeypatch, capsys):
    taken = iter([1.0, 1.0, 1.3])
    monkeypatch.setattr(speed, "RECORDS", tmp_path / "speed.jsonl")
    monkeypatch.setattr(speed, "BENCHMARKS", {"probe": (lambda: {"seconds": next(taken), "size": 3}, 1)})
    monkeypatch.setattr(speed, "context", lambda: {"commit": "abc", "dirty": False, "load": 0.0, "beside": []})
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", ["speed", "probe", "--record", "--compare"])
        speed.main()
    monkeypatch.setattr(sys, "argv", ["speed", "probe", "--record", "--compare"])
    with pytest.raises(SystemExit, match="slower than recorded: probe"):
        speed.main()
    rows = [json.loads(line) for line in (tmp_path / "speed.jsonl").read_text().splitlines()]
    assert [r["seconds"] for r in rows] == [1.0, 1.0, 1.3] and rows[0]["commit"] == "abc" and rows[0]["size"] == 3
    assert "SLOWER by 30%" in capsys.readouterr().out


def test_the_solver_benchmark_solves(monkeypatch):
    result = speed.solver()
    assert result["terms"] == 1_000_000 and result["seconds"] > 0
