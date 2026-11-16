"""The board's rules on made-up surveys."""

import json

import numpy as np

from calibration.survey import scroll_leaderboard as leaderboard
from calibration.survey import scroll_survey as survey
from calibration.survey import survey_row_power as row_power

ZERO = [{"class": name, "excess": 0.01, "log_ratio": 0.0, "cells": 1000} for name in ("blank", "text", "all")]


def written(folder, ratios, excess, open_share=0.5):
    """Pieces with the given statistics, their windows open to the given share."""
    folder.mkdir(parents=True)
    for k, (ratio, gain) in enumerate(zip(ratios, excess)):
        row = {
            "piece": f"piece-{k:04d}",
            "ruler": survey.RULER,
            "pixels": 1000,
            "forward": 100 + gain * 1000,
            "reversed": 100.0,
            "band_forward": np.exp(2 * ratio),
            "band_reversed": 1.0,
            "faces_open": {"both": [800, 100 + gain * 800, 100]},
        }
        (folder / f"piece-{k:04d}.json").write_text(json.dumps(row))
        sheet = np.ones((10, 10), dtype=np.float16)
        opened = (np.arange(100).reshape(10, 10) < open_share * 100).astype(np.float16)
        np.savez(folder / f"piece-{k:04d}.npz", sheet0=sheet, open0=opened)


def spread(centre, count=14, seed=0):
    return centre + np.random.default_rng(seed).normal(0, 0.05, count)


def surveyed(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "first" / "replicate"
    monkeypatch.setattr(leaderboard, "SURVEYED", first)
    monkeypatch.setattr(leaderboard, "REPLICATED", second)
    monkeypatch.setattr(leaderboard, "SECOND_MODEL", tmp_path / "second-model")
    monkeypatch.setattr(leaderboard, "NULL_RECORD", tmp_path / "null.json")
    monkeypatch.setattr(leaderboard, "BOARD", tmp_path / "board.json")
    monkeypatch.setattr(leaderboard, "ROW_RECORD", tmp_path / "rows.json")
    monkeypatch.setattr(leaderboard, "RIM_RECORD", tmp_path / "rims.json")
    monkeypatch.setattr(leaderboard, "ELIGIBLE", {f"PHerc000{k}": "1/band" for k in range(1, 6)})
    monkeypatch.setattr(leaderboard, "CONTROLS", {"PHerc0841": "1/band"})
    monkeypatch.setattr(leaderboard, "load", list)
    monkeypatch.setattr(leaderboard, "survey_volumes", lambda rows: [])
    monkeypatch.setattr(leaderboard, "scan_of", lambda item, volumes: {"pieces_4cm2": 3})
    (tmp_path / "null.json").write_text(json.dumps({"rows": ZERO}))
    written(first / "PHerc0001", spread(0.3), spread(0.08))
    written(second / "PHerc0001", spread(0.3, seed=1), spread(0.08, seed=1))
    written(first / "PHerc0002", spread(0.6, seed=2), spread(0.1, seed=2))
    written(second / "PHerc0002", spread(0.05, seed=3), spread(0.01, seed=3))
    written(first / "PHerc0003", spread(0.3, seed=4), spread(0.08, seed=4))
    written(second / "PHerc0003", spread(0.3, seed=5), spread(0.08, seed=5), open_share=0.1)
    written(first / "PHerc0004", spread(0.05, seed=6), spread(0.01, seed=6))
    written(first / "PHerc0841", spread(0.25, seed=7), spread(0.06, seed=7))
    return leaderboard.board()


def test_the_board_follows_the_protocol(tmp_path, monkeypatch):
    out = surveyed(tmp_path, monkeypatch)
    rows = {r["sample"]: r for r in out["prize"]}
    order = [r["sample"] for r in out["prize"]]
    assert rows["PHerc0001"]["confirmed"] and rows["PHerc0001"]["target"]
    assert rows["PHerc0001"]["both_faces"]["mean"] > rows["PHerc0004"]["both_faces"]["mean"]
    assert rows["PHerc0002"]["confirmed"] is False and not rows["PHerc0002"]["target"]
    assert rows["PHerc0003"]["confirmed"] and not rows["PHerc0003"]["target"]
    assert rows["PHerc0004"]["confirmed"] is None
    assert set(order[:2]) == {"PHerc0001", "PHerc0003"} and order.index("PHerc0002") > 1
    assert [r["sample"] for r in out["controls"]] == ["PHerc0841"] and "PHerc0841" not in rows
    assert out["unread"] == ["PHerc0005"]
    board = leaderboard.table(out)
    assert "| 1 | PHerc000" in board and "PHerc0841" not in board


def test_a_sliver_of_open_sheet_cannot_swing_a_scroll():
    rng = np.random.default_rng(0)
    excess, pixels = np.array([0.05] * 9 + [0.9]), np.array([1000.0] * 9 + [10.0])
    assert abs(leaderboard.summary_of(excess, rng, pixels)["mean"] - 0.05) < 0.002
    assert leaderboard.summary_of(excess, rng)["mean"] > 0.13


def test_shrinkage_pulls_a_noisy_scroll_harder_than_a_sure_one():
    means, errors = leaderboard.shrunken([0.4, 0.4, 0.0, 0.0, 0.1], [0.2, 0.02, 0.02, 0.02, 0.02])
    assert means[0] < means[1] and errors[0] > errors[1]
    assert abs(means[1] - 0.4) < 0.02


def test_a_row_of_an_older_ruler_never_reaches_the_board(tmp_path):
    written(tmp_path / "stage", [0.3, 0.3], [0.08, 0.08])
    path = tmp_path / "stage" / "piece-0001.json"
    row = json.loads(path.read_text())
    del row["ruler"]
    path.write_text(json.dumps(row))
    assert [r["piece"] for r in leaderboard.pieces_in(tmp_path / "stage")] == ["piece-0000"]


def test_a_crop_cannot_make_an_excess(tmp_path, monkeypatch):
    class Reader:
        def read(self, stack, fixed=False):
            return np.full(stack.shape[1:], float(stack.shape[1] > survey.RULER_SIDE), dtype=np.float32)

    monkeypatch.setattr(survey, "_reader", Reader())
    monkeypatch.setattr(survey, "open_columns", lambda part: np.ones(part.shape[1:], dtype=bool))
    face = np.ones((5, survey.RULER_SIDE + 300, survey.RULER_SIDE + 300), dtype=np.uint8)
    row = survey.ruler_row(face, face[::-1], Reader().read(face), 0.009, tmp_path / "piece")
    assert row["pixels"] > 0 and row["forward"] == row["reversed"] == row["pixels"]
    assert row["faces_open"]["both"] == [row["pixels"]] * 3 and row["faces_open"]["neither"] == [0, 0, 0]
    assert (tmp_path / "reversed-piece.png").exists()


def test_rows_of_writing_stand_above_clutter():
    rng = np.random.default_rng(0)
    side = 3000
    clutter = rng.integers(1, 255, (side // 8, side // 8)).astype(np.uint8).repeat(8, 0).repeat(8, 1)
    rows = (128 + 100 * np.sin(2 * np.pi * np.arange(side) * 0.009362 / 5.0)).astype(np.uint8)[:, None].repeat(side, 1)
    written = np.maximum(rows, 1)
    assert row_power.row_ratio(written, 0.009362)["ratio"] > 10
    assert 0.3 < row_power.row_ratio(clutter, 0.009362)["ratio"] < 3
    assert row_power.row_ratio(np.ones((1000, 1000), dtype=np.uint8), 0.009362) is None
    framed = np.pad(written, 300)
    assert row_power.row_ratio(framed, 0.009362, row_power.NO_RIM_WINDOW_MM, row_power.NO_RIM_MM)["ratio"] > 10


def test_writing_shows_one_way_and_structure_both():
    rng = np.random.default_rng(0)
    written = [{"ratio": 3.0 + k / 10, "reversed": {"ratio": 1.0}} for k in range(10)]
    folds = [{"ratio": 3.0 + k / 10, "reversed": {"ratio": 3.0 + (9 - k) / 10}} for k in range(10)]
    assert leaderboard.paired_rows(written, rng)["chance"] < 0.01
    assert leaderboard.paired_rows(folds, rng)["chance"] > 0.3
    assert leaderboard.paired_rows(written[:2], rng) is None
