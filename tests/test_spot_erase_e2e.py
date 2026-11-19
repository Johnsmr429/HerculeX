"""A circle erased in the viewer leaves the spots database at the next catalogue build."""

import json

import numpy as np
import tifffile

from spots import catalogue, database, sources
from website import site_server

SCAN = "part/page"


def sheet_with_one_piece(tmp_path, monkeypatch):
    """A viewer page laid out from one flat 10 x 10 mm piece, with its marks and database in tmp_path."""
    for module in (site_server, sources):
        monkeypatch.setattr(module, "HAND_MARKS", tmp_path / "marks")
    monkeypatch.setattr(site_server, "SHEETS", tmp_path / "sheets")
    monkeypatch.setattr(database, "DB", tmp_path / "db")
    sheet = sources.Sheet("parts/page", "PHercTest", "scan", "level", "page", tmp_path / "pieces")
    monkeypatch.setattr(catalogue, "all_sheets", lambda: iter([sheet]))
    monkeypatch.setattr(catalogue, "logs", lambda: [sheet.key])
    monkeypatch.setattr(catalogue, "TARGETS_SINCE", "")
    piece = {
        "unit": "rows",
        "piece": 1,
        "shape": [11, 11],
        "stride": 1,
        "um": 1.0,
        "x_mm": 0,
        "y_mm": 0,
        "width_mm": 10,
        "height_mm": 10,
        "area": 100,
        "flip_rows": False,
        "flip_cols": False,
    }
    (tmp_path / "sheets" / "page").mkdir(parents=True)
    (tmp_path / "sheets" / "page" / "placed.json").write_text(json.dumps({"pieces": [piece]}))
    folder = sheet.pieces / "rows" / "piece-0001"
    folder.mkdir(parents=True)
    y, x = np.indices((11, 11), dtype=np.float32)
    for axis, values in zip("xyz", (100 + x, 200 + y, np.full((11, 11), 300, np.float32)), strict=True):
        tifffile.imwrite(folder / f"{axis}.tif", values)
    (folder / "meta.json").write_text(json.dumps({"width_mm": 10, "area_mm2": 100, "winding": 1}))


def circle(left):
    """A Spot circle drawn in the viewer, as its POST reaches the marks store; returns the mark's id."""
    points = [[left, 0], [left + 4, 0], [left + 4, 4], [left, 4]]
    return site_server.saved({"scan": SCAN, "kind": "spot", "width_mm": 0.3, "layer": "ink", "points": points})["id"]


def erase(mark_id):
    site_server.saved({"scan": SCAN, "remove": mark_id})


def test_an_erased_circle_leaves_the_database(tmp_path, monkeypatch):
    sheet_with_one_piece(tmp_path, monkeypatch)
    first, second = circle(0), circle(6)
    rows, gone = catalogue.build()
    assert sorted(r["mark_id"] for r in rows) == sorted([first, second]) and not gone
    spot = first[:8]
    database.write("surfaces", [{"spot_id": spot, "surface": "original"}])
    database.write("reads", [{"spot_id": spot, "surface": "original", "model": "m", "side": "inward", "depth": 0}])
    database.write("verdicts", [{"spot_id": spot, "surface": "original", "judged_at": "now", "readable": "no"}])

    erase(first)
    rows, gone = catalogue.build()
    assert gone == [spot]
    assert [r["mark_id"] for r in rows] == [r["mark_id"] for r in database.rows("spots")] == [second]
    assert database.rows("surfaces") == [] and database.rows("reads") == []
    assert [v["spot_id"] for v in database.rows("verdicts")] == [spot]

    erase(second)
    rows, gone = catalogue.build()
    assert rows == [] and gone == [second[:8]] and database.rows("spots") == []
