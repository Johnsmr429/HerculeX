"""Obsolete saved geometry stays visible in the catalogue without preventing current circles from being located."""

import json

import numpy as np
import tifffile

from spots import catalogue, database, sources


def surface(folder, shape=(5, 5)):
    """Write a small planar surface with positive scan coordinates."""
    folder.mkdir(parents=True, exist_ok=True)
    y, x = np.indices(shape, dtype=np.float32)
    for axis, values in zip("xyz", (100 + x, 200 + y, np.full(shape, 300, np.float32))):
        tifffile.imwrite(folder / f"{axis}.tif", values)
    (folder / "meta.json").write_text(json.dumps({"width_mm": 4, "area_mm2": 16, "winding": 1}))


def fixture(tmp_path, monkeypatch):
    """Make two marked regions, one missing foreground piece and one intact surface."""
    monkeypatch.setattr(sources, "HAND_MARKS", tmp_path / "marks")
    monkeypatch.setattr(database, "DB", tmp_path / "db")
    sheet = sources.Sheet("test", "PHercTest", "scan", "level", "page", tmp_path / "pieces")
    monkeypatch.setattr(catalogue, "all_sheets", lambda: iter([sheet]))
    monkeypatch.setattr(catalogue, "logs", lambda: [sheet.key])
    monkeypatch.setattr(catalogue, "TARGETS_SINCE", "")
    marks = [
        {
            "id": name,
            "kind": "spot",
            "layout": "saved",
            "time": "2026-11-17T01:00:00Z",
            "points": [[x, 0], [x + 4, 0], [x + 4, 4], [x, 4]],
        }
        for name, x in (("missing1", 0), ("current2", 100))
    ]
    sheet.log.parent.mkdir(parents=True)
    sheet.log.write_text("".join(json.dumps(m) + "\n" for m in marks))
    base = {
        "unit": "rows",
        "shape": [5, 5],
        "stride": 1,
        "um": 1.0,
        "y_mm": 0,
        "width_mm": 4,
        "height_mm": 4,
        "area": 16,
        "flip_rows": False,
        "flip_cols": False,
    }
    placements = [base | {"piece": p, "x_mm": x} for p, x in ((1, 0), (2, 100), (3, 0))]
    layouts = sheet.log.parent / "layouts"
    layouts.mkdir()
    (layouts / "saved.json").write_text(json.dumps({"pieces": placements}))
    surface(sheet.pieces / "rows" / "piece-0002")
    surface(sheet.pieces / "rows" / "piece-0003")
    return sheet


def test_missing_surface_does_not_abort_or_claim_a_different_sheet(tmp_path, monkeypatch):
    sheet = fixture(tmp_path, monkeypatch)
    unchanged = sheet.log.read_bytes()
    rows, gone = catalogue.build()
    missing, current = {r["spot_id"]: r for r in rows}.values()
    assert not gone and len(database.rows("spots")) == 2
    assert not missing["location_available"] and not missing["target"] and not missing["layout_matches"]
    assert "piece-0001" in missing["location_error"]
    assert missing["pieces"] == [] and missing["cells_covered"] == 0
    assert np.isnan(missing["centre_xyz"]).all()
    assert current["location_available"] and current["target"] and current["cells_covered"] == 16
    assert current["voxel_um"] == 1000 and current["node_mm"] == 1
    assert sheet.log.read_bytes() == unchanged
    surface(sheet.pieces / "rows" / "piece-0001")
    recovered, _ = catalogue.build()
    assert all(r["location_available"] and r["target"] and not r["location_error"] for r in recovered)
    assert sheet.log.read_bytes() == unchanged


def test_resized_piece_is_unavailable_before_array_broadcast(tmp_path, monkeypatch):
    sheet = fixture(tmp_path, monkeypatch)
    surface(sheet.pieces / "rows" / "piece-0001", (4, 5))
    rows, _ = catalogue.build()
    assert not rows[0]["location_available"]
    assert "differs from saved layout" in rows[0]["location_error"]
    assert rows[0]["pieces"] == [] and rows[1]["location_available"]


def test_missing_layout_preserves_both_standing_marks(tmp_path, monkeypatch):
    sheet = fixture(tmp_path, monkeypatch)
    (sheet.log.parent / "layouts" / "saved.json").unlink()
    rows, _ = catalogue.build()
    assert len(rows) == 2 and not any(r["location_available"] for r in rows)
    assert all("saved.json" in r["location_error"] and not r["target"] for r in rows)
