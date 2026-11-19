"""An archived page gets new controls while retaining its images and label identity."""

import json

from website.refresh_viewers import FLOOR, META, SCAN, refresh


def test_archived_viewers_keep_their_data(tmp_path):
    expected = {}
    for scan in ("level/PHerc0211/20250821151803", "part/sheet-PHerc0211-113keV-level-2026-11-13"):
        folder = tmp_path / scan.replace("/", "-")
        folder.mkdir()
        expected[folder.name] = scan
        meta = {
            "scan": scan,
            "title": "6 November",
            "layers": [{"path": "taught", "stamp": 123, "mm": 0.009, "levels": 8}],
            "tile": 512,
            "area": {"x": 0, "y": 0, "w": 10, "h": 20},
            "shelves": [],
            "places": [],
            "boxes": [],
            "back": "../index.html",
        }
        (folder / "index.html").write_text(
            "const M=" + json.dumps(meta) + ",TOP=32; floor=kept.floor??128,scan=kept.scan??0.25;"
        )
        (folder / "tile.jpg").write_bytes(b"original pixels")
        (folder / "marks.jsonl").write_text("saved labels\n")
    assert len(refresh(tmp_path)) == 2
    for folder in tmp_path.iterdir():
        html = (folder / "index.html").read_text()
        kept = json.loads(META.search(html)[1])
        assert kept["layers"] == meta["layers"]
        assert kept["scan"] == expected[folder.name]
        assert kept["title"] == "6 November"
        assert 'id="labelopen"' in html and 'class="menu right"' in html
        assert FLOOR.search(html)[1] == "128" and SCAN.search(html)[1] == "0.25"
        assert (folder / "tile.jpg").read_bytes() == b"original pixels"
        assert (folder / "marks.jsonl").read_text() == "saved labels\n"
    assert refresh(tmp_path) == []
