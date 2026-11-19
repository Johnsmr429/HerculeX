"""A build made on another machine becomes the scan's build and page; the one it replaces stays beside it as a dated
part page with its own layout, and a model's read of the new build sits beside it under the model's name."""

import json

from runner import place_build
from website.refresh_viewers import META


def page(folder, scan, title):
    folder.mkdir(parents=True)
    meta = {"scan": scan, "title": title, "layers": []}
    text = f"<script>const M={json.dumps(meta)},TOP=32; floor=kept.floor??128,scan=kept.scan??0.25;</script>"
    (folder / "index.html").write_text(text)


def build(folder, layout):
    (folder / "sheet").mkdir(parents=True)
    (folder / "sheet" / "layout.json").write_text(json.dumps(layout))
    (folder / "sheet" / "placed.json").write_text(json.dumps(layout))
    (folder / "sheet" / "review.json").write_text("{}")


def test_a_new_build_takes_the_scan_and_the_old_one_stays_beside_it(tmp_path, monkeypatch):
    outputs, staged = tmp_path / "outputs", tmp_path / "box"
    level, sheets, nodes = (
        outputs / "unwrapping" / "level",
        outputs / "website" / "sheets",
        outputs / "unwrapping" / "nodes",
    )
    for name, path in (("OUTPUTS", outputs), ("LEVEL_ROUTE", level), ("SHEETS", sheets), ("NODE_ROUTE", nodes)):
        monkeypatch.setattr(place_build, name, path)
    (nodes / "S" / "V" / "nodes").mkdir(parents=True)
    build(level / "S" / "V", {"pieces": ["old"]})
    page(sheets / "sheet-S-1keV-level", "level/S/V", "old build")
    build(staged / "unwrapping" / "level" / "S" / "V", {"pieces": ["new"]})
    page(staged / "website" / "sheets" / "sheet-S-1keV-level", "level/S/V", "new build")

    (kept,) = place_build.place(staged, "S")
    assert json.loads((level / "S" / "V" / "sheet" / "layout.json").read_text()) == {"pieces": ["new"]}
    assert (level / "S" / "V" / "nodes").resolve() == (nodes / "S" / "V" / "nodes").resolve()
    assert "new build" in (sheets / "sheet-S-1keV-level" / "index.html").read_text()
    archived = next((outputs / "archive").glob("*-build/unwrapping/level/S/V"))
    assert json.loads((archived / "sheet" / "layout.json").read_text()) == {"pieces": ["old"]}
    assert kept.name.startswith("sheet-S-1keV-level-") and (kept / "placed.json").is_file()
    assert json.loads(META.search((kept / "index.html").read_text())[1])["scan"] == f"part/{kept.name}"

    read = tmp_path / "read"
    build(read / "unwrapping" / "level" / "S" / "V", {"pieces": ["new"]})
    page(read / "website" / "sheets" / "sheet-S-1keV-level", "level/S/V", "new build read by m")
    (beside,) = place_build.place(read, "S", model="m")
    assert beside.name == "sheet-S-1keV-level-m" and (level / "S" / "V" / "sheet" / "review-m.json").is_file()
    assert "new build" in (sheets / "sheet-S-1keV-level" / "index.html").read_text()


def test_a_build_that_never_reached_a_page_is_archived_by_its_own_date(tmp_path, monkeypatch):
    outputs, staged = tmp_path / "outputs", tmp_path / "box"
    level, sheets = outputs / "unwrapping" / "level", outputs / "website" / "sheets"
    for name, path in (("OUTPUTS", outputs), ("LEVEL_ROUTE", level), ("SHEETS", sheets), ("NODE_ROUTE", tmp_path)):
        monkeypatch.setattr(place_build, name, path)
    (level / "S" / "V" / "pieces").mkdir(parents=True)
    sheets.mkdir(parents=True)
    build(staged / "unwrapping" / "level" / "S" / "V", {"pieces": ["new"]})
    page(staged / "website" / "sheets" / "sheet-S-1keV-level", "level/S/V", "new build")

    assert place_build.place(staged, "S") == []
    assert json.loads((level / "S" / "V" / "sheet" / "layout.json").read_text()) == {"pieces": ["new"]}
    assert next((outputs / "archive").glob("*-build/unwrapping/level/S/V/pieces")).is_dir()
