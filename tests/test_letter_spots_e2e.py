"""Letter spots are found where a piece's read shows strokes, not at a flood, at speckle or off the sheet."""

import json

import numpy as np
from PIL import Image

from ink.datasets import letter_spots as spots

SIDE, UM = 2304, 9.362
BARS = ((200, 500, 150, 190), (200, 500, 300, 340), (200, 500, 450, 490))


def piece(folder):
    """One piece's scan image and read: three strokes, a flood, speckle, and a strip off the sheet."""
    ink, scan = np.zeros((SIDE, SIDE), dtype=np.uint8), np.full((SIDE, SIDE), 200, dtype=np.uint8)
    scan[:, 1900:] = 0
    for y0, y1, x0, x1 in BARS:
        ink[y0:y1, x0:x1] = 255
    y, x = np.ogrid[:SIDE, :SIDE]
    ink[(y - 1500) ** 2 + (x - 1000) ** 2 <= 250**2] = 255
    for at in range(0, 700, 70):
        ink[1600 + at : 1620 + at, 100 + at % 500 : 120 + at % 500] = 255
    faces = folder / "level_route" / "PHercTest" / "scan" / "papyrus-faces" / "level-0" / "rows-0000-0012"
    faces.mkdir(parents=True)
    Image.fromarray(ink).save(faces / "piece-0007-taught.png")
    Image.fromarray(scan).save(faces / "piece-0007.png")
    record = folder / "level_route" / "PHercTest" / "scan" / "pieces" / "rows-0000-0012" / "piece-0007"
    record.mkdir(parents=True)
    (record / "meta.json").write_text(json.dumps({"voxel_um": UM}))


def test_letter_spots_are_strokes_on_the_sheet(tmp_path, monkeypatch):
    piece(tmp_path)
    monkeypatch.setattr(spots, "ROUTE_OUTPUTS", tmp_path / "level_route")
    monkeypatch.setattr(spots, "LETTER_SPOTS", tmp_path / "letter_spots")
    monkeypatch.setattr(spots, "WORKERS", 1)
    spots.candidates("PHercTest", "scan")
    found = json.loads((tmp_path / "letter_spots" / "level_route" / "PHercTest" / "candidates.json").read_text())
    assert found["pieces"] == 1 and 1 <= found["windows_found"] <= 3
    (window,) = found["windows"]
    assert (window["unit"], window["piece"], window["y"], window["x"]) == ("rows-0000-0012", 7, 0, 0)
    strokes = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in BARS) / spots.SIDE**2
    assert abs(window["strokes"] - strokes) < 0.01 and window["wide"] == 0.0
