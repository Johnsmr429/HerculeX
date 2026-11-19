import json
import re

import numpy as np
from PIL import Image

from website import panel_pages


def meta_of(page):
    return json.loads(re.search(r"const M\s*=\s*(\{.*?\}),\s*main\s*=", page, re.DOTALL)[1].replace("<\\/", "</"))


def test_places_and_pictures_keep_their_order_in_the_site_look(tmp_path):
    for place, names in {
        "rows-0000-0012-piece-0139": ["0-scan", "1-site", "10-model-depth-mean"],
        "b-place": ["2-x"],
    }.items():
        (tmp_path / place).mkdir()
        for name in names:
            Image.fromarray(np.zeros((4, 4), np.uint8)).save(tmp_path / place / f"{name}.png")
    (tmp_path / "index.html").write_text("<!doctype html><title>Old &amp; title</title><h1>x</h1>")
    panel_pages.write(tmp_path, panel_pages.title_of(tmp_path))
    page = (tmp_path / "index.html").read_text()
    meta = meta_of(page)
    assert meta["title"] == "Old & title"
    assert [p["name"] for p in meta["places"]] == ["b-place", "rows-0000-0012-piece-0139"]
    assert [p["title"] for p in meta["places"][1]["pictures"]] == ["scan", "site", "model depth mean"]
    assert meta["places"][1]["pictures"][2]["src"] == "rows-0000-0012-piece-0139/10-model-depth-mean.png"
    assert re.search(r"background:\s*#000", page) and re.search(r"font:\s*10pt Calibri,\s*Carlito,\s*sans-serif", page)
    assert "<h1" not in page and "<p" not in page
