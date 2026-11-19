"""End to end: a sheet composed into a tile store, read back by the store and by byte range from the site server."""

import functools
import io
import json
import shutil
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

from ink.model import models
from runner import review_page, sheet_stage
from storage.tile_store import INDEX, TileStore
from website import sheet_tiles, site_server
from website.sheet_tiles import TILE, shown
from website.site_pages import crop

PIXEL_MM = 0.01


def laid_piece(folder, n, pixels, x_mm):
    """A piece's scan image on disc and its place on the sheet."""
    unit = folder / "unit"
    unit.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-layer0"):
        Image.fromarray(pixels).save(unit / f"piece-{n:04d}{suffix}.png")
    h, w = pixels.shape
    return {
        "unit": "unit",
        "piece": n,
        "area": float(h * w),
        "um": PIXEL_MM,
        "stride": 1,
        "x_mm": x_mm,
        "y_mm": 0.02,
        "width_mm": w * PIXEL_MM,
        "height_mm": h * PIXEL_MM,
        "flip_rows": False,
        "flip_cols": False,
        "group": n,
    }


def test_a_sheet_keeps_its_voxels_in_one_pack_and_serves_them_by_range(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    base, sheet = tmp_path / "scroll" / "scan", tmp_path / "sheets" / "sheet"
    faces = base / "faces" / "level-0"
    first = rng.integers(40, 200, (700, 900), dtype=np.uint8)
    second = rng.integers(40, 200, (300, 300), dtype=np.uint8)
    pieces = [laid_piece(faces, 0, first, 0.02), laid_piece(faces, 1, second, 2 * TILE * PIXEL_MM + 1.0)]
    layout = base / "sheet" / "layout.json"
    layout.parent.mkdir()
    layout.write_text(json.dumps({"pieces": pieces}))
    monkeypatch.setattr(sheet_tiles, "ROUTE_OUTPUTS", tmp_path)
    monkeypatch.setattr(sheet_tiles, "SHEETS", sheet.parent)
    monkeypatch.setattr(sheet_tiles, "PROCESSES", 2)
    sheet_tiles.run("scroll", "scan", PIXEL_MM, sheet.name, ("sheet", "layer0", "layer-3"))
    record = json.loads((sheet / "layers.json").read_text())[0]
    store = TileStore(sheet)
    assert sorted(p.name for p in sheet.iterdir()) == sorted([INDEX, "layers.json", store.pack.name])
    assert (
        set(store.layers) == {"sheet", "layer0"}
        and record["raw"]
        and record["levels"] == len(store.layers["sheet"]["grid"])
    )

    whole = np.zeros((2 * TILE, 4 * TILE), dtype=np.uint8)
    for c in range(4):
        for r in range(2):
            tile = store.image("sheet", 0, c, r)
            if tile is not None:
                whole[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile
    assert (whole[:700, :900] == first).all() and not whole[:, 900 : 2 * TILE].any()
    assert store.tile("sheet", 0, 1, 1)[:4] == b"\x89PNG" and store.tile("sheet", 1, 0, 0)[:2] == b"\xff\xd8"
    assert store.tile("sheet", 0, 2, 1) is None and store.tile("sheet", 0, 9, 9) is None
    assert store.place("layer0", 0, 0, 0) == store.place("sheet", 0, 0, 0)
    assert store.pack.stat().st_size < 1.2 * sum(length for *_, length in store.tiles("sheet"))

    box = {"x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0}
    seen = np.asarray(crop(sheet, record, box))
    assert (seen == shown(first[100:300, 100:300], record)).all()

    monkeypatch.setattr(site_server, "SHEETS", tmp_path / "sheets")
    monkeypatch.setitem(site_server.MOUNTS, "/sheets/", tmp_path / "sheets")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(site_server.Site, directory=site_server.HERE))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        offset, length = store.place("sheet", 0, 0, 0)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/sheets/sheet/{store.pack.name}?v=1",
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
        )
        with urllib.request.urlopen(request) as answer:
            body = answer.read()
            assert answer.status == 206 and answer.headers["Content-Range"].startswith(f"bytes {offset}-")
        assert body == store.tile("sheet", 0, 0, 0)
        assert (np.asarray(Image.open(io.BytesIO(body))) == store.image("sheet", 0, 0, 0)).all()
    finally:
        server.shutdown()


def test_stock_review_preserves_the_written_face_and_removes_duplicate_images(tmp_path, monkeypatch):
    sample, scan = "PHerc0191", "20250821151635"
    route, sheets = tmp_path / "level", tmp_path / "sheets"
    base = route / sample / scan
    faces, layout = base / "faces" / "level-0", base / "sheet"
    layout.mkdir(parents=True)
    pixels = np.random.default_rng(7).integers(60, 180, (700, 900), dtype=np.uint8)
    piece = laid_piece(faces, 0, pixels, 0.02)
    (layout / "layout.json").write_text(json.dumps({"pieces": [piece]}))
    stem = faces / "unit" / "piece-0000"
    Image.fromarray(pixels).save(stem.with_name(stem.name + "-taught.png"))
    side = stem.with_name(stem.name + "-taught-side.json")
    side.write_text('{"written":"far","near":0.0,"far":0.0}')
    coverage = {"pieces": 1, "ink_pieces": 1, "voxel_um": 10, "ink_cm2": 1, "laid_cm2": 1}
    monkeypatch.setattr(review_page.piece_reading, "MODEL", models.models()["reader-v3"])
    monkeypatch.setattr(review_page.piece_reading, "digest", lambda: "stock")
    monkeypatch.setattr(review_page.piece_reading, "queue_of", lambda *_: [])
    monkeypatch.setattr(review_page.survey_status, "snapshot", lambda *_, **__: coverage)
    monkeypatch.setattr(review_page, "ROUTE_OUTPUTS", route)
    monkeypatch.setattr(review_page, "SHEETS", sheets)
    monkeypatch.setattr(review_page.site_pages, "ROUTE_OUTPUTS", route)
    monkeypatch.setattr(review_page.site_pages, "LEVEL_ROUTE", route)
    monkeypatch.setattr(review_page.site_pages, "SHEETS", sheets)
    monkeypatch.setattr(review_page.site_pages, "scans", lambda *_: [(sample, scan, sample, sample)])
    monkeypatch.setattr(review_page.site_pages, "card", lambda _name, _place, out, *_: out.touch())
    monkeypatch.setattr(sheet_tiles, "ROUTE_OUTPUTS", route)
    monkeypatch.setattr(sheet_tiles, "SHEETS", sheets)
    monkeypatch.setattr(sheet_tiles, "PROCESSES", 1)
    monkeypatch.setattr(site_server, "LEVEL_ROUTE", route)
    monkeypatch.setattr(site_server, "HAND_MARKS", tmp_path / "marks")
    monkeypatch.setitem(site_server.MOUNTS, "/sheets/", sheets)

    def compose(_module, sample, scan, mm, name, layers, *_):
        sheet_tiles.run(
            sample,
            scan,
            float(mm),
            name,
            layers.split(","),
            model=review_page.piece_reading.MODEL.title,
            ink_name="ink",
            written_face=True,
        )

    monkeypatch.setattr(sheet_stage, "step", compose)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(site_server.Site, directory=site_server.HERE))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        review_page.publish(sample, scan, url)
        review_page.publish(sample, scan, url)
    finally:
        server.shutdown()
    assert not list(stem.parent.glob("*.png"))
    assert side.is_file()
    placed = json.loads((layout / "placed.json").read_text())
    assert placed["pieces"][0]["flip_cols"]
    assert placed == json.loads((layout / "placed-taught.json").read_text())
    review = sheets / f"sheet-{sample}"
    assert set(TileStore(review).layers) == {"sheet", "layer0", "taught"}
    assert len(list(review.glob("tiles-*.bin"))) == 1
    assert review_page.piece_reading.MODEL.title in (sheets / "index.html").read_text()

    other = f"sheet-{sample}-rebuilt"
    shutil.copytree(review, sheets / other)
    (sheets / other / "card.jpg").unlink()
    review_page.site_pages.add_card(other, surface="rebuilt")
    titles = {card["href"]: card["title"] for card in review_page.site_pages.front_cards()}
    assert titles[f"{other}/index.html"].startswith(f"{sample} · Reader v3 · rebuilt · ")
    assert f"{other}/index.html" in (sheets / "index.html").read_text() and (sheets / other / "card.jpg").is_file()
    review_page.site_pages.add_card(other, remove=True)
    assert f"{other}/index.html" not in (sheets / "index.html").read_text()


class InProcess:
    """A pool that maps in this process, so the test sees every decode."""

    @staticmethod
    def imap_unordered(function, jobs):
        return map(function, jobs)


def test_one_run_of_strips_decodes_each_image_once_and_paints_what_single_strips_paint(tmp_path, monkeypatch):
    from storage.tile_store import TileWriter

    rng = np.random.default_rng(5)
    faces = tmp_path / "faces"
    strip_mm = sheet_tiles.STRIP * TILE * PIXEL_MM
    pieces = []
    for n, x_mm in enumerate((0.1, strip_mm - 2.0, strip_mm + 1.0, 2 * strip_mm - 3.0, 2 * strip_mm + 0.5)):
        pixels = rng.integers(40, 200, (600, 700), dtype=np.uint8)
        pieces.append(laid_piece(faces, n, pixels, x_mm) | {"flip_cols": n % 2 == 1})
        Image.fromarray((pixels > 120).astype(np.uint8) * 200).save(faces / "unit" / f"piece-{n:04d}-taught.png")
    opened = []
    real_open = sheet_tiles.Image.open
    monkeypatch.setattr(sheet_tiles.Image, "open", lambda path, *a: opened.append(str(path)) or real_open(path, *a))
    monkeypatch.setattr(sheet_tiles, "SCRATCH", tmp_path)
    monkeypatch.setattr(sheet_tiles, "shown_range", lambda pieces, folder, layer: (0.0, 255.0))
    layers = [("taught", pieces, "taught"), ("layer0", pieces, "layer0"), ("sheet", pieces, "sheet")]

    def tiles(runs, name):
        monkeypatch.setattr(sheet_tiles, "PROCESSES", 1)
        monkeypatch.setattr(sheet_tiles, "RUNS_PER_PROCESS", runs)
        opened.clear()
        with TileWriter(tmp_path / name) as writer:
            sheet_tiles.compose_layers(layers, faces, writer, PIXEL_MM, InProcess())
        store = TileStore(tmp_path / name)
        with open(store.pack, "rb") as pack:
            found = {}
            for path in store.layers:
                for level, c, r, offset, length in store.tiles(path):
                    pack.seek(offset)
                    found[path, level, c, r] = pack.read(length)
        return found, [p for p in opened if "unit" in p]

    together, decoded = tiles(1, "one-run")
    apart, decoded_apart = tiles(1000, "single-strips")
    assert together == apart and len(together) > 20
    assert sorted(decoded) == sorted(set(decoded)) and len(decoded) == 3 * len(pieces) < len(decoded_apart)
