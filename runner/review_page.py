"""A read scan published as its whole-scroll page, with its card on the main page.

The pipeline's sheet stage ends here, so a finished unwrap appears on the main page (http://localhost:8765/sheets/)
with no further step: tiles of the scan at the mesh, the sheet's brightest voxels and the ink (layer0, sheet, taught),
the viewer page titled "SCROLL · MODEL · level · DATE", its card, and a receipt in the scan's sheet folder. When the
local site is running, the page's tiles and marks are checked over HTTP. The face PNGs the tiles came from are removed.

./run.sh runner.review_page SCROLL SCAN [--url http://127.0.0.1:8765]
"""

import argparse
import fcntl
import json
import os
import urllib.error
import urllib.request

from ink.model.weights import STOCK_READER
from ink.reading import piece_reading
from runner import sheet_stage
from storage.data_roots import FACES, PRODUCTION_FACES, ROUTE_OUTPUTS, SHEETS, stage_folder
from storage.tile_store import TileStore
from website import site_pages, survey_status

CHECKPOINT = STOCK_READER


def model_name(weights):
    """The ink model's name on the page: HERCULEX_MODEL_NAME, else Reader v3 for its checkpoint, else the file's name."""
    if named := os.environ.get("HERCULEX_MODEL_NAME"):
        return named
    return "Reader v3" if weights.resolve() == CHECKPOINT.resolve() else weights.stem


def live(url):
    """Whether the local site answers, so the published page can be checked over HTTP."""
    try:
        with urllib.request.urlopen(f"{url}/sheets/index.html", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


LAYERS = ("layer0", "sheet", "taught")


def check_page(folder, scan, url):
    """Check the pack and the routes the browser uses for tiles and marks."""
    store = TileStore(folder)
    with urllib.request.urlopen(f"{url}/sheets/{folder.name}/index.html", timeout=15) as response:
        if scan not in response.read().decode():
            raise ValueError(f"{folder.name}: viewer has no marks key for {scan}")
    with urllib.request.urlopen(f"{url}/marks?scan={scan}", timeout=15) as response:
        if not isinstance(json.load(response), list):
            raise ValueError(f"{scan}: invalid marks response")
    for layer in LAYERS:
        full = next((tile for tile in store.tiles(layer) if tile[0] == 0), None)
        if full is None:
            raise ValueError(f"{folder.name}: no full-resolution {layer} tiles")
        _, col, row, start, count = full
        request = urllib.request.Request(
            f"{url}/sheets/{folder.name}/{store.pack.name}",
            headers={"Range": f"bytes={start}-{start + count - 1}"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 206 or response.read() != store.tile(layer, 0, col, row):
                raise ValueError(f"{folder.name}: could not read a {layer} tile over HTTP")
    return store.pack.stat().st_size


def render_page(sheet, folder, title, scan):
    pieces = json.loads((sheet / "placed.json").read_text())["pieces"]
    shelves = site_pages.shelves_of_pieces(pieces)
    area = {"x": 0, "y": 0, "w": max(s["w"] for s in shelves), "h": shelves[-1]["y"] + shelves[-1]["h"]}
    places = site_pages.places_of(pieces)
    site_pages.page(
        folder,
        title,
        site_pages.layers_of(folder.name),
        area,
        "../index.html",
        scan,
        places=places,
        shelves=shelves,
        tint=site_pages.INK_TINT,
    )
    site_pages.card(folder.name, places[0], folder / "card.jpg", site_pages.OVERVIEW_LEVEL)


def publish(sample, volume, url):
    if FACES != PRODUCTION_FACES or piece_reading.UNITS:
        raise ValueError("whole-scroll review requires every unit in the scan's production face folder")
    weights = piece_reading.WEIGHTS
    model_label = model_name(weights)
    base = ROUTE_OUTPUTS / sample / volume
    sheet = stage_folder(base, "sheet")
    layout = sheet / "layout.json"
    scan = site_pages.scan_of(sample, volume)
    entry = next(((tag, title) for s, v, tag, title in site_pages.scans() if (s, v) == (sample, volume)), None)
    if entry is None:
        raise ValueError(f"{scan}: no layout to publish")
    tag, _ = entry
    name = f"sheet-{tag}"
    folder = SHEETS / name
    receipt = sheet / "review.json"
    model = piece_reading.digest()
    SHEETS.mkdir(parents=True, exist_ok=True)
    with (SHEETS / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        kept = json.loads(receipt.read_text()) if receipt.is_file() else {}
        if kept.get("model_digest") != model or receipt.stat().st_mtime < layout.stat().st_mtime:
            pending = piece_reading.queue_of(sample, volume)
            if pending:
                raise ValueError(f"{scan}: {sum(len(jobs) for _, jobs in pending)} pieces still need ink")
            coverage = survey_status.snapshot(sample, volume, use_review=False)
            if not coverage["pieces"] or coverage["ink_pieces"] != coverage["pieces"]:
                raise ValueError(f"{scan}: ink maps missing from the whole-scroll layout")
            sheet_stage.step(
                "website.sheet_tiles",
                sample,
                volume,
                str(coverage["voxel_um"] / 1000),
                name,
                ",".join(LAYERS),
                "--model",
                model_label,
                "--ink-name",
                "ink",
                "--written-face",
            )
            render_page(sheet, folder, site_pages.card_title(sample, name), scan)
            # the receipt is kept once the page is composed, so a check that fails is retried without composing again
            kept = {
                "page": f"/sheets/{name}/index.html",
                "weights": str(weights),
                "model_digest": model,
                "pack_bytes": TileStore(folder).pack.stat().st_size,
                "coverage": coverage,
            }
            pending_receipt = receipt.with_suffix(".json.part")
            pending_receipt.write_text(json.dumps(kept, indent=2))
            pending_receipt.replace(receipt)
        site_pages.write_start_page(site_pages.start_status(), site_pages.front_cards())
        if live(url):
            with urllib.request.urlopen(f"{url}/sheets/index.html", timeout=15) as response:
                if f"{name}/index.html" not in response.read().decode():
                    raise ValueError(f"{name}: viewer missing from the live homepage")
            size = check_page(folder, scan, url)
        else:
            size = TileStore(folder).pack.stat().st_size
            print(f"{url} is not running: the page was written but not checked over HTTP", flush=True)
        faces = stage_folder(base, "faces") / FACES
        for piece in json.loads(layout.read_text())["pieces"]:
            image = faces / piece["unit"] / f"piece-{piece['piece']:04d}.png"
            for source in (image, piece_reading.layer0(image), piece_reading.taught_png(image)):
                source.unlink(missing_ok=True)
        print(
            f"{kept['page']}: {kept['coverage']['ink_pieces']} pieces, {size / 1e9:.2f} GB of review tiles", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample")
    parser.add_argument("scan")
    parser.add_argument("--url", default=os.environ.get("HERCULEX_SITE", "http://127.0.0.1:8765"))
    args = parser.parse_args()
    publish(args.sample, args.scan, args.url.rstrip("/"))
