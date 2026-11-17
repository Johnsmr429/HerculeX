"""The start page, a viewer for every sheet and largest area, and the proposals.

./run.sh website.site_pages                                   every page and the start page, rewritten
./run.sh website.site_pages card PAGE [SCROLL] [--surface W]  put a published page on the start page
./run.sh website.site_pages card PAGE --remove                take it off again
./run.sh website.site_pages part NAME TITLE SCROLL SCAN [UNIT,UNIT]

A scan's whole-scroll page goes on the start page by itself when the pipeline publishes it. Any other published page
(a rebuilt surface, a read by another model) is added with `card`: it is listed in website/sheets/cards.json, gets a
card picture if it has none, and is labelled like every card, "scroll · model · surface · date", from its layers.
"""

import argparse
import itertools
import json
import math
import time

import numpy as np
from PIL import Image

from sampling.sheet_images import mosaics_of
from scans.catalog.scan_catalog import load, survey_volumes
from scans.catalog.scan_choice import prize_order
from storage.data_roots import (
    BROWSER_ASSETS,
    FACES,
    GROWN_ROUTE,
    LEVEL_ROUTE,
    PRODUCTION_FACES,
    ROUTE_OUTPUTS,
    SHEETS,
    SITE_ATLAS,
    stage_entries,
    stage_folder,
)
from storage.tile_store import TileStore
from website import suggested_areas, window_proposals
from website.refresh_viewers import refresh
from website.sheet_tiles import TILE, page, shown
from website.survey_status import snapshot

CARD = (1800, 1125)
OVERVIEW_LEVEL = 3
INK_TINT = "purple"
SCREEN_SHAPE = 1.9
SHELF_GAP_MM = 20.0
# in SHEETS: the pages put on the start page with `card`, each its page, scroll and surface if given
CARDS = "cards.json"
# the atlas pages the start page links to, once website.atlas_build has made them
ATLAS_LINKS = (
    ("index", "Atlas"),
    ("alphabet", "Alphabet"),
    ("fragments", "Infrared"),
    ("published", "Published"),
    ("windows", "Lines"),
    ("letters", "Letters"),
    ("compare", "Scans"),
)


def layers_of(name, up=""):
    return [
        layer | {"path": up + name + "/" + layer["path"] if up else layer["path"]}
        for layer in json.loads((SHEETS / name / "layers.json").read_text())
    ]


def crop(sheet, layer, box, level=0):
    """A box of a sheet from its tiles at a zoom level, as the viewer shows it, black where there is no scan."""
    store = TileStore(sheet)
    mm = layer["mm"] * 2**level
    tile_mm = TILE * mm
    x0, y0 = box["x"], box["y"]
    x1, y1 = x0 + box["w"], y0 + box["h"]
    canvas = Image.new("L", (int((x1 - x0) / mm), int((y1 - y0) / mm)), 0)
    for c in range(max(0, int(x0 // tile_mm)), int(x1 // tile_mm) + 1):
        for r in range(max(0, int(y0 // tile_mm)), int(y1 // tile_mm) + 1):
            pixels = store.image(layer["path"], level, c, r)
            if pixels is not None:
                at = (round((c * tile_mm - x0) / mm), round((r * tile_mm - y0) / mm))
                canvas.paste(Image.fromarray(shown(pixels, layer)), at)
    return canvas


def scan_layer(name):
    """The raw layer at the mesh of a sheet."""
    layers = json.loads((SHEETS / name / "layers.json").read_text())
    by_path = {layer["path"]: layer for layer in layers}
    return next(by_path[path] for path in ("layer0", "taught-scan", "sheet") if path in by_path)


def card(name, place, out, level=0):
    """A start-page picture."""
    layer = scan_layer(name)
    w, h = CARD[0] * layer["mm"] * 2**level, CARD[1] * layer["mm"] * 2**level
    best, fill = None, -1.0
    for fx in (0.2, 0.35, 0.5, 0.65, 0.8):
        for fy in (0.2, 0.35, 0.5, 0.65, 0.8):
            box = {"x": place["x"] + place["w"] * fx - w / 2, "y": place["y"] + place["h"] * fy - h / 2, "w": w, "h": h}
            image = crop(SHEETS / name, layer, box, level)
            share = float((np.asarray(image) > 0).mean())
            if share > fill:
                best, fill = image, share
    best.save(out, quality=92)


def layout_of(sample, volume):
    return json.loads((stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "layout.json").read_text())["pieces"]


def read_pieces(sample, volume, faces=PRODUCTION_FACES, units=None):
    """The laid-out pieces read so far into some faces, of some units if given."""
    folder = stage_folder(ROUTE_OUTPUTS / sample / volume, "faces") / faces
    return [
        p
        for p in layout_of(sample, volume)
        if (not units or p["unit"] in units) and (folder / p["unit"] / f"piece-{p['piece']:04d}.png").exists()
    ]


def places_of(pieces, most=200):
    """The largest continuous areas of some pieces as places to jump to, each the bounding box of its pieces."""
    places = []
    for members in sorted(mosaics_of(pieces), key=lambda m: -sum(p["area"] for p in m))[:most]:
        x0, y0 = min(p["x_mm"] for p in members), min(p["y_mm"] for p in members)
        x1, y1 = max(p["x_mm"] + p["width_mm"] for p in members), max(p["y_mm"] + p["height_mm"] for p in members)
        places.append({"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)})
    return places


def largest_mosaics(sample, volume, most=200):
    """The largest continuous areas read so far as places to jump to."""
    return places_of(read_pieces(sample, volume), most)


def shelves_of(sample, volume):
    return shelves_of_pieces(layout_of(sample, volume))


def shelves_of_pieces(pieces):
    """A sheet metres long cut into shelves stacked under one another to fill a screen, each cut where it crosses the least papyrus."""
    height = max(p["y_mm"] + p["height_mm"] for p in pieces)
    length = max(p["x_mm"] + p["width_mm"] for p in pieces)
    starts, ends = (np.array([p["x_mm"] + p["width_mm"] * side for p in pieces]) for side in (0, 1))
    areas = np.array([p["area"] for p in pieces])
    count = max(1, round(math.sqrt(length / (SCREEN_SHAPE * (height + SHELF_GAP_MM)))))
    edges = [0.0]
    for k in range(1, count):
        near = np.linspace(0.95, 1.05, 101) * k * length / count
        edges.append(float(near[np.argmin([areas[(starts < x) & (ends > x)].sum() for x in near])]))
    edges.append(length)
    return [
        {"x": round(a, 1), "y": round(k * (height + SHELF_GAP_MM), 1), "w": round(b - a, 1), "h": round(height, 1)}
        for k, (a, b) in enumerate(itertools.pairwise(edges))
    ]


def part_page(name, title, sample, volume, units):
    """The viewer of some units, or of all when none are named, read into FACES, with its own places, shelves and marks."""
    read = read_pieces(sample, volume, FACES, units)
    if not read:
        raise ValueError(f"no read pieces of {', '.join(sorted(units)) or 'any unit'} in papyrus-faces/{FACES}")
    shelves = shelves_of_pieces(read)
    page(
        SHEETS / name,
        title,
        layers_of(name),
        {"x": 0, "y": 0, "w": max(s["w"] for s in shelves), "h": shelves[-1]["y"] + shelves[-1]["h"]},
        "../index.html",
        f"part/{name}",
        places=places_of(read),
        shelves=shelves,
        tint=INK_TINT,
    )


def scan_of(sample, volume):
    """How the marks server names a scan."""
    return f"level/{sample}/{volume}"


def sheet_pages(name, proposal_name, title, sample, volume):
    """A sheet's page and its proposals page."""
    places = largest_mosaics(sample, volume)
    if not places:
        return
    shelves = shelves_of(sample, volume)
    whole = {"x": 0, "y": 0, "w": max(s["w"] for s in shelves), "h": shelves[-1]["y"] + shelves[-1]["h"]}
    suggested = suggested_areas.squares(sample, volume)
    page(
        SHEETS / name,
        title,
        layers_of(name),
        whole,
        "../index.html",
        scan_of(sample, volume),
        places=places,
        shelves=shelves,
        suggested=suggested,
        tint=INK_TINT,
    )
    card(name, places[0], SHEETS / name / "card.jpg", OVERVIEW_LEVEL)
    target = SHEETS / proposal_name
    target.mkdir(parents=True, exist_ok=True)
    boxes = [{k: b[k] for k in ("x", "y", "w", "h")} for b in window_proposals.run(sample, volume)]
    if boxes:
        card(name, boxes[0], target / "card.jpg")
    page(
        target,
        title + " · proposed",
        layers_of(name, "../"),
        whole,
        "../index.html",
        scan_of(sample, volume),
        boxes=boxes,
        shelves=shelves,
        suggested=suggested,
        tint=INK_TINT,
    )


def scans(root=ROUTE_OUTPUTS):
    """Every scan laid out on a sheet by one route, the prize scrolls first."""
    energy = {r["volume"]: r["energy_kev"] for r in survey_volumes(load())}
    found = sorted((path.parts[-4], path.parts[-3]) for path in stage_entries(root, "sheet", "layout.json"))
    route = "-level" if root == LEVEL_ROUTE else ""
    kev = {v: f"{e:.0f}" if e else "" for v, e in energy.items()}  # a scan without a recorded energy goes by its id
    return [
        (
            s,
            v,
            f"{s}-{kev[v]}keV{route}" if kev.get(v) else f"{s}-{v}{route}",
            f"{s} · {kev[v]} keV" if kev.get(v) else f"{s} · {v}",
        )
        for s, v in sorted(found, key=lambda scan: prize_order(scan[0]))
    ]


def read_on(name, layer):
    """The day a sheet's ink map was composed, from its layer's stamp."""
    stamp = next(
        (
            entry.get("stamp")
            for entry in json.loads((SHEETS / name / "layers.json").read_text())
            if entry["path"] == layer
        ),
        None,
    )
    return time.strftime("%-d %B %Y", time.gmtime(stamp)) if stamp else ""


# a page's surface in one word, by its name, the first match winning, unless its card names one
SURFACES = (
    ("two-unwraps-", "grown+level"),
    ("-continuous-frame", "frame"),
)
INK_LAYERS = ("taught", "new-ink", "ink")


def registered():
    """The pages added to the start page with `card`, as {page, scroll, surface?} in the order they were added."""
    listed = SHEETS / CARDS
    return json.loads(listed.read_text()) if listed.is_file() else []


def surface_of(name):
    named = next((entry.get("surface") for entry in registered() if entry["page"] == name), None)
    return named or next((word for key, word in SURFACES if key in name), "level")


def card_title(sample, name, surface=None):
    """A start page card's label: scroll, the ink model its ink layer names, surface, the day its ink was composed."""
    layers = {layer["path"]: layer for layer in layers_of(name)}
    path = next((path for path in INK_LAYERS if path in layers), None)
    model = layers[path].get("model") if path else None
    return " · ".join(filter(None, (sample, model, surface or surface_of(name), read_on(name, path) if path else "")))


def ink_stamp(name):
    """When a page's ink was composed, the moment its card's date names; 0 for a page without an ink layer."""
    layers = {layer["path"]: layer for layer in layers_of(name)}
    return next((layers[path].get("stamp") or 0 for path in INK_LAYERS if path in layers), 0)


def start_status():
    """The start page's line: every scroll it lists, by the one scan of its card (the level route's where a scroll
    has both), summed: area laid and area read."""
    both = {p.name.removeprefix("two-unwraps-") for p in SHEETS.glob("two-unwraps-*") if (p / "index.html").exists()}
    chosen = {}
    for root in (GROWN_ROUTE, LEVEL_ROUTE):
        for sample, volume, tag, _ in scans(root):
            if sample in both or (SHEETS / f"sheet-{tag}" / "index.html").exists():
                chosen[sample] = (root, volume)
    reports = [snapshot(sample, volume, root=root) for sample, (root, volume) in chosen.items()]
    return {
        "scrolls": len(reports),
        "laid_cm2": sum(r["laid_cm2"] for r in reports),
        "ink_cm2": sum(r["ink_cm2"] for r in reports),
    }


def front_cards():
    """The scrolls on the start page, one card per scroll page, the newest read first."""
    cards, both = (
        [],
        {p.name.removeprefix("two-unwraps-") for p in (SHEETS).glob("two-unwraps-*") if (p / "index.html").exists()},
    )
    for sample, _, tag, _ in scans(GROWN_ROUTE):
        if (SHEETS / f"sheet-{tag}" / "index.html").exists():
            cards.append(
                {
                    "href": f"sheet-{tag}/index.html",
                    "image": f"sheet-{tag}/card.jpg",
                    "title": card_title(sample, f"sheet-{tag}", "grown"),
                }
            )
    for sample in sorted(both, key=prize_order):
        name = f"two-unwraps-{sample}"
        cards.append(
            {
                "href": f"{name}/index.html",
                "image": f"{name}/card.jpg",
                "title": card_title(sample, name),
            }
        )
    for sample, _, tag, _ in scans(LEVEL_ROUTE):
        if sample not in both and (SHEETS / f"sheet-{tag}" / "index.html").exists():
            cards.append(
                {
                    "href": f"sheet-{tag}/index.html",
                    "image": f"sheet-{tag}/card.jpg",
                    "title": card_title(sample, f"sheet-{tag}"),
                }
            )
    # pages added with `card`; studies (passages, known letters) are reached by their own address
    for entry in registered():
        name = entry["page"]
        if (SHEETS / name / "index.html").is_file() and (SHEETS / name / "card.jpg").is_file():
            cards.append(
                {"href": f"{name}/index.html", "image": f"{name}/card.jpg", "title": card_title(entry["scroll"], name)}
            )
    return sorted(cards, key=lambda card: -ink_stamp(card["href"].split("/")[0]))


def page_extent(name):
    """The box a page's full-resolution scan tiles cover, in sheet millimetres."""
    layer = scan_layer(name)
    tiles = [t for t in TileStore(SHEETS / name).tiles(layer["path"]) if t[0] == 0]
    span = TILE * layer["mm"]
    cols, rows = [t[1] for t in tiles], [t[2] for t in tiles]
    x, y = min(cols) * span, min(rows) * span
    return {"x": x, "y": y, "w": (max(cols) + 1) * span - x, "h": (max(rows) + 1) * span - y}


def add_card(name, sample=None, surface=None, remove=False):
    """Put a published page on the start page, or take it off, and rewrite the start page."""
    folder = SHEETS / name
    if not remove and not (folder / "index.html").is_file():
        raise SystemExit(f"no published page at {folder}")
    entries = [entry for entry in registered() if entry["page"] != name]
    if not remove:
        entries.append({"page": name, "scroll": sample or sample_of(name)} | ({"surface": surface} if surface else {}))
        if not (folder / "card.jpg").is_file():
            extent = page_extent(name)
            level = OVERVIEW_LEVEL if extent["w"] > 8 * CARD[0] * scan_layer(name)["mm"] else 0
            card(name, extent, folder / "card.jpg", level)
    pending = SHEETS / f"{CARDS}.part"
    pending.write_text(json.dumps(entries, indent=1) + "\n")
    pending.replace(SHEETS / CARDS)
    cards = front_cards()
    write_start_page(start_status(), cards)
    print(*(card["title"] for card in cards), sep="\n")


def sample_of(name):
    """The scroll a page folder is named for (sheet-PHerc0191-113keV-level, two-unwraps-PHerc0211, ...)."""
    return next(part for part in name.split("-") if part.startswith("PHerc"))


def scroll_page_titles(cards):
    """Each whole-scroll viewer's title, as its start page card names it: the cards' pages, and the continuous
    frame's scroll pages, which are reached by their own address."""
    titles = {card["href"].split("/")[0]: card["title"] for card in cards}
    for folder in SHEETS.glob("scroll-*-continuous-frame"):
        if (folder / "index.html").is_file() and (folder / "layers.json").is_file():
            titles[folder.name] = card_title(sample_of(folder.name), folder.name)
    return titles


def write_start_page(report, cards):
    """Publish the homepage from the status report and selected cards, and title every whole-scroll viewer as its
    card is labelled."""
    html = (BROWSER_ASSETS / "start_page.html").read_text()
    built = [(page, name) for page, name in ATLAS_LINKS if (SITE_ATLAS / f"{page}.html").is_file()]
    atlas = "".join(f'<a href="../atlas/{page}.html">{name}</a>' for page, name in built)
    for key, value in {"__STATUS__": json.dumps(report), "__CARDS__": json.dumps(cards), "__ATLAS__": atlas}.items():
        html = html.replace(key, value)
    target = SHEETS / "index.html"
    pending = target.with_suffix(".html.part")
    pending.write_text(html)
    pending.replace(target)
    refresh(SHEETS, titles=scroll_page_titles(cards))


def run():
    """Pages and the start page of everything composed so far."""
    listed = scans()
    if not listed:
        print("No laid-out scans to publish.", flush=True)
        return
    for sample, volume, tag, title in listed:
        area, sheet, proposals = f"largest-area-{tag}", f"sheet-{tag}", f"proposals-{tag}"
        if (SHEETS / area / "layers.json").exists() and (places := largest_mosaics(sample, volume, 1)):
            place = places[0]
            page(
                SHEETS / area,
                title + " · largest area",
                layers_of(area),
                place,
                "../index.html",
                scan_of(sample, volume),
            )
            card(area, place, SHEETS / area / "card.jpg")
        if (SHEETS / sheet / "layers.json").exists() and not (
            stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "review.json"
        ).exists():
            sheet_pages(sheet, proposals, title, sample, volume)
    cards = front_cards()
    report = start_status()
    write_start_page(report, cards)
    (SHEETS / "status.json").write_text(json.dumps(report, indent=1))
    refresh()
    print("pages done", [c["title"] for c in cards], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="website.site_pages", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command")
    add = commands.add_parser("card", help="put a published page on the start page")
    add.add_argument("page")
    add.add_argument("scroll", nargs="?")
    add.add_argument("--surface", help="the surface in one word, e.g. rebuilt")
    add.add_argument("--remove", action="store_true")
    part = commands.add_parser("part", help="a viewer of some units")
    part.add_argument("arguments", nargs="+", metavar="NAME TITLE SCROLL SCAN [UNIT,UNIT]")
    args = parser.parse_args()
    if args.command == "card":
        add_card(args.page, args.scroll, args.surface, args.remove)
    elif args.command == "part":
        given = args.arguments
        part_page(*given[:4], set(given[4].split(",")) if len(given) > 4 else set())
    else:
        run()
