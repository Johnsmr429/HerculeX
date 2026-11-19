"""Scrolls built on another machine placed among the outputs, so the one viewer shows them.

./run.sh runner.place_build STAGING SCROLL [SCROLL ...] [--model NAME]

STAGING holds the other machine's outputs (unwrapping/level/SCROLL/SCAN and website/sheets/sheet-*). Each scan's
build becomes the scan's build and its page the scan's page. A build it replaces moves to archive/DATE-build/, and the
replaced page stays beside the new one as sheet-...-DATE, a part page with its own layout and its own card. With
--model NAME, the staged pages are a read of the same build by that ink model: they are placed as sheet-...-NAME with
their read's receipt, beside the scan's own page.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from storage.data_roots import BROWSER_ASSETS, LEVEL_ROUTE, NODE_ROUTE, OUTPUTS, SHEETS
from website import site_pages
from website.refresh_viewers import FLOOR, META, SCAN


def say(*words):
    print(time.strftime("%H:%M:%S"), *words, flush=True)


def as_part(page, layout):
    """A page that keeps its own layout, so marks drawn on it lie on its own pieces."""
    shutil.copy2(layout, page / "placed.json")
    old = (page / "index.html").read_text()
    meta = json.loads(META.search(old)[1]) | {"scan": f"part/{page.name}"}
    html = (
        (BROWSER_ASSETS / "sheet_viewer.html")
        .read_text()
        .replace("__TITLE__", meta["title"])
        .replace("__META__", json.dumps(meta))
        .replace("__FLOOR__", FLOOR.search(old)[1])
        .replace("__SCAN__", SCAN.search(old)[1])
    )
    (page / "index.html.part").write_text(html)
    (page / "index.html.part").replace(page / "index.html")


def archive(scroll, scan):
    """The scan's build moved to the archive and its page renamed by the build's date; the archived page or None."""
    build = LEVEL_ROUTE / scroll / scan
    layout = build / "sheet" / "layout.json"
    date = time.strftime("%Y-%m-%d", time.localtime((layout if layout.exists() else build).stat().st_mtime))
    target = OUTPUTS / "archive" / f"{date}-build" / "unwrapping" / "level" / scroll / scan
    target.parent.mkdir(parents=True, exist_ok=True)
    build.rename(target)
    archived = None
    for page in sorted(SHEETS.glob(f"sheet-{scroll}-*-level")):
        if json.loads(META.search((page / "index.html").read_text())[1]).get("scan") == f"level/{scroll}/{scan}":
            archived = page.with_name(f"{page.name}-{date}")
            page.rename(archived)
            placed = target / "sheet" / "placed.json"
            as_part(archived, placed if placed.exists() else target / "sheet" / "layout.json")
    say(f"{scroll} {scan}: the build of {date} archived{f', its page kept as {archived.name}' if archived else ''}")
    return archived


def place(staging, scroll, model=None):
    """One scroll's staged builds and pages placed; the cards of the pages that sit beside the scans' own."""
    cards = []
    for build in sorted(p for p in (staging / "unwrapping" / "level" / scroll).iterdir() if p.is_dir()):
        scan, into = build.name, LEVEL_ROUTE / scroll / build.name
        if model:
            receipt = build / "sheet" / "review.json"
            if receipt.exists():
                shutil.copy2(receipt, into / "sheet" / f"review-{model}.json")
            continue
        if into.exists():
            cards.append(archive(scroll, scan))
        into.parent.mkdir(parents=True, exist_ok=True)
        build.rename(into)
        nodes = into / "nodes"
        if not nodes.is_symlink() and (NODE_ROUTE / scroll / scan / "nodes").is_dir():
            nodes.symlink_to(f"../../../nodes/{scroll}/{scan}/nodes")
    for page in sorted((staging / "website" / "sheets").glob(f"sheet-{scroll}-*")):
        name = f"{page.name}-{model}" if model else page.name
        page.rename(SHEETS / name)
        if model:
            cards.append(SHEETS / name)
        say(f"{scroll}: page {name} placed")
    return [card for card in cards if card]


def main():
    parser = argparse.ArgumentParser(prog="runner.place_build", description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("scrolls", nargs="+")
    parser.add_argument("--model", help="the ink model that read the staged pages")
    args = parser.parse_args()
    for scroll in args.scrolls:
        for card in place(args.staging, scroll, args.model):
            site_pages.add_card(card.name, scroll)
    site_pages.write_start_page(site_pages.start_status(), site_pages.front_cards())
    say("start page rewritten")


if __name__ == "__main__":
    main()
