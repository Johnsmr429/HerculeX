"""Which scroll, scan, route, page and pieces each hand-marked sheet's circles lie on."""

import json
from dataclasses import dataclass
from pathlib import Path

from storage.data_roots import GROWN_ROUTE, HAND_MARKS, LEVEL_ROUTE, OUTPUTS, route_id, stage_entries
from storage.data_roots import SHEETS as REVIEW_PAGES

REGISTRY = HAND_MARKS / "sheets.json"  # pages whose marks are not found by themselves: rebuilt runs, older routes


@dataclass(frozen=True)
class Sheet:
    key: str  # marks folder under HAND_MARKS
    scroll: str
    scan: str
    route: str  # level or grown, or frame (a continuous-frame rebuild)
    page: str  # website sheet the circles were drawn on
    pieces: Path  # unit folders of the pieces that page shows
    part: str = ""  # the half of a two-unwraps page: "new" is the level route, "first" the grown route

    @property
    def log(self):
        return HAND_MARKS / self.key / "marks.jsonl"

    def layout(self, digest):
        """The placement a stroke was drawn on, as the marks server kept it."""
        data = json.loads((HAND_MARKS / self.key / "layouts" / f"{digest}.json").read_text())
        return data[self.part] if self.part else data["pieces"]


def registered():
    """Sheets listed in marks/sheets.json, each with its pieces folder relative to the outputs."""
    if not REGISTRY.is_file():
        return ()
    return tuple(Sheet(**(entry | {"pieces": OUTPUTS / entry["pieces"]})) for entry in json.loads(REGISTRY.read_text()))


def all_sheets():
    """The registered sheets, then every published scroll page, whose receipt carries its scan and pieces."""
    listed = registered()
    known = {s.key for s in listed}
    yield from listed
    for route in (LEVEL_ROUTE, GROWN_ROUTE):
        for receipt in sorted(stage_entries(route, "sheet", "review.json")):
            scan = receipt.parent.parent
            page = Path(json.loads(receipt.read_text())["page"]).parent.name
            key = f"{route_id(route)}/{scan.parent.name}/{scan.name}"
            if key not in known and (REVIEW_PAGES / page / "index.html").is_file():
                yield Sheet(
                    key, scan.parent.name, scan.name, route_id(route).removesuffix("_route"), page, scan / "pieces"
                )


def sheets_of(key):
    return [s for s in all_sheets() if s.key == key]


def logs():
    """Every marks log under HAND_MARKS, so a log that belongs to no sheet is noticed."""
    return sorted(str(p.parent.relative_to(HAND_MARKS)) for p in HAND_MARKS.glob("**/marks.jsonl"))
