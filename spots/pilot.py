"""One circled spot through narrowing, the m7 rebuild, the 67-layer render, the depth sweep and its images.

./run.sh spots.pilot SPOT_ID [build] [render] [read] [images]      the named stages, or all of them in order

Each stage keeps its outputs under SPOT_CANDIDATES/spots/SPOT_ID and skips what is already there, so a stage
cut short starts again where it stopped. Results go to the database (spots/database.py) as they are made.
"""

import json
import sys
import time

from spots import database, surfaces
from spots.sources import sheets_of

STAGES = ("build", "grow", "render", "read", "images")


def spot_of(spot_id):
    found = [r for r in database.rows("spots") if r["spot_id"] == spot_id]
    if not found:
        raise SystemExit(f"no spot {spot_id} in the catalogue; run ./run.sh spots.catalogue")
    return found[0]


def sheet_of(spot):
    return next(iter(sheets_of(spot["sheet_key"])))


def status(spot, value):
    database.write("spots", [{**spot, "status": value}])


def build(spot, sheet):
    placed = sheet.layout(spot["layout"])
    records = surfaces.built(spot, sheet, placed)
    rows = []
    for surface in ("original", "m7"):
        rows.append(
            {
                "spot_id": spot["spot_id"],
                "surface": surface,
                "pieces": [r["name"] for r in records],
                "nodes": sum(r["snap"]["nodes"] for r in records),
                "snapped_share": (
                    round(
                        sum(r["snap"]["snapped"] * r["snap"]["nodes"] for r in records)
                        / max(sum(r["snap"]["nodes"] for r in records), 1),
                        4,
                    )
                    if surface == "m7"
                    else 0.0
                ),
                "snap": json.dumps({r["name"]: r["snap"] for r in records}) if surface == "m7" else "",
                "margin_mm": surfaces.MARGIN_MM,
                "layers": surfaces.LAYERS,
                "folder": str(surfaces.work_of(spot["spot_id"]) / "surfaces" / surface),
            }
        )
    database.write("surfaces", rows)
    return records


def records_of(spot):
    return json.loads((surfaces.work_of(spot["spot_id"]) / "pieces.json").read_text())["pieces"]


def main(argv):
    spot = spot_of(argv[0])
    sheet = sheet_of(spot)
    wanted = [s for s in argv[1:] if s in STAGES] or list(STAGES)
    for stage in wanted:
        began = time.time()
        if stage == "build":
            build(spot, sheet)
        elif stage == "grow":
            grown = surfaces.grow_all(spot, records_of(spot))
            database.write(
                "surfaces",
                [
                    {
                        "spot_id": spot["spot_id"],
                        "surface": "grown",
                        "pieces": [r["name"] for r in grown],
                        "nodes": sum(r["grow"]["nodes"] for r in grown),
                        "nodes_before": sum(r["grow"]["nodes_before"] for r in grown),
                        "snap": json.dumps({r["name"]: r["grow"] for r in grown}),
                        "margin_mm": surfaces.MARGIN_MM,
                        "layers": surfaces.LAYERS,
                        "folder": str(surfaces.work_of(spot["spot_id"]) / "surfaces" / "grown"),
                    }
                ],
            )
        elif stage == "render":
            surfaces.render_all(spot, sheet, records_of(spot))
        elif stage == "read":
            from spots import sweep

            sweep.read_spot(spot, records_of(spot))
        elif stage == "images":
            from spots import images

            images.write_spot(spot, records_of(spot))
        status(spot_of(spot["spot_id"]), stage)
        print(spot["spot_id"], stage, "done in", round(time.time() - began, 1), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
