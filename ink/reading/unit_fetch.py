"""Each unit's CT for the ink read, fetched as soon as the unit's pieces are written, while the pieces stage still
unwraps the units after it, so the read finds its stores ready instead of waiting on the link.

./run.sh ink.reading.unit_fetch SCROLL SCAN      or ./run.sh runner.pipeline run fetch SCROLL

A unit's pieces never change once its record is written, and the planes a piece's stack reads lie symmetric about its
surface, so they don't depend on the layout's turn: a store fetched now holds exactly the bytes the read would fetch.
The fetch leaves the stores to the read once the read has taken over (a layout or a read started after the newest
finished unit), and stops when every unit is fetched, when the scan's stores reach HERCULEX_FETCH_AHEAD_GIB, when the
disc is short of room, or when no unit has finished for HERCULEX_FETCH_IDLE_HOURS. A store it fails to fetch is left
to the read, which fetches it itself.
"""

import json
import os
import shutil
import sys
import time

from ink.reading.piece_reading import READING
from runner import events, stage_queue
from sampling import column_sampling as sampler
from sampling import piece_images
from storage.data_roots import LEVEL_ROUTE, NODE_ROUTE, ct_bricks, scan_folder, stage_entries, stage_folder

BUDGET_GIB = float(os.environ.get("HERCULEX_FETCH_AHEAD_GIB", "200"))
IDLE_HOURS = float(os.environ.get("HERCULEX_FETCH_IDLE_HOURS", "1.5"))
POLL_SECONDS = 30


class NoRoom(Exception):
    """The disc is short of room for fetching ahead; the read, which may go lower, fetches the rest."""


def finished_units(base):
    """The scan's piece units that are current, by the pieces stage's own test."""
    from unwrapping.level import run_levels

    pieces = base / "pieces"
    units = sorted(pieces.glob("rows-*")) or sorted(pieces.glob("band-*"))
    return [unit for unit in units if stage_queue.done(unit, run_levels.METHOD, run_levels.SAME_OUTPUT)]


def node_units(sample, volume):
    """The scan's finished sheet-node units, which the pieces stage turns into piece units one by one."""
    nodes = stage_folder(scan_folder(sample, volume, route=NODE_ROUTE), "nodes")
    units = sorted(nodes.glob("rows-*")) or sorted(nodes.glob("band-*"))
    return [unit for unit in units if stage_queue.done(unit)]


def unit_jobs(unit):
    """Read jobs of every written piece of a unit; the turn and axis don't change the planes."""
    record = json.loads((unit / stage_queue.RECORD).read_text())
    return [(unit / f"piece-{r['piece']:04d}", None, None, 1, None) for r in record["pieces"] if r.get("written")]


def stored_gib(sample, units):
    return (
        sum(f.stat().st_size for unit in units for f in ct_bricks(sample, unit.name).glob("*") if f.is_file()) / 2**30
    )


def taken_over(base, finished):
    """Whether the sheet stage has taken over these pieces: a layout written, or a read started, after the newest of
    the finished units. A fetch started late, or started again, sees it as well as one that ran all along."""
    if not finished:
        return False
    newest = max((unit / stage_queue.RECORD).stat().st_mtime for unit in finished)
    sheet = stage_folder(base, "sheet")
    return any(
        (sheet / name).is_file() and (sheet / name).stat().st_mtime > newest for name in ("layout.json", READING)
    )


def room_ahead(store):
    """The fetch's report: offsets saved, and no wait for room while it holds a store the read may want."""
    store.save()
    free = shutil.disk_usage(store.folder).free
    if free < sampler.ROOM_BYTES:
        raise NoRoom(f"{free / 2**30:.0f} GiB free at {store.folder}")


def follow(sample, volume, fetch=piece_images.fetch_bricks, clock=time.time, sleep=time.sleep):
    """Fetch every finished unit's store, in the order units finish, until there is nothing left to do."""
    base = LEVEL_ROUTE / sample / volume
    fetched, last_new = set(), clock()
    while True:
        finished = finished_units(base)
        if taken_over(base, finished):
            print(f"{sample} {volume}: the read has taken over; it fetches from here", flush=True)
            return fetched
        for unit in finished:
            if unit.name in fetched:
                continue
            if stored_gib(sample, finished) >= BUDGET_GIB:
                print(f"{sample} {volume}: stores reached {BUDGET_GIB:g} GiB; the read fetches the rest", flush=True)
                return fetched
            began = clock()
            try:
                with events.span("fetch", store=unit.name):
                    ran = fetch(
                        sample,
                        volume,
                        unit_jobs(unit),
                        ct_bricks(sample, unit.name),
                        report=room_ahead,
                        still=lambda: not taken_over(base, finished_units(base)),
                    )
            except NoRoom as error:
                print(f"{sample} {unit.name}: stopped fetching ahead, {error}", flush=True)
                return fetched
            except OSError as error:
                print(f"{sample} {unit.name}: left to the read, {error}", flush=True)
                ran = True
            if ran is False:
                print(f"{sample} {volume}: the read has taken over; it fetches from here", flush=True)
                return fetched
            print(f"{sample} {unit.name}: CT ready in {clock() - began:.0f} s", flush=True)
            fetched.add(unit.name)
            last_new = clock()
        wanted = {unit.name for unit in node_units(sample, volume)}
        if wanted and wanted <= fetched:
            print(f"{sample} {volume}: every unit's CT is ready", flush=True)
            return fetched
        if clock() - last_new > IDLE_HOURS * 3600:
            print(f"{sample} {volume}: no unit finished for {IDLE_HOURS:g} h; stopping", flush=True)
            return fetched
        sleep(POLL_SECONDS)


def scans_of(sample, volume=None):
    """The scan the pipeline reads for a scroll, or the one asked for; stores are named by unit, not by scan, so two
    scans of a scroll are never fetched ahead together."""
    if volume is not None:
        return [volume]
    from scans.catalog.scan_choice import chosen_scans

    with_nodes = {n.parts[-3] for n in stage_entries(NODE_ROUTE, "nodes", "*") if n.parts[-4] == sample}
    return [row["volume"] for row in chosen_scans({sample}) if row["volume"] in with_nodes]


def run(samples, band=False, volume=None, since=None):
    """The pipeline's fetch stage."""
    del band, since
    for sample in samples:
        for scan in scans_of(sample, volume):
            if node_units(sample, scan):
                follow(sample, scan)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: ./run.sh ink.reading.unit_fetch SCROLL SCAN")
    follow(sys.argv[1], sys.argv[2])
