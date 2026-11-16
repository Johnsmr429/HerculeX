"""Pieces to the website, the same way for every scan: laid out, read, then published on the main page."""

import json
import os
import subprocess
import sys

from runner import stage_queue
from storage.data_roots import HERE, ROUTE_OUTPUTS, stage_folder

SITE = os.environ.get("HERCULEX_SITE", "http://127.0.0.1:8765")  # where a running viewer is checked


def started(module, *arguments, **settings):
    return subprocess.Popen(
        [sys.executable, "-m", module, *arguments], cwd=HERE, env=os.environ | {"MallocLargeCache": "0"} | settings
    )


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def step(module, *arguments, **settings):
    process = started(module, *arguments, **settings)
    try:
        if process.wait():
            raise RuntimeError(f"{module} failed on {arguments}")
    except BaseException:
        stop(process)
        raise


def newer(sources, product, *, layout_mtime=0.0):
    """Whether any inputs exist and the product needs newer inputs or placement."""
    latest = max((s.stat().st_mtime for s in sources if s.exists()), default=None)
    return latest is not None and (
        product is None or not product.exists() or max(layout_mtime, latest) > product.stat().st_mtime
    )


def units_of(pieces):
    return sorted(pieces.glob("rows-*")) or sorted(pieces.glob("band-*"))


def uses_frame(units):
    return len(units) > 1 and units[0].name.startswith("rows-")


def ready(samples):
    """Scans with their pieces, joints and frame ready for layout."""
    for pieces in sorted(ROUTE_OUTPUTS.glob("*/*/pieces")):
        sample, volume = pieces.parts[-3], pieces.parts[-2]
        if samples and sample not in samples:
            continue
        rows = sorted(pieces.glob("rows-*"))
        units = rows or sorted(pieces.glob("band-*"))
        nodes = (stage_folder(pieces.parent, "nodes")).glob("rows-*")
        whole = {u.name for u in rows} == {u.name for u in nodes} and (
            len(rows) < 2 or stage_queue.done(pieces / "joints")
        )
        frame = pieces / "frame"
        if (
            units
            and whole
            and all(stage_queue.done(u) for u in units)
            and (
                not uses_frame(units)
                or (
                    stage_queue.done(frame)
                    and not newer([u / stage_queue.RECORD for u in units], frame / stage_queue.RECORD)
                )
            )
        ):
            yield sample, volume


def laid_out(sample, volume):
    """The pieces of a scan on its sheet, laid again when the pieces or frame change."""
    base = ROUTE_OUTPUTS / sample / volume
    layout = stage_folder(base, "sheet") / "layout.json"
    units = units_of(base / "pieces")
    pieces_changed = newer([u / stage_queue.RECORD for u in units], layout)
    framed = uses_frame(units)
    if pieces_changed:
        (stage_folder(base, "sheet") / "joints.json").unlink(missing_ok=True)
    if pieces_changed or (framed and newer([base / "pieces" / "frame" / stage_queue.RECORD], layout)):
        step(
            "unwrapping.level.scroll_layout" if framed else "unwrapping.layout.mosaic_layout",
            sample,
            volume,
            *(["band"] if units[0].name.startswith("band-") else []),
        )
        layout = stage_folder(base, "sheet") / "layout.json"
    return json.loads(layout.read_text())["pieces"]


def ink_of(sample, volume):
    if laid_out(sample, volume):
        step("ink.reading.piece_reading", sample, volume)
    else:
        print(f"{sample} {volume}: no laid-out pieces to read", flush=True)


def sheet_of(sample, volume):
    """One scan from its pieces to its page on the main page: every laid-out piece read, then published."""
    if not laid_out(sample, volume):
        print(f"{sample} {volume}: no pieces to publish", flush=True)
        return
    step("ink.reading.piece_reading", sample, volume)
    from runner import review_page

    review_page.publish(sample, volume, SITE)


def run(samples=(), *, volume=None, mode="sheet", require_ready=False):
    scans_ready = [(s, v) for s, v in ready(set(samples)) if volume is None or v == volume]
    missing = set(samples) - {s for s, _ in scans_ready}
    if require_ready and missing:
        selection = f" ({volume})" if volume else ""
        raise RuntimeError(
            f"no ready pieces in {ROUTE_OUTPUTS.name} for {', '.join(sorted(missing))}{selection}. "
            "Check pipeline status and finish the pieces stage for that scan."
        )
    action = {"layout": laid_out, "ink": ink_of, "sheet": sheet_of}[mode]
    for sample, scan in scans_ready:
        print(f"{mode} {sample} {scan} ({ROUTE_OUTPUTS.name})", flush=True)
        action(sample, scan)
