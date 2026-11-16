"""Unwrapping stage: sheet nodes to pieces from exact integer levels, flattened and written as tifxyz."""

import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from runner import stage_queue
from storage.data_roots import LEVEL_ROUTE, NODE_ROUTE, stage_entries, stage_folder
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level import level_pieces, scroll_axis, scroll_frame, scroll_layout
from unwrapping.level.piece_flattening import (
    MIN_PIECE_NODES,
    UNIT_NEEDS,
    bridge_holes,
    isometry,
    unit_isometry,
    write_piece,
)
from unwrapping.level.piece_joiner import SEAM_PLANES, boundary_labels, join, seam_joins

HERE = Path(__file__).parent
SAME_OUTPUT = (HERE / "same_output_methods.txt").read_text().split()
CURL_PLANES = 15
READ_BLOCK_CELLS = 112
READ_OVERLAP_CELLS = 16
METHOD = stage_queue.method_of(
    *[
        HERE / f
        for f in (
            "level_pieces.py",
            "orientation.py",
            "level_solver.c",
            "sheet_flattening.c",
            "piece_growth.py",
            "piece_flattening.py",
            "winding_order.py",
            "sheet_graph.py",
        )
    ]
)
MIN_WRITTEN_MM2 = 25.0
BLOCK_BASE = 1_000_000
WRITERS = 4


def piece_name(rank):
    return f"piece-{rank:04d}"


def piece_rows(
    nodes,
    normal,
    u,
    v,
    members,
    piece_first,
    potential,
    theta,
    winding,
    frame_of,
    radius,
    meta,
    out,
    scale,
    voxel_um,
    rasterise,
    block=None,
    overlap=0,
):
    """Write every piece large enough to hold text, in blocks of at most block cells a side when given, each carrying overlap cells of its neighbours' real mesh."""

    def piece(rank):
        group = members[piece_first[rank] : piece_first[rank + 1]]
        flat, locked, crowding = rasterise(nodes[group], normal[group], u[group], v[group])
        if locked.sum() < MIN_PIECE_NODES:
            return []
        filled, known = bridge_holes(flat, locked)
        rows_used, cols_used = np.nonzero(known.any(axis=1))[0], np.nonzero(known.any(axis=0))[0]
        box = (slice(rows_used[0], rows_used[-1] + 1), slice(cols_used[0], cols_used[-1] + 1))
        span = potential[group] + theta[group] / (2 * np.pi)
        side = CELL * scale * voxel_um / 1000
        row = meta | {
            "piece": rank,
            "winding": int(np.median(winding[group])),
            "frame": int(frame_of[group[0]]),
            "turns_spanned": float(np.ptp(span)),
            "cells": int(locked.sum()),
            "nodes": len(group),
            "crowding": float(crowding),
            "area_mm2": float(locked.sum()) * side**2,
            "width_mm": float(locked.shape[1]) * side,
            "height_mm": float(locked.shape[0]) * side,
            "median_radius_mm": float(np.median(radius[group])) * voxel_um * scale / 1000,
        }
        row["written"] = row["area_mm2"] >= MIN_WRITTEN_MM2
        if not row["written"]:
            return [row]
        grid, seen, kept = filled[:, box[0], box[1]], known[box], locked[box]
        if block is None or max(seen.shape) <= block:
            row["isometry"] = isometry(grid, seen, kept, CELL)
            write_piece(out / piece_name(rank), grid, seen, kept, scale, row)
            return [row]
        rows = []
        height, width = seen.shape
        for r0 in range(0, height, block):
            for c0 in range(0, width, block):
                r1, c1 = min(height, r0 + block), min(width, c0 + block)
                cells = int(kept[r0:r1, c0:c1].sum())
                if cells < MIN_PIECE_NODES // 4:
                    continue
                top, left, bottom, right = (
                    min(overlap, r0),
                    min(overlap, c0),
                    min(overlap, height - r1),
                    min(overlap, width - c1),
                )
                here = (slice(r0 - top, r1 + bottom), slice(c0 - left, c1 + right))
                name = rank if not rows else BLOCK_BASE * len(rows) + rank
                part = row | {
                    "piece": name,
                    "source": rank,
                    "block": [r0 - top, c0 - left],
                    "overlap": [top, bottom, left, right],
                    "cells": cells,
                    "area_mm2": cells * side**2,
                    "width_mm": float(seen[here].shape[1]) * side,
                    "height_mm": float(seen[here].shape[0]) * side,
                }
                part["written"] = part["area_mm2"] >= MIN_WRITTEN_MM2
                if part["written"]:
                    part["isometry"] = isometry(grid[:, here[0], here[1]], seen[here], kept[here], CELL)
                    write_piece(out / piece_name(name), grid[:, here[0], here[1]], seen[here], kept[here], scale, part)
                rows.append(part)
        return rows

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        return [r for rows in pool.map(piece, range(len(piece_first) - 1)) for r in rows]


def extract(chunks, out, sample, volume, voxel_um, level, planes=None, shared=(), curl=None, block=None, overlap=0):
    """Pieces of one field into out, the piece of every node in the shared plane ranges kept for joining: the unit's
    record, and the level solve's notes for it, among them the piece terms the scan's one frame needs."""
    notes = {}
    clock = [("start", time.time())]
    nodes, theta, _, _, pairs, record, knots = level_pieces.prepare(chunks, planes)
    clock.append(("prepare", time.time()))
    inside = np.zeros(len(nodes), dtype=bool)
    plane = (nodes["z"] // CELL).astype(np.int64)
    for lo, hi in shared:
        inside |= (plane >= lo) & (plane < hi)
    del plane
    boundary_ids = record.pop("ids")[inside]
    root, potential, a, b, between, stats = level_pieces.grow_pieces(nodes, theta, pairs, record, curl)
    notes.update({k: record.pop(k) for k in ("piece_terms", "node_level")})
    notes.update({k: record[k] for k in ("level_route", "orientation", "frame_costs", "centres") if k in record})
    clock.append(("grow", time.time()))
    sizes = np.bincount(root, minlength=len(nodes))
    placed = float(sizes[sizes >= MIN_PIECE_NODES].sum() / len(nodes))
    del sizes
    members, piece_first, sign, u, v = level_pieces.flatten_pieces(nodes, root, a, b)
    clock.append(("flatten", time.time()))
    if shared:
        rank = np.full(len(nodes), -1, dtype=np.int32)
        for r in range(len(piece_first) - 1):
            rank[members[piece_first[r] : piece_first[r + 1]]] = r
        np.savez(
            out / "boundary.npz",
            ids=boundary_ids,
            piece=rank[inside],
            pieces=len(piece_first) - 1,
            potential=potential[inside],
            theta=theta[inside],
        )
        del rank
    del inside
    # levels are the order: one frame, the curl the level solve took
    winding, frame_of = notes.pop("node_level"), np.zeros(len(nodes), dtype=np.int32)
    chirality = notes["piece_terms"]["chirality"]
    curl_costs = {int(k): v for k, v in notes["frame_costs"].items()}
    ordered, terms = 0, 0
    near = pairs[pairs["rank"] == 1][::7]
    gap_um = (
        float(
            np.median(np.sqrt(sum((nodes[c][near["b"]].astype(np.float64) - nodes[c][near["a"]]) ** 2 for c in "xyz")))
        )
        * voxel_um
        * 2**level
    )
    del pairs, between
    clock.append(("order", time.time()))
    every = slice(None, None, 97)
    flat_step = np.hypot(u[b[every]] - u[a[every]], v[b[every]] - v[a[every]])
    true_step = np.sqrt(sum((nodes[c][b[every]].astype(np.float64) - nodes[c][a[every]]) ** 2 for c in "xyz"))
    stretch = np.percentile((flat_step / np.maximum(true_step, 1e-6))[sign[a[every]] != 0], [5, 50, 95])
    del a, b
    scale = 2**level
    normal = np.stack([nodes[c].astype(np.float32) for c in ("nx", "ny", "nz")], axis=1) * sign[:, None].astype(
        np.float32
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    centre = scroll_axis.centre_at(knots, nodes["z"])
    radius = np.hypot(nodes["x"] - centre[0], nodes["y"] - centre[1])
    del centre
    meta = {
        "sample": sample,
        "scan": volume,
        "voxel_um": voxel_um,
        "level_of_nodes": level,
        "shape": [s * scale for s in record["shape"]],
    }
    rows = piece_rows(
        nodes,
        normal,
        u,
        v,
        members,
        piece_first,
        potential,
        theta,
        winding,
        frame_of,
        radius,
        meta,
        out,
        scale,
        voxel_um,
        level_pieces.rasterise,
        block,
        overlap,
    )
    clock.append(("write", time.time()))
    isometric = unit_isometry(rows)  # fails the unit, its old pieces kept, when its grids do not keep lengths
    sized = np.array([r["area_mm2"] for r in rows] or [0.0])
    frames = np.bincount(frame_of[frame_of >= 0]) if (frame_of >= 0).any() else np.zeros(1, dtype=np.int64)
    in_frame = frame_of == frames.argmax()
    return {
        "sample": sample,
        "volume": volume,
        "voxel_um": voxel_um,
        "level": level,
        "chunks": record["chunks"],
        "cell_planes": record["cell_planes"],
        "shared": [list(r) for r in shared],
        "chirality": int(chirality),
        "curl_costs": {str(k): float(c) for k, c in curl_costs.items()},
        "ordered_pieces": int(ordered),
        "order_terms": int(terms),
        "gap_um": round(gap_um, 1),
        "frame_share": float(in_frame.mean()),
        "turns_in_frame": int(np.ptp(winding[in_frame])) if in_frame.any() else 0,
        "nodes": len(nodes),
        "placed": placed,
        "pieces": rows,
        "area_cm2": float(sized.sum() / 100),
        "area_in_1cm2_pieces": float(sized[sized >= 100].sum() / 100),
        "pieces_1cm2": int((sized >= 100).sum()),
        "pieces_4cm2": int((sized >= 400).sum()),
        "largest_mm2": float(sized.max()),
        "stretch_p5_p50_p95": stretch.tolist(),
        "isometry": isometric,
        "normals_turned": record["turned"],
        "frame_costs": record.get("frame_costs"),
        "stats": stats.tolist(),
        "seconds": {name: round(t1 - t0, 1) for (_, t0), (name, t1) in itertools.pairwise(clock)},
        "method": METHOD,
    }, notes


def units(samples, band=False, volume=None):
    """Every unit of sheet nodes whose pieces are missing or were made by other code."""
    for nodes in sorted(stage_entries(NODE_ROUTE, "nodes", "*")):
        sample = nodes.parts[-4]
        if (
            (samples and sample not in samples)
            or (volume is not None and nodes.parts[-3] != volume)
            or not stage_queue.done(nodes)
            or (band and not nodes.name.startswith("band-"))
        ):
            continue
        if not band and nodes.name.startswith("band-") and any(nodes.parent.glob("rows-*")):
            continue
        out = LEVEL_ROUTE / sample / nodes.parts[-3] / "pieces" / nodes.name
        if not stage_queue.done(out, METHOD, SAME_OUTPUT, UNIT_NEEDS):
            yield nodes, out


def share_nodes(nodes):
    """The scan folder points at its sheet nodes, kept where the first route put them (data_roots.NODE_ROUTE)."""
    link = stage_folder(LEVEL_ROUTE / nodes.parts[-4] / nodes.parts[-3], "nodes")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and not link.exists():
        link.unlink()
    if not link.exists():
        link.symlink_to(os.path.relpath(nodes.parent, link.parent))


def scan_curl(folders):
    """One curl for a scan, by the costs of a slice of each unit."""
    scan = LEVEL_ROUTE / folders[0].parts[-4] / folders[0].parts[-3]
    kept = scan / "curl.json"
    if kept.exists():
        return json.loads(kept.read_text())["curl"]
    costs = {"1": 0.0, "-1": 0.0}
    for nodes in folders:
        low, high = stage_queue.record_of(nodes)["cell_planes"]
        middle = (low + high) // 2
        n, theta, _, _, pairs, field, _ = level_pieces.prepare([nodes], (middle - CURL_PLANES, middle + CURL_PLANES))
        level_pieces.quick_pieces(n, theta, field.pop("edges"), pairs, level_pieces.pg.node_quality(n), field)
        for key, cost in field["frame_costs"].items():
            costs[key] += cost * len(n)
        print("curl", nodes.name, field["frame_costs"], flush=True)
    curl = 1 if costs["1"] < costs["-1"] else -1
    scan.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps({"curl": curl, "costs": costs}))
    return curl


def run_unit(nodes, out, curl=None):
    record = stage_queue.record_of(nodes)
    lo, hi = record["cell_planes"]
    with stage_queue.building(out) as part:
        summary, notes = extract(
            [nodes],
            part,
            record["sample"],
            record["volume"],
            record["scan_voxel_um"],
            record["level"],
            shared=[(lo, lo + SEAM_PLANES), (hi - SEAM_PLANES, hi)],
            curl=curl,
            block=READ_BLOCK_CELLS,
            overlap=READ_OVERLAP_CELLS,
        )
        np.savez(part / "levels.npz", **notes.pop("piece_terms"))
        stage_queue.finish(part, summary | notes)
    print(
        record["sample"],
        nodes.name,
        len(summary["pieces"]),
        "pieces",
        round(summary["area_cm2"], 1),
        "cm2,",
        round(summary["area_in_1cm2_pieces"], 1),
        "cm2 in pieces of 1 cm2,",
        summary["pieces_4cm2"],
        "of 4 cm2, largest",
        round(summary["largest_mm2"] / 100, 1),
        "cm2,",
        summary["seconds"],
        flush=True,
    )


def join_scan(scan):
    """Pieces of a scan's consecutive chunks joined across every seam, once all its chunks are unwrapped."""
    chunks = sorted(stage_folder(scan, "nodes").glob("rows-*"))
    units = [scan / "pieces" / c.name for c in chunks]
    if len(units) < 2 or not all(
        stage_queue.done(u, METHOD, SAME_OUTPUT, UNIT_NEEDS) and stage_queue.done(c) for u, c in zip(units, chunks)
    ):
        return
    method = stage_queue.method_of(HERE / "piece_joiner.py", *[u / stage_queue.RECORD for u in units])
    out = scan / "pieces" / "joints"
    if stage_queue.done(out, method):
        return
    start = time.time()
    seams = []
    for k in range(len(units) - 1):
        seam = stage_queue.record_of(chunks[k])["cell_planes"][1]
        seams.append(seam_joins(chunks[k : k + 2], seam, [boundary_labels(units[k]), boundary_labels(units[k + 1])]))
    counts = [int(np.load(u / "boundary.npz")["pieces"]) for u in units]
    labels = join(counts, seams)
    with stage_queue.building(out) as part:
        np.save(part / "sheets.npy", labels.astype(np.int32))
        stage_queue.finish(
            part,
            {
                "units": [u.name for u in units],
                "pieces": counts,
                "joins": [len(j) for j in seams],
                "sheets": int(labels.max()) + 1,
                "seconds": round(time.time() - start, 1),
                "method": method,
            },
        )
    print(
        scan.parts[-2],
        "joined",
        len(units),
        "chunks:",
        sum(len(j) for j in seams),
        "joins,",
        int(labels.max()) + 1,
        "sheets of",
        sum(counts),
        "pieces",
        flush=True,
    )


def run(samples=(), band=False, *, volume=None):
    curls, failed = {}, 0
    for nodes, out in units(set(samples), band, volume):
        share_nodes(nodes)
        rows = sorted(
            p.parent for p in nodes.parent.glob("rows-*/record.json") if not p.parent.name.endswith(".building")
        )
        scan = nodes.parts[-3]
        if scan not in curls:
            curls[scan] = scan_curl(rows) if len(rows) > 1 and not band else None
        with stage_queue.claimed(out) as mine:
            if mine:
                try:
                    run_unit(nodes, out, curls[scan])
                except (MemoryError, RuntimeError, ValueError, OSError) as error:
                    print("failed", nodes, repr(error)[:200], flush=True)
                    failed += 1
    for scan in sorted(LEVEL_ROUTE.glob("*/*")):
        if scan.is_dir() and (not samples or scan.parent.name in samples) and (volume is None or scan.name == volume):
            join_scan(scan)
            scroll_frame.solve(scan, METHOD, SAME_OUTPUT)
    if failed:
        raise RuntimeError(f"unwrapping failed on {failed} unit(s). Rerun to resume.")


def rows_of(sample):
    return sorted(stage_entries(NODE_ROUTE, "nodes", "rows-*", sample=sample))


def one_unit(sample, name):
    """One unit of a scan by itself, for a board job sized to it."""
    nodes = next(n for n in rows_of(sample) if n.name == name)
    share_nodes(nodes)
    out = LEVEL_ROUTE / sample / nodes.parts[-3] / "pieces" / name
    with stage_queue.claimed(out) as mine:
        if mine and not stage_queue.done(out, METHOD, SAME_OUTPUT, UNIT_NEEDS):
            run_unit(nodes, out, scan_curl(rows_of(sample)))


def finish(sample):
    """What follows a scan's units."""
    scan = LEVEL_ROUTE / sample / rows_of(sample)[0].parts[-3]
    join_scan(scan)
    scroll_frame.solve(scan, METHOD, SAME_OUTPUT)
    scroll_layout.run(sample, scan.name)


if __name__ == "__main__":
    actions = {"curl": lambda sample: print(scan_curl(rows_of(sample))), "unit": one_unit, "finish": finish}
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(
            "Usage: ./run.sh unwrapping.level.run_levels {run [SCROLL ...],curl SCROLL,unit SCROLL UNIT,finish SCROLL}"
        )
    elif sys.argv[1] in actions:
        actions[sys.argv[1]](*sys.argv[2:])
    else:
        run(sys.argv[2:])
