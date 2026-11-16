"""Labels of published surfaces carried onto our own pieces."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ni
from scipy.spatial import cKDTree

from calibration.ink.surface_ink_retention import (
    PAD,
    STEP,
    bricks_of,
    own_pixels,
    own_stacks,
)
from calibration.unwrap.sheet_controls import load_mesh, meshes
from ink.datasets.labelled_cases import load_truth
from ink.model.input_stack import face_layer_of
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points
from storage.brick_store import BrickStore
from storage.data_roots import (
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    OWN_MESH_FINE_LETTERS,
    OWN_MESH_LABELS,
    PUBLISHED_PAIRS,
    calibration_output,
)

ROUTES = {
    "first": calibration_output("surface_ink"),
    "fresh": calibration_output("surface_ink_fresh"),
    "level": calibration_output("surface_ink_level"),
}
ROUTE_FOLDERS = {"first": "grown_route", "fresh": "grown_route_fresh_nodes", "level": "level_route"}
KINDS = {"labels": OWN_MESH_LABELS, "fine": OWN_MESH_FINE_LETTERS}
NEAR_UM = 100.0
LEAST_POINTS = 2000
CELL, SMOOTH_CELLS = 4, 2.0
SAMPLED_PIXELS = 40000
MOST_LAYER_OFFSET = 8
CENTRE = 0.5  # A published label pixel shows the surface at the render pixel's centre.


def truth_of(case):
    """A case's ink and known pixels: its hand labels, or a published window's fine-map stroke cores and the blank far from writing."""
    if case.get("truth") == "fine-map":
        from ink.datasets.agreed_targets import of_published

        return of_published(Path(case["window"]))
    return load_truth(case)


def fine_cases_of(sample, volume):
    """Published windows of a scan with fine-map ink, each as a case of its own."""
    plan = json.loads((PUBLISHED_PAIRS / "plan.json").read_text())
    for _, meta, y, x in plan:
        scan = Path(meta["layers"]).name.removesuffix(".zarr")
        window = PUBLISHED_PAIRS / meta["sample"] / f"{meta['segment']}-{y}-{x}.npy"
        if meta["sample"] == sample and scan.endswith(volume) and window.exists():
            height, width = np.load(window, mmap_mode="r").shape[1:]
            yield {
                "case_id": f"{meta['sample']}-{meta['segment']}-{y}-{x}",
                "sample": meta["sample"],
                "segment": meta["segment"],
                "scan": scan,
                "box": [y, y + height, x, x + width],
                "voxel_um": float(scan.split("um")[0]),
                "truth": "fine-map",
                "window": str(window),
            }


def case_points(case, everywhere=False):
    """Every supervised point of a case at the scored step, or every point of its mesh to place pixels by."""
    ink, known = truth_of(case)
    y0, _, x0, _ = case["box"]
    folder = next(
        path
        for name, path in meshes(case["sample"], case["scan"].split("volume-")[1])
        if name.startswith(case["segment"])
    )
    meta = json.loads((folder / "meta.json").read_text())
    mesh = load_mesh(folder)
    rows, columns = np.meshgrid(np.arange(0, ink.shape[0], STEP), np.arange(0, ink.shape[1], STEP), indexing="ij")
    grid = [(rows + y0 + CENTRE) * meta["scale"][1], (columns + x0 + CENTRE) * meta["scale"][0]]
    good = ni.map_coordinates((mesh[0] > 0).astype(np.float32), grid, order=1, cval=0) > 0.999
    use = good if everywhere else good & known[rows, columns]
    xyz = np.stack([ni.map_coordinates(p, grid, order=1, cval=-1) for p in mesh], axis=-1)
    return xyz[use], ink[rows, columns][use], rows[use], columns[use]


def cases_of(sample, volume):
    """Labelled cases of a scan near 9 um."""
    for path in sorted(LABELLED_CASES.glob(f"{sample}-*/case.json")):
        case = json.loads(path.read_text())
        if case["scan"].endswith(volume) and 7 <= case["voxel_um"] <= 10:
            yield case


def published_stack(case):
    path = Path(case["window"]) if "window" in case else LABELLED_CASE_STACKS / f"{case['case_id']}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    store = BrickStore(LABELLED_CASES / case["case_id"] / "layers")
    y0, y1, x0, x1 = case["box"]
    return store.read((0, store.shape[0]), (y0, y1), (x0, x1))


def published_pixels(r, c, rows, columns, top, left, height, width):
    """For each pixel of a stack's area its published row and column, and whether a label reaches it."""
    shape = (height // CELL + 2, width // CELL + 2)
    where = ((r - top) // CELL, (c - left) // CELL)
    weight, row_sum, column_sum = np.zeros((3, *shape))
    np.add.at(weight, where, 1)
    np.add.at(row_sum, where, rows)
    np.add.at(column_sum, where, columns)
    spread = ni.gaussian_filter(weight, SMOOTH_CELLS)
    cells = [ni.gaussian_filter(total, SMOOTH_CELLS) / np.maximum(spread, 1e-6) for total in (row_sum, column_sum)]
    seen = ni.binary_erosion(ni.binary_closing(weight > 0, iterations=2), iterations=1)
    grid = np.meshgrid(np.arange(height) / CELL, np.arange(width) / CELL, indexing="ij")
    return [ni.map_coordinates(c_, grid, order=1) for c_ in cells] + [
        ni.map_coordinates(seen.astype(np.float32), grid, order=0) > 0.5
    ]


def layer_agreement(own, published, rng):
    """Best correlation of raw values between a layer of each stack, summed over five layers at one offset, in each direction, and the same with the pixels shuffled."""

    def scored(a, b):
        a, b = ((v - v.mean(1, keepdims=True)) / np.maximum(v.std(1, keepdims=True), 1e-6) for v in (a, b))
        table = a @ b.T / a.shape[1]
        return max(
            (float(np.mean(np.diagonal(table, offset))), int(offset))
            for offset in range(
                max(-MOST_LAYER_OFFSET - 1, 5 - table.shape[0]), min(MOST_LAYER_OFFSET + 1, table.shape[1] - 5) + 1
            )
        )

    forward, backward = scored(own, published), scored(own[::-1], published)
    best = max(forward, backward)
    return {
        "correlation": round(best[0], 3),
        "offset": best[1],
        "reversed": backward > forward,
        "shuffled": round(scored(own[:, rng.permutation(own.shape[1])], published)[0], 3),
    }


def kept_windows(sample, route_folder, root=OWN_MESH_LABELS):
    """Kept windows of our own pieces whose raw values place them on the labelled sheet, with their records and carried labels."""
    from ink.training.model_training import LEAST_CORRELATION, LEAST_OVER_SHUFFLED

    for path in sorted((Path(root) / route_folder / sample).glob("*.npy")):
        row = json.loads(path.with_suffix(".json").read_text())
        labels = np.load(path.with_name(f"{path.stem}-labels.npz"))
        if "published" in labels and row["correlation"] >= max(
            LEAST_CORRELATION, LEAST_OVER_SHUFFLED * row["shuffled"]
        ):
            yield path, row, labels


def carried(case, folder, stack, um, rng, cells=sampler.MESH_SMOOTHING_CELLS):
    """A case's labels carried onto our stack of a piece sampled at a smoothing: the stack cut around them, the labels, and how its layers meet the published ones."""
    points, _, rows, columns = case_points(case, everywhere=True)
    r, c, touching = own_pixels(folder, points, um, cells)
    r, c = np.minimum(r[touching], stack.shape[1] - 1), np.minimum(c[touching], stack.shape[2] - 1)
    top, left = max(int(r.min()) - PAD, 0), max(int(c.min()) - PAD, 0)
    piece, stack = stack, np.ascontiguousarray(stack[:, top : int(r.max()) + 1 + PAD, left : int(c.max()) + 1 + PAD])
    at_row, at_column, seen = published_pixels(r, c, rows[touching], columns[touching], top, left, *stack.shape[1:])
    ink, known = truth_of(case)
    at_row, at_column = (
        np.clip(at_row.round().astype(np.int64), 0, ink.shape[0] - 1),
        np.clip(at_column.round().astype(np.int64), 0, ink.shape[1] - 1),
    )
    sheet = stack[stack.shape[0] // 2] > 0
    known_here = seen & known[at_row, at_column] & sheet
    ys, xs = np.nonzero(known_here)
    some = rng.permutation(len(ys))[:SAMPLED_PIXELS]
    published = np.asarray(published_stack(case))[:, at_row[ys[some], xs[some]], at_column[ys[some], xs[some]]].astype(
        np.float32
    )
    row = layer_agreement(stack[:, ys[some], xs[some]].astype(np.float32), published, rng) | {
        "pixels": int(known_here.sum()),
        "shape": list(stack.shape),
        "window": [top, left],
    }
    row["face"] = int(face_layer_of(piece[::-1] if row["reversed"] else piece))
    labels = {
        "ink": ink[at_row, at_column] & known_here,
        "known": known_here,
        "published": np.stack([ys, xs, at_row[ys, xs], at_column[ys, xs]]).astype(np.int32),
    }
    return stack[::-1] if row["reversed"] else stack, labels, row


def keep_window(out, name, stack, labels, row):
    """A carried window, its labels and its record, side by side."""
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.npy", stack)
    np.savez_compressed(out / f"{name}-labels.npz", **labels)
    (out / f"{name}.json").write_text(json.dumps(row))


def census_path(route, band, kind="labels"):
    return KINDS[kind] / ROUTE_FOLDERS[route] / "census" / f"{band}.json"


def labels_folder(route, sample, kind="labels"):
    """Where a route's carried labels of a kind live."""
    return KINDS[kind] / ROUTE_FOLDERS[route] / sample


def cases_of_kind(sample, volume, kind):
    return fine_cases_of(sample, volume) if kind == "fine" else cases_of(sample, volume)


def build(band, route="first", kind="labels"):
    """Each pair of the census, each piece sampled once for all its cases."""
    base = ROUTES[route] / band / "pieces"
    record = record_of(base)
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    axis = axis_points(sample, volume)
    out = labels_folder(route, sample, kind)
    out.mkdir(parents=True, exist_ok=True)
    cases = {case["case_id"]: case for case in cases_of_kind(sample, volume, kind)}
    piece, kept, bricks = None, None, f"labels-{route}-{band}-{kind}"
    for pair in sorted(
        json.loads(census_path(route, band, kind).read_text()), key=lambda p: (p["piece"], -p["points"])
    ):
        name = f"{pair['case']}-{pair['piece']:04d}"
        if (out / f"{name}.json").exists():
            continue
        folder = base / f"piece-{pair['piece']:04d}"
        if pair["piece"] != piece:
            piece = pair["piece"]
            kept = [
                s
                for _, s in own_stacks(
                    folder,
                    sample,
                    volume,
                    axis,
                    bricks,
                    (sampler.MESH_SMOOTHING_CELLS,),
                    sampler.BRIDGE_CELLS,
                    keep=True,
                )
            ][-1]
        stack, labels, found = carried(cases[pair["case"]], folder, kept, um, np.random.default_rng(3))
        row = pair | found
        keep_window(out, name, stack, labels, row)
        print(json.dumps(row), flush=True)
    shutil.rmtree(bricks_of(bricks), ignore_errors=True)


def scored(route, sample, weights):
    """AUC of a model on a route's carried labels, each pair read from its written face."""
    from calibration.ink.ink_yield import rank_auc
    from calibration.ink.own_mesh_windows import face_read
    from ink.model.tile_reader import TileReader
    from ink.training.model_training import (
        LEAST_CORRELATION,
        LEAST_OVER_SHUFFLED,
        NEVER_TAUGHT,
    )

    reader = TileReader(weights)
    rows, reads = [], {}
    for path in sorted(labels_folder(route, sample).glob("*.npy")):
        row = json.loads(path.with_suffix(".json").read_text())
        if path.stem in NEVER_TAUGHT or row["correlation"] < max(
            LEAST_CORRELATION, LEAST_OVER_SHUFFLED * row["shuffled"]
        ):
            continue
        labels = np.load(path.with_name(f"{path.stem}-labels.npz"))
        read = face_read(reader, np.load(path))
        ys, xs, prow, pcol = labels["published"]
        values, ink = read[ys, xs], labels["ink"][ys, xs]
        rows.append(
            {
                "pair": path.stem,
                "pixels": len(ink),
                "ink_share": round(float(ink.mean()), 3),
                "auc": round(float(rank_auc(values[ink], values[~ink])), 4),
            }
        )
        for key, value, is_ink in zip(
            zip([row["case"]] * len(ys), prow.tolist(), pcol.tolist()), values.tolist(), ink.tolist()
        ):
            reads.setdefault(key, []).append((value, is_ink))
        print(route, json.dumps(rows[-1]), flush=True)
    return rows, {k: (float(np.mean([v for v, _ in r])), r[0][1]) for k, r in reads.items()}


def compare(sample, weights, out):
    """Every route with carried labels read by one model."""
    from calibration.ink.ink_yield import rank_auc

    found = {route: scored(route, sample, weights) for route in ROUTES if labels_folder(route, sample).exists()}
    shared = sorted(set.intersection(*(set(reads) for _, reads in found.values())))
    summary = {"weights": str(weights), "sample": sample, "shared_pixels": len(shared)}
    for route, (rows, reads) in found.items():
        values, ink = (np.array(column) for column in zip(*reads.values()))
        both, both_ink = (
            (np.array(column) for column in zip(*(reads[k] for k in shared)))
            if shared
            else (np.zeros(0), np.zeros(0, bool))
        )
        summary[route] = {
            "pairs": rows,
            "pixels": len(reads),
            "auc_all": round(float(rank_auc(values[ink], values[~ink])), 4),
            "auc_shared": round(float(rank_auc(both[both_ink], both[~both_ink])), 4) if len(shared) else None,
        }
        print(
            route,
            "pixels",
            len(reads),
            "AUC on all it covers",
            summary[route]["auc_all"],
            "on shared",
            summary[route]["auc_shared"],
            flush=True,
        )
    Path(out).write_text(json.dumps(summary, indent=1))


def census(band, route="first", kind="labels"):
    """Of each case of a kind on a band's scan, how many points lie on each piece of the band."""
    base = ROUTES[route] / band / "pieces"
    record = record_of(base)
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    trees = {}
    for folder in sorted(base.glob("piece-*")):
        xyz, _ = sampler.load_surface(folder)
        if (xyz[0] > 0).any():
            trees[folder] = cKDTree(xyz[:, xyz[0] > 0].T)
    rows = []
    for case in cases_of_kind(sample, volume, kind):
        points, ink, _, _ = case_points(case)
        found = 0
        for folder, tree in trees.items():
            near = int((tree.query(points[::25], workers=2)[0] * um < NEAR_UM).sum())
            if near:
                _, _, touching = own_pixels(folder, points, um)
                if touching.sum() >= LEAST_POINTS:
                    rows.append(
                        {
                            "case": case["case_id"],
                            "piece": int(folder.name.split("-")[1]),
                            "points": int(touching.sum()),
                            "ink_share": round(float(ink[touching].mean()), 3),
                        }
                    )
                    found += int(touching.sum())
        print(case["case_id"], "points", len(points), "on our pieces", found, flush=True)
    census_path(route, band, kind).parent.mkdir(parents=True, exist_ok=True)
    census_path(route, band, kind).write_text(json.dumps(rows, indent=1))
    print(band, "pieces", len(trees), "pairs", len(rows), "points", sum(r["points"] for r in rows), flush=True)


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    if command == "compare":
        compare(rest[0], Path(rest[1]), rest[2])
    else:
        {"build": build, "census": census}[command](*rest)
