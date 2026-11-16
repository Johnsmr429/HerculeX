"""Surfaces and readers scored on known letters, and the features a stacked classifier is taught from.

./run.sh spots.controls BAND [BAND ...]     bands as in datasets/meshes/level_route/census (e.g. PHerc0841-...)

At most MOST_PAIRS pairs a band are scored (rows already written count).

For every census pair of a band (a labelled case and the level-route piece it lies on), as own_mesh_labels.build does:

    original   the piece sampled as production samples it (bridged, smoothed 1.5 cells)
    m7         every node moved onto the published m7 surface prediction by the winners' rule (surfaces.snapped),
               sampled bridged and not smoothed again, as the spots' m7 surface is
    gated      (when spots.combine is present) m7 where a node's snap moved it at most GATE_VOXELS, else the original

The original is cut round the case and its labels carried by own_mesh_labels.carried (published points onto the
nearest pixel of that surface within 60 µm). The snap keeps every node's grid place, so the other surfaces are cut at
the same window and scored at the same pixels with the same labels (a surface whose render differs in size is
carried on its own, marked same_pixels_as_original false). Every reader of READERS reads each at the production
window (face_layer_of + 2) shifted by each of SHIFTS layers. Per reader: the rank AUC of ink against blank on the
known pixels at shift 0 and for the mean of shifts -2..+2. Every reader's value at every shift at up to KEEP_PIXELS
known pixels is kept with its label and its ys, xs in the window, for spots.stacker and for scoring combinations of
the surfaces. Pairs too weakly tied to their published layers are skipped as own_mesh_labels.scored skips them. Rows
go to SPOT_CANDIDATES/controls/BAND.jsonl, features to controls/features/BAND/PAIR-SURFACE.npz.
"""

import json
import shutil
import sys

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.surface_ink_retention import bricks_of, own_stacks
from ink.datasets.own_mesh_labels import ROUTES, carried, cases_of, census_path
from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of
from ink.model.tile_reader import TileReader
from ink.training.model_training import LEAST_CORRELATION, LEAST_OVER_SHUFFLED, NEVER_TAUGHT
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points, centre_at
from spots.surfaces import Prediction, snapped, write_tifxyz
from spots.sweep import WEIGHTS
from storage.data_roots import SPOT_CANDIDATES, TRAINED_WEIGHTS

try:  # the gated surface (spots.combine): m7 where the snap moved little, else the original
    from spots.combine import gate_report, gated
except ImportError:
    gated = gate_report = None

OUT = SPOT_CANDIDATES / "controls"
GATE_VOXELS = 3
MOST_PAIRS = 12  # per band
SHIFTS = tuple(range(-4, 5))
BAND_SHIFTS = (-2, -1, 0, 1, 2)
KEEP_PIXELS = 30000
READERS = {
    "reader-v3": WEIGHTS["reader-v3"],
    "student": WEIGHTS["student"],
    "own-mesh": WEIGHTS["own-mesh"],
    "thin-marks": TRAINED_WEIGHTS / "reader-single-depth-thin-marks-r1.pth",
    "stroke-core": TRAINED_WEIGHTS / "reader-stroke-core-without-0841.pth",
    "own-reads-stroke-core": TRAINED_WEIGHTS / "reader-own-reads-stroke-core-without-0841.pth",
}


def features(readers, stack, ys, xs):
    """(pixels, readers, shifts) of every reader at every shift round the production window, at the given pixels."""
    first = min(max(face_layer_of(stack) + PAST_FACE, 0), stack.shape[0] - DEPTH)
    out = np.zeros((len(ys), len(readers), len(SHIFTS)), np.float32)
    for r, reader in enumerate(readers.values()):
        for s, shift in enumerate(SHIFTS):
            at = min(max(first + shift, 0), stack.shape[0] - DEPTH)
            out[:, r, s] = reader.read(np.ascontiguousarray(stack[at : at + DEPTH]))[ys, xs]
    return out


def band_scores(band, readers):
    base = ROUTES["level"] / band / "pieces"
    record = record_of(base)
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    axis = axis_points(sample, volume)
    prediction = Prediction(sample, volume)
    cases = {case["case_id"]: case for case in cases_of(sample, volume)}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{band}.jsonl"
    kept = OUT / "features" / band
    kept.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["pair"] for line in out.read_text().splitlines()} if out.exists() else set()
    work = OUT / "work" / band
    bricks = f"spot-controls-{band}"
    pairs = sorted(json.loads(census_path("level", band).read_text()), key=lambda p: (p["piece"], -p["points"]))
    for pair in pairs:
        name = f"{pair['case']}-{pair['piece']:04d}"
        if len(done) >= MOST_PAIRS:
            break
        if name in done or name in NEVER_TAUGHT or pair["case"] not in cases:
            continue
        folder = base / f"piece-{pair['piece']:04d}"
        xyz, meta = sampler.load_surface(folder)
        snapped_folder = work / f"piece-{pair['piece']:04d}-m7"
        if not (snapped_folder / "meta.json").exists():
            middle = np.median(xyz[:, xyz[0] > 0], axis=1)
            inward = sampler.grid_normals(xyz, centre_at(axis, middle[2], middle), meta["stride"] * meta["voxel_um"])
            moved, report = snapped(xyz, inward, prediction)
            write_tifxyz(snapped_folder, moved, meta | {"snap": report})
        row = {
            "pair": name,
            "band": band,
            "sample": sample,
            "snap": json.loads((snapped_folder / "meta.json").read_text())["snap"],
        }
        surfaces = [("original", folder, sampler.MESH_SMOOTHING_CELLS), ("m7", snapped_folder, 0)]
        if gated is not None:
            gated_folder = work / f"piece-{pair['piece']:04d}-gated-{GATE_VOXELS}"
            moved, _ = sampler.load_surface(snapped_folder)
            row["gate"] = gate_report(xyz, moved, GATE_VOXELS)
            if not (gated_folder / "meta.json").exists():
                write_tifxyz(gated_folder, gated(xyz, moved, GATE_VOXELS), meta | {"gate": row["gate"]})
            surfaces.append(("gated", gated_folder, 0))
        rng = np.random.default_rng(3)
        first = None  # the original's labels and window: the other surfaces are cut at the same pixels
        for surface, source, cells in surfaces:
            stack = next(
                s
                for _, s in own_stacks(source, sample, volume, axis, bricks, (cells,), sampler.BRIDGE_CELLS, keep=True)
            )
            same = first is not None and stack.shape == first[2]
            if same:
                labels, found, _ = first
                (top, left), (_, h, w) = found["window"], found["shape"]
                cut = stack[:, top : top + h, left : left + w]
                cut = cut[::-1] if found["reversed"] else cut
            else:
                cut, labels, found = carried(cases[pair["case"]], source, stack, um, rng, cells=cells)
            if found["correlation"] < max(LEAST_CORRELATION, LEAST_OVER_SHUFFLED * found["shuffled"]):
                row[surface] = {"skipped": "weakly tied to the published layers", "correlation": found["correlation"]}
                continue
            if surface == "original":
                first = (labels, found, stack.shape)
            ys, xs, _, _ = labels["published"]
            ink = labels["ink"][ys, xs]
            some = np.random.default_rng(7).permutation(len(ys))[:KEEP_PIXELS]
            f = features(readers, cut, ys[some], xs[some])
            np.savez_compressed(
                kept / f"{name}-{surface}.npz",
                features=f.astype(np.float16),
                ink=ink[some],
                readers=list(readers),
                shifts=list(SHIFTS),
                ys=ys[some],
                xs=xs[some],
                window=np.array(found["window"]),
                same_pixels_as_original=same,
            )
            band_idx = [SHIFTS.index(s) for s in BAND_SHIFTS]
            row[surface] = {
                "pixels": int(len(ys)), "kept": int(len(some)), "ink_share": round(float(ink.mean()), 3),
                "correlation": found["correlation"], "offset": found["offset"], "same_pixels_as_original": same,
                "auc": {r: round(float(rank_auc(f[ink[some], k, SHIFTS.index(0)], f[~ink[some], k, SHIFTS.index(0)])), 4) for k, r in enumerate(readers)},
                "auc_5_depths": {r: round(float(rank_auc(f[ink[some], k][:, band_idx].mean(1), f[~ink[some], k][:, band_idx].mean(1))), 4) for k, r in enumerate(readers)},
            }  # fmt: skip
        done.add(name)
        with out.open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    shutil.rmtree(bricks_of(bricks), ignore_errors=True)
    return out


def summary(paths):
    """Per scroll and in all, each reader's pixel-weighted AUC (5-depth mean) on each surface, over pairs both scored."""
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    both = [r for r in rows if "auc" in r.get("original", {}) and "auc" in r.get("m7", {})]
    out = {}
    for group in [*sorted({r["sample"] for r in both}), "all"]:
        part = [r for r in both if group in ("all", r["sample"])]
        if not part:
            continue
        weights = np.array([r["original"]["pixels"] for r in part], dtype=float)
        out[group] = {"pairs": len(part)}
        for reader in READERS:
            for s in ("original", "m7"):
                out[group][f"{s} {reader}"] = round(
                    float(np.average([r[s]["auc_5_depths"][reader] for r in part], weights=weights)), 4
                )
    return out


if __name__ == "__main__":
    readers = {name: TileReader(path) for name, path in READERS.items()}
    paths = [band_scores(band, readers) for band in sys.argv[1:]]
    print(json.dumps(summary(paths), indent=1))
