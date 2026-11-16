"""Our own windows of a labelled scroll sampled again under other mesh smoothings, labels carried at each smoothing, kept apart from the training data."""

import json
import sys
from pathlib import Path

import numpy as np

from calibration.ink.agreed_truth import LEAST_INK, run
from calibration.ink.surface_ink_retention import own_stacks
from ink.datasets.own_mesh_labels import ROUTE_FOLDERS, ROUTES, carried, keep_window
from ink.model.input_stack import face_layer_of
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.piece_images import axis_points
from storage.data_roots import LABELLED_CASES, OWN_MESH_LABELS, OWN_MESH_SMOOTHING

OUT = OWN_MESH_SMOOTHING
RESAMPLES = 4000


def folder_of(cells, route, sample):
    return OUT / f"{cells:g}" / ROUTE_FOLDERS[route] / sample


def build(band, route, *cells):
    """Each kept window of a band sampled again at every smoothing in one fetch, its labels carried at that smoothing; the production smoothing must give back the kept window."""
    cells = sorted({float(c) for c in cells} | {sampler.MESH_SMOOTHING_CELLS})
    base = ROUTES[route] / band / "pieces"
    record = record_of(base)
    sample, volume, um = record["sample"], record["volume"], record["voxel_um"]
    axis = axis_points(sample, volume)
    kept_folder = OWN_MESH_LABELS / ROUTE_FOLDERS[route] / sample
    log = []
    for path in sorted(kept_folder.glob("*.json")):
        row = json.loads(path.read_text())
        name = path.stem
        if all((folder_of(c, route, sample) / f"{name}.npy").exists() for c in cells):
            continue
        case = json.loads((LABELLED_CASES / row["case"] / "case.json").read_text())
        piece = base / f"piece-{row['piece']:04d}"
        kept = np.load(kept_folder / f"{name}.npy", mmap_mode="r")
        entry = {"name": name, "kept_face": int(face_layer_of(kept))}
        stacks = own_stacks(
            piece, sample, volume, axis, f"smoothing-{route}-{sample}-{row['piece']:04d}", cells, sampler.BRIDGE_CELLS
        )
        for sigma, stack in stacks:
            stack, labels, found = carried(case, piece, stack, um, np.random.default_rng(3), sigma)
            keep_window(folder_of(sigma, route, sample), name, stack, labels, row | found)
            entry[f"{sigma:g}"] = {"face": int(face_layer_of(stack)), "shape": list(stack.shape)} | {
                key: found[key] for key in ("correlation", "pixels")
            }
            if sigma == sampler.MESH_SMOOTHING_CELLS:
                entry["control_equal"] = bool(stack.shape == kept.shape and np.array_equal(stack, kept))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if not entry.get("control_equal"):
            raise SystemExit(f"{name}: the production smoothing does not give back the kept window; stopping")
    (OUT / f"{band}-{route}.json").write_text(json.dumps(log, indent=1))


def paired(control_path, other_path, model_key):
    """Each window's AUC under another smoothing against the production one, weighted by agreed ink, with a 90% interval from resampling windows."""
    control, other = (json.loads(Path(p).read_text()) for p in (control_path, other_path))
    rng = np.random.default_rng(0)
    rows = []
    for group in ("level_route shift +0", "level_route oracle depth"):
        a = {r["name"]: r for r in control[f"{model_key} {group}"]["rows"] if r["ink"] >= LEAST_INK}
        b = {r["name"]: r for r in other[f"{model_key} {group}"]["rows"] if r["ink"] >= LEAST_INK}
        names = [n for n in a if n in b]
        moves = np.array([b[n]["auc"] - a[n]["auc"] for n in names])
        weights = np.array([min(a[n]["ink"], b[n]["ink"]) for n in names], dtype=float)
        draws = rng.integers(0, len(names), (RESAMPLES, len(names)))
        spread = (moves[draws] * weights[draws]).sum(1) / weights[draws].sum(1)
        rows.append(
            {
                "group": group,
                "windows": len(names),
                "move": round(float(np.average(moves, weights=weights)), 4),
                "interval": [round(float(v), 4) for v in np.percentile(spread, [5, 95])],
                "up": int((moves > 0.002).sum()),
                "down": int((moves < -0.002).sum()),
            }
        )
    return rows


def score(sample, cells, out, *weights):
    """The agreed-truth ruler at the production depth, with its oracle, on the windows of one smoothing."""
    run(sample, out, "0", list(weights), OUT / f"{float(cells):g}")


if __name__ == "__main__":
    {"build": build, "score": score, "paired": lambda *a: print(json.dumps(paired(*a), indent=1))}[sys.argv[1]](
        *sys.argv[2:]
    )
