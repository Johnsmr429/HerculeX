"""Labelled windows of scrolls left out, read on our own pieces and not the published surfaces."""

import json
import sys

import numpy as np

from calibration.ink.ink_yield import rank_auc
from calibration.ink.surface_ink_retention import (
    PAD,
    own_pixels,
    own_stacks,
    published_points,
)
from ink.model.tile_reader import TileReader
from ink.training.model_training import PAST_FACE, face_layer_of
from runner.stage_queue import record_of
from sampling import column_sampling as sampler
from sampling.fibre_faces import recto_face
from sampling.piece_images import axis_points
from storage.data_roots import LABELLED_CASES, SURFACE_RECORDS, calibration_output

WINDOWS = calibration_output("own_mesh_windows")
CONTROLS = {
    "PHerc0500P2": "PHerc0500P2-20250820143440-rows-0012-0024",
    "PHerc0841": "PHerc0841-20250821151531-rows-0050-0062",
}
SET_ASIDE = {"PHerc0841-20260220214732-9.366um-1.2m-113keV-0086"}


def build():
    """Each scored window of the surface study, on the pieces it found there."""
    rows = [
        r
        for r in json.loads((SURFACE_RECORDS / "ink-retention.json").read_text())
        if r["case"].split("-")[0] in CONTROLS
    ]
    for row in rows:
        sample = row["case"].split("-")[0]
        out = WINDOWS / sampler.SAMPLING / sample / f"{row['case']}-{row['piece']:04d}.npz"
        if out.exists():
            continue
        base = calibration_output("surface_ink") / CONTROLS[sample]
        record = record_of(base / "pieces")
        volume, um = record["volume"], record["voxel_um"]
        case = json.loads((LABELLED_CASES / row["case"] / "case.json").read_text())
        _, _, _, points, ink = published_points(case)
        folder = base / "pieces" / f"piece-{row['piece']:04d}"
        r, c, touching = own_pixels(folder, points, um)
        stack = [
            s
            for _, s in own_stacks(
                folder,
                sample,
                volume,
                axis_points(sample, volume),
                f"windows-{sample}-{row['piece']:04d}",
                (sampler.MESH_SMOOTHING_CELLS,),
                sampler.BRIDGE_CELLS,
            )
        ][-1]
        rr, cc = np.minimum(r[touching], stack.shape[1] - 1), np.minimum(c[touching], stack.shape[2] - 1)
        y0, y1, x0, x1 = (
            max(int(rr.min()) - PAD, 0),
            int(rr.max()) + 1 + PAD,
            max(int(cc.min()) - PAD, 0),
            int(cc.max()) + 1 + PAD,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, stack=stack[:, y0:y1, x0:x1], rows=rr - y0, columns=cc - x0, ink=ink[touching])
        print(out.name, stack[:, y0:y1, x0:x1].shape, int(touching.sum()), flush=True)


def face_read(reader, stack):
    first = min(max(face_layer_of(stack) + PAST_FACE, 0), stack.shape[0] - 17)
    return reader.read(np.ascontiguousarray(stack[first : first + 17]))


def summary(names, points, aucs):
    """Scores weighted by points."""
    points, aucs = np.array(points, dtype=float), np.array(aucs, dtype=float)
    kept = np.array([name not in SET_ASIDE for name in names])
    return {
        "own_mesh": round(float((points[kept] * aucs[kept]).sum() / points[kept].sum()), 4),
        "own_mesh_all": round(float((points * aucs).sum() / points.sum()), 4),
        "own_mesh_windows": [round(float(a), 3) for a in aucs],
    }


def scored(weights, sample):
    """AUC of a model on its left-out scroll's windows on our meshes, each read from the face its fibres name, weighted by points."""
    reader = TileReader(weights)
    rows = []
    for path in sorted((WINDOWS / sampler.SAMPLING / sample).glob("*.npz")):
        window = np.load(path)
        stack, r, c, ink = window["stack"], window["rows"], window["columns"], window["ink"]
        read = face_read(reader, stack[::-1] if recto_face(stack)["face"] == "far" else stack)
        rows.append((path.stem, len(ink), rank_auc(read[r, c][ink], read[r, c][~ink])))
    return summary(*zip(*rows)) if rows else {}


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else print(scored(*sys.argv[1:3]))
