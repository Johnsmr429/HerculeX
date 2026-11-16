"""Whether more models and more depths read better."""

import json

import numpy as np
from PIL import Image

from calibration.ink.ink_yield import rank_auc
from calibration.ink.own_mesh_windows import WINDOWS, summary
from calibration.text.spot_tests import piece_stack, shown
from ink.model.tile_reader import TileReader
from ink.reading.piece_reading import WEIGHTS
from ink.training.model_training import DEPTH, PAST_FACE, face_layer_of
from sampling import column_sampling as sampler
from sampling.fibre_faces import recto_face
from storage.data_roots import (
    INK_RECORDS,
    READER_SEED43,
    ROUTE_OUTPUTS,
    TRAINED_WEIGHTS,
    calibration_output,
    stage_folder,
)

SHIFTS = (-2, 0, 2)
LEFT_OUT = {
    "PHerc0841": ("F-without-0841.pth", "without-0841-seed43.pth", "e10-stem-0841.pth", "e5-muon-0841.pth"),
    "PHerc0500P2": ("F-without-0500P2.pth", "e10b-stem-0500P2.pth"),
}
TOP_RATIO = 1.2
SAMPLE, VOLUME = "PHerc0211", "20250821151803"


def maps_of(readers, stack):
    """Each model's read at each depth, (models, depths, height, width), from the face the fibres name."""
    stack = stack[::-1] if recto_face(stack)["face"] == "far" else stack
    face = face_layer_of(stack) + PAST_FACE
    out = np.zeros((len(readers), len(SHIFTS), *stack.shape[1:]), dtype=np.float32)
    for k, shift in enumerate(SHIFTS):
        first = min(max(face + shift, 0), stack.shape[0] - DEPTH)
        block = np.ascontiguousarray(stack[first : first + DEPTH])
        for m, reader in enumerate(readers):
            out[m, k] = reader.read(block)
    return out


def ways(maps):
    """The ways of combining a window's maps."""
    middle = SHIFTS.index(0)
    return {
        "first model": maps[0, middle],
        "models averaged": maps[:, middle].mean(axis=0),
        "first two multiplied": maps[0, middle] * maps[1, middle],
        "depths averaged": maps[0].mean(axis=0),
        "models and depths averaged": maps.mean(axis=(0, 1)),
    }


def scored(sample):
    """Each way on a scroll's own-mesh windows, models that left the scroll out."""
    readers = [TileReader(TRAINED_WEIGHTS / name) for name in LEFT_OUT[sample]]
    names, points, aucs = [], [], {}
    for path in sorted((WINDOWS / sampler.SAMPLING / sample).glob("*.npz")):
        window = np.load(path)
        r, c, ink = window["rows"], window["columns"], window["ink"]
        names.append(path.stem)
        points.append(len(ink))
        for way, read in ways(maps_of(readers, window["stack"])).items():
            aucs.setdefault(way, []).append(rank_auc(read[r, c][ink], read[r, c][~ink]))
        print(sample, path.stem[-4:], {way: round(float(a[-1]), 3) for way, a in aucs.items()}, flush=True)
    return {way: summary(names, points, a) for way, a in aucs.items()}


def top_pieces():
    """The census pieces of PHerc0211 that read most like text."""
    census = json.loads((calibration_output("scroll_triage") / "census-pieces.json").read_text())
    return sorted((row for row in census if row["ratio"] >= TOP_RATIO), key=lambda row: -row["ratio"])


def read_top():
    """The top pieces read every way, as images for the eye."""
    placed = {
        (p["unit"], p["piece"]): p
        for p in json.loads((stage_folder(ROUTE_OUTPUTS / SAMPLE / VOLUME, "sheet") / "layout.json").read_text())[
            "pieces"
        ]
    }
    readers = [TileReader(WEIGHTS), TileReader(READER_SEED43)]
    out = calibration_output("model_agreement") / SAMPLE
    out.mkdir(parents=True, exist_ok=True)
    for row in top_pieces():
        unit, piece = row["piece"].rsplit("-", 1)
        stack = piece_stack(SAMPLE, VOLUME, unit, int(piece), placed.get((unit, int(piece)), {}).get("turn", 1))
        for way, read in ways(maps_of(readers, np.asarray(stack))).items():
            Image.fromarray(shown(read)).save(out / f"{row['piece']}-{way.replace(' ', '-')}.png", compress_level=3)
        print(row["piece"], round(row["ratio"], 2), flush=True)


def run():
    (INK_RECORDS / "model-agreement.json").write_text(
        json.dumps({sample: scored(sample) for sample in LEFT_OUT}, indent=1)
    )
    read_top()


if __name__ == "__main__":
    run()
