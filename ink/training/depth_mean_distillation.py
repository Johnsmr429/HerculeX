"""Reader v3 taught to give from one window what its own mean over nearby depths gives, on windows moved and wandering as on our pieces."""

import json
import sys

import numpy as np

from ink.datasets.depth_mean_targets import OWN_PIECES, path_of
from ink.model import tile_reader
from ink.model.input_stack import DEPTH, face_layer_of
from ink.model.wide_reader import WideStem
from ink.training import model_training
from ink.training.reader_v3_start_training import LEFT_OUT, SEED, starting_weights, taught_from
from storage.data_roots import DEPTH_MEAN_TARGETS, OWN_MESH_LABELS, PUBLISHED_PAIRS

STEPS = 4000
LEARNING_RATE = 3e-5
OWN_MESH_SHARE = 0.5
PUBLISHED = 0.5
JITTER = 4
WANDERING, WANDER_LAYERS, WANDER_PX = 0.5, 4.0, 45
INKED = 0.6
OWN_PIECE_SHARE = 0
OWN_SURFACE_PUBLISHED = 0.3
NAME = "reader-depth-mean-student-without-0841"


class Soft:
    """A kept depth-mean map read a slice at a time as probabilities."""

    def __init__(self, path):
        self.bytes = np.load(path, mmap_mode="r")
        self.shape = self.bytes.shape

    def __getitem__(self, where):
        return self.bytes[where].astype(np.float32) / 255


def taught_where(case, kind, name):
    """A window's soft targets and the sheet pixels they cover, with spots to crop round."""
    case.ink = Soft(path_of(kind, name))
    height, width = case.ink.shape
    sheet = np.asarray(case.stack[case.stack.shape[0] // 2, :height, :width]) > 0
    case.known = sheet
    case.inked = np.argwhere(np.asarray(case.ink[:, :]) >= INKED)[::101].copy()
    case.spots = np.argwhere(sheet)[::509].copy()


class StudentCase(model_training.Labelled):
    """A labelled case taught on its teacher's depth-mean map."""

    def __init__(self, case):
        super().__init__(case)
        taught_where(self, "labelled_cases", case["case_id"])


class StudentOwn(model_training.OwnLabelled):
    """One of our own pieces taught on its teacher's depth-mean map."""

    def __init__(self, path):
        super().__init__(path)
        taught_where(self, f"own_mesh_labels/{path.parent.parent.name}/{path.parent.name}", path.stem)


class StudentWindow:
    """A published window taught on its teacher's depth-mean map."""

    def __init__(self, path):
        self.sample, self.name, self.fixed = path.parent.name, path.stem, {}
        self.stack = np.load(path, mmap_mode="r")
        faces = path.with_suffix(".json")
        if not faces.exists():
            faces.write_text(json.dumps([face_layer_of(self.stack), face_layer_of(self.stack[::-1])]))
        self.start, self.back_start = json.loads(faces.read_text())
        taught_where(self, f"published_pairs/{path.parent.name}", path.stem)


class StudentPiece:
    """One of our own whole pieces of a taught scroll with its teacher's depth-mean map."""

    def __init__(self, path):
        self.sample, self.name, self.fixed = path.stem.split("-")[0], path.stem, {}
        self.stack = np.load(path, mmap_mode="r")
        self.start, self.back_start = face_layer_of(self.stack), face_layer_of(self.stack[::-1])
        taught_where(self, "own_pieces", path.stem)


def published_of(left_out):
    found = {}
    for path in sorted(PUBLISHED_PAIRS.glob("PHerc*/*.npy")):
        if path.parent.name != left_out and path_of(f"published_pairs/{path.parent.name}", path.stem).exists():
            found.setdefault(path.parent.name, []).append(StudentWindow(path))
    print("published windows", {k: len(v) for k, v in found.items()}, flush=True)
    return found


def own_cases(left_out):
    found = [
        StudentOwn(p)
        for p in sorted((OWN_MESH_LABELS / "grown_route").glob("PHerc*/*.npy"))
        if p.parent.name.split("-")[0] != left_out
        and path_of(f"own_mesh_labels/grown_route/{p.parent.name}", p.stem).exists()
    ]
    if OWN_PIECE_SHARE:
        found += [
            StudentPiece(p)
            for p in sorted(OWN_PIECES.glob("PHerc*.npy"))
            if p.stem.split("-")[0] != left_out and path_of("own_pieces", p.stem).exists()
        ]
    print("own-mesh windows and pieces", len(found), flush=True)
    return found


def loss_of(torch, logits, ink, known):
    """Cross entropy against the teacher's probabilities on the sheet, least where the student equals the teacher; no overlap term, which would saturate strokes."""
    cross = torch.nn.functional.binary_cross_entropy_with_logits(logits, ink, reduction="none")
    return (cross * known).sum() / known.sum().clamp(min=1.0)


def widened(depth):
    """Training on windows of `depth` layers centred where the 17-layer window is, read by a stem compiled for them."""
    model_training.PAST_FACE -= (depth - model_training.DEPTH) // 2
    model_training.DEPTH = depth
    tile_reader.DepthStem = lambda module, device: WideStem(module, device, depth)


def taught(name=NAME, steps=STEPS, depth=DEPTH):
    """The student, PHerc0841 left out of teacher and student alike."""
    if depth != DEPTH:
        widened(depth)
    if not DEPTH_MEAN_TARGETS.exists():
        raise SystemExit("build the teacher's maps first: ink.datasets.depth_mean_targets")
    model_training.Labelled, model_training.own_cases, model_training.published_of = (
        StudentCase,
        own_cases,
        published_of,
    )
    model_training.loss_of, model_training.backgrounds_of = loss_of, lambda target: []
    model_training.LEARNING_RATE, model_training.STEM_LEARNS = LEARNING_RATE, 0
    model_training.PUBLISHED, model_training.JITTER = PUBLISHED, JITTER
    model_training.BACK_OF_SHEET, model_training.SCRAMBLED = 0.0, 0.0
    model_training.WANDERING, model_training.WANDER_LAYERS, model_training.WANDER_PX = (
        WANDERING,
        WANDER_LAYERS,
        WANDER_PX,
    )
    print(taught_from(starting_weights(), name, LEFT_OUT, LEFT_OUT, SEED, steps, OWN_MESH_SHARE), flush=True)


def own_surface(name, steps=STEPS):
    """The student taught mostly on our own surfaces of the labelled scrolls, whole pieces included."""
    global OWN_PIECE_SHARE, PUBLISHED
    OWN_PIECE_SHARE, PUBLISHED = 1, OWN_SURFACE_PUBLISHED
    taught(name, steps)


if __name__ == "__main__":
    if sys.argv[1:2] == ["own-surface"]:
        own_surface(sys.argv[2], *map(int, sys.argv[3:4]))
    else:
        taught(*sys.argv[1:2], *map(int, sys.argv[2:4]))
