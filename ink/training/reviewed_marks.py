"""Lean post-training on the known letters reviewed by eye on our own mesh: the depth-mean student for the multi-depth read, stock Reader v3 for the single-depth read."""

import json
import sys

import numpy as np
from scipy import ndimage as ni

from calibration.ink.agreed_truth import agreed
from ink.datasets.known_letters import (
    HOLDOUT_SCROLLS,
    NEAR_MM,
    READ_NOT_TAUGHT,
    cases_near_9um,
    letters_of,
    mark_of,
    reviewed,
)
from ink.datasets.own_mesh_labels import KINDS, ROUTE_FOLDERS, kept_windows
from ink.model.input_stack import face_layer_of
from ink.training import model_training
from ink.training.reader_v3_start_training import SEED, starting_weights, taught_from, taught_model_as_start
from storage.data_roots import TRAINED_WEIGHTS

TIERS = {
    "multi-depth": {"start": "reader-depth-mean-student-without-0841", "jitter": 4, "wandering": 0.5},
    "single-depth": {"start": None, "jitter": 1, "wandering": 0.0},
}
STEPS = 1500
LEARNING_RATE = 2e-5
ROUTE = "level"


def accepted():
    """Accepted marks of scrolls outside the holdout and of segments not kept to be read, by case."""
    found = {}
    for mark, verdict in reviewed().items():
        if (
            verdict["verdict"] == "accept"
            and mark.split("-")[0] not in HOLDOUT_SCROLLS
            and not mark.startswith(READ_NOT_TAUGHT)
        ):
            where, number = mark.rsplit("-letter-", 1)
            found.setdefault(where, set()).add(int(number))
    return found


def case_targets(case, letters, numbers):
    """A case's stroke cores of its accepted letters as ink, and its agreed blank near them as blank."""
    _, ink, blank, _ = agreed(case)
    height, width = min(ink.shape[0], letters.shape[0]), min(ink.shape[1], letters.shape[1])
    ink, blank, letters = ink[:height, :width], blank[:height, :width], letters[:height, :width]
    chosen = np.isin(letters, list(numbers))
    reach = round(NEAR_MM * 1000 / case["voxel_um"])
    near = ni.binary_dilation(chosen, iterations=reach)
    return ink & chosen, blank & near & ~ni.binary_dilation(letters > 0, iterations=1)


class MarkWindow:
    """One of our own windows taught only on the accepted letters it holds, read from its piece's face."""

    def __init__(self, path, row, labels, targets):
        self.sample, self.name, self.fixed = path.parent.name, path.stem, {}
        self.stack = np.load(path, mmap_mode="r")
        ink_t, blank_t = targets
        ys, xs, at_row, at_column = labels["published"]
        inside = (at_row < ink_t.shape[0]) & (at_column < ink_t.shape[1])
        ys, xs, at_row, at_column = ys[inside], xs[inside], at_row[inside], at_column[inside]
        self.ink = np.zeros(self.stack.shape[1:], dtype=bool)
        self.known = np.zeros(self.stack.shape[1:], dtype=bool)
        self.ink[ys, xs] = ink_t[at_row, at_column]
        self.known[ys, xs] = ink_t[at_row, at_column] | blank_t[at_row, at_column]
        self.start = int(row.get("face", face_layer_of(self.stack)))
        self.back_start = face_layer_of(self.stack[::-1])
        self.inked = np.argwhere(self.ink)[::101].copy()
        self.spots = np.argwhere(self.known)[::509].copy()


def mark_windows():
    """Every window of our own pieces that holds accepted letters, with their targets."""
    chosen, found = accepted(), []
    for kind in KINDS:
        for case in cases_near_9um(kind):
            numbers = chosen.get(mark_of(case, 0).rsplit("-letter-", 1)[0])
            if not numbers:
                continue
            letters, _ = letters_of(case)
            targets = case_targets(case, letters, numbers)
            for path, row, labels in kept_windows(case["sample"], ROUTE_FOLDERS[ROUTE], KINDS[kind]):
                if row["case"] == case["case_id"]:
                    window = MarkWindow(path, row, labels, targets)
                    if window.ink.any():
                        found.append(window)
    print(
        "mark windows",
        len(found),
        "ink pixels",
        sum(int(w.ink.sum()) for w in found),
        "known",
        sum(int(w.known.sum()) for w in found),
        flush=True,
    )
    return found


def loss_of(torch, logits, ink, known):
    """Cross entropy on the reviewed marks' ink and the blank beside them, nothing elsewhere."""
    cross = torch.nn.functional.binary_cross_entropy_with_logits(logits, ink, reduction="none")
    return (cross * known).sum() / known.sum().clamp(min=1.0)


def taught(tier, name=None, steps=STEPS):
    """One tier's reader taught lean on the reviewed marks of the training scrolls."""
    setting = TIERS[tier]
    windows = mark_windows()
    if not windows:
        raise SystemExit("no accepted marks yet: review them first (known_letters review)")
    start = (
        taught_model_as_start(TRAINED_WEIGHTS / f"{setting['start']}.pth") if setting["start"] else starting_weights()
    )
    model_training.cases = list
    model_training.own_cases = lambda left_out: windows
    model_training.published_of = lambda left_out: {}
    model_training.backgrounds_of = lambda target: []
    model_training.loss_of = loss_of
    model_training.LEARNING_RATE, model_training.STEM_LEARNS = LEARNING_RATE, 0
    model_training.PUBLISHED, model_training.SCRAMBLED, model_training.BACK_OF_SHEET = 0.0, 0.0, 0.0
    model_training.JITTER, model_training.WANDERING = setting["jitter"], setting["wandering"]
    name = name or f"reader-{tier}-reviewed-marks"
    print(taught_from(start, name, "none", None, SEED, int(steps), 1.0), flush=True)
    (TRAINED_WEIGHTS / f"{name}.json").write_text(
        json.dumps({"tier": tier, "steps": int(steps), "marks": sum(len(v) for v in accepted().values())}, indent=1)
    )


if __name__ == "__main__":
    taught(*sys.argv[1:])
