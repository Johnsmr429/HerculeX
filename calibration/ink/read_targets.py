"""What a reader's sure ink and sure blank on our own pieces are worth as training targets, against hand labels they never saw."""

import json
import sys
from pathlib import Path

import numpy as np

from calibration.ink.own_mesh_windows import face_read
from calibration.ink.stroke_widths import ROUTES, own
from ink.datasets.letter_spots import LEAST_INSIDE, LEAST_READ_INK
from ink.model.stock_model import SURE
from ink.model.turned_reader import named_reader
from ink.training.model_training import READ_BLANK, SURE_BLANK, SURE_INK
from storage.data_roots import calibration_output

OUT = calibration_output("read_targets")
BLANK = READ_BLANK / 255
KEYS = ("ink_share", "blank_share", "ink_right", "blank_right", "label_found")


def worth(ink, blank, label, known):
    """Counts of a target's ink and blank on known pixels, and of those the hand label agrees with."""
    return np.array(
        [
            known.sum(),
            (ink & known).sum(),
            (blank & known).sum(),
            (ink & label & known).sum(),
            (blank & ~label & known).sum(),
            (label & known).sum(),
        ],
        dtype=np.float64,
    )


def shares(counts):
    known, ink, blank, ink_right, blank_right, label = counts
    values = (ink / known, blank / known, ink_right / max(ink, 1), blank_right / max(blank, 1), ink_right / label)
    return {key: round(float(value), 4) for key, value in zip(KEYS, values)}


def run(sample, out, first_path, paths):
    """Each reader's targets, alone and where the scroll's first read agrees, over all windows and over those the letter-spot rule keeps."""
    _, first_reader = named_reader(first_path)
    firsts = {
        (route, name): face_read(first_reader, stack) for route in ROUTES for name, stack, *_ in own(sample, route)
    }
    del first_reader
    kept = {"sample": sample, "first": Path(first_path).stem}
    for path in paths:
        model, reader = named_reader(path)
        for route in ROUTES:
            totals, windows = {}, 0
            for name, stack, label, known, fine, _ in own(sample, route):
                read, first = face_read(reader, stack), firsts[route, name]
                on_sheet = np.asarray(stack[stack.shape[0] // 2]) > 0
                ink, first_ink = (read >= SURE) & on_sheet, (first >= SURE) & on_sheet
                letters = (
                    ink.sum() >= LEAST_READ_INK * on_sheet.sum() and (ink & first_ink).sum() >= LEAST_INSIDE * ink.sum()
                )
                targets = {
                    "fine map": (fine >= SURE_INK, (fine <= SURE_BLANK) & (fine > 0)),
                    "first read": (first >= SURE, first <= BLANK),
                    "reader": (read >= SURE, read <= BLANK),
                    "reader where both agree": ((read >= SURE) & (first >= SURE), (read <= BLANK) & (first <= BLANK)),
                    f"reader, windows {'kept' if letters else 'dropped'}": (read >= SURE, read <= BLANK),
                }
                for rule, (target_ink, target_blank) in targets.items():
                    totals[rule] = totals.get(rule, 0) + worth(target_ink, target_blank, label, known)
                windows += bool(letters)
            kept[f"{model} {route}"] = {rule: shares(counts) for rule, counts in totals.items()} | {
                "windows_kept": windows
            }
            print(model, route, json.dumps(kept[f"{model} {route}"]), flush=True)
        del reader
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
