"""Two maps of the same passage shown blind, in a random order, to a vision model asked which reads as clearer writing and which letters it can name.

The model is any command line that takes the prompt on standard input and the image's path as its last argument,
named by HERCULEX_JUDGE (default: the Codex CLI, read-only).
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

from storage.data_roots import SHEETS, calibration_output

OUT = calibration_output("readability_judge")
SIDE = 1024
JUDGE = os.environ.get("HERCULEX_JUDGE", "codex exec --skip-git-repo-check -s read-only -i").split()
PROMPT = (
    "Two ink-detection maps, A (left) and B (right), of the same patch of a carbonised Herculaneum papyrus "
    "read from a CT scan; each is about 14 mm across, ink is dark, black at the borders is off the papyrus. "
    "Ancient Greek letters are 2 to 4 mm tall. Which map shows writing more clearly: answer exactly one of "
    "A, B or SAME on the first line. On the second line list any Greek letters you can name with confidence in "
    "the clearer map, or NONE. Do not guess."
)


def pair_image(left, right, path):
    """The two panels side by side at one scale, a grey bar between."""
    height = max(left.shape[0], right.shape[0])
    pad = [np.pad(p, ((0, height - p.shape[0]), (0, 0)), constant_values=0) for p in (left, right)]
    both = np.concatenate([pad[0], np.full((height, 16), 128, np.uint8), pad[1]], 1)
    scale = SIDE / max(both.shape)
    Image.fromarray(both).resize((round(both.shape[1] * scale), round(both.shape[0] * scale)), Image.LANCZOS).save(path)


def judged(image):
    """The vision model's verdict on one pair."""
    result = subprocess.run(
        [*JUDGE, str(image)],
        input=PROMPT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return lines[-2:] if len(lines) >= 2 else lines


def run(page_name, first_title, second_title, seed=7):
    """Every passage of a page judged once, the panel order drawn at random and recorded."""
    page = SHEETS / page_name
    out = OUT / page_name / second_title.replace(" ", "-")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for folder in sorted(p for p in page.iterdir() if p.is_dir()):
        files = {f.stem.split("-", 1)[1].replace("-", " "): f for f in folder.glob("*.png")}
        if first_title not in files or second_title not in files:
            continue
        panels = [np.asarray(Image.open(files[t])) for t in (first_title, second_title)]
        swapped = rng.random() < 0.5
        image = out / f"{folder.name}.png"
        pair_image(*(panels[::-1] if swapped else panels), image)
        verdict = judged(image)
        first = verdict[0].upper() if verdict else ""
        left, right = (second_title, first_title) if swapped else (first_title, second_title)
        winner = {"S": "same", "A": left, "B": right}.get(first[:1], "unclear")
        rows.append({"passage": folder.name, "swapped": swapped, "verdict": verdict, "clearer": winner})
        print(json.dumps(rows[-1]), flush=True)
    (out / "verdicts.json").write_text(json.dumps(rows, indent=1))
    tally = {k: sum(r["clearer"] == k for r in rows) for k in (first_title, second_title, "same", "unclear")}
    print(tally, flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:4])
