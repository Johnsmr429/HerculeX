"""Blind pairs for the eye: one square read from each face, in a sealed order."""

import json
import sys

import numpy as np
from PIL import Image
from scipy.stats import binomtest

from calibration.ink.ink_yield import rank_auc
from calibration.ink.model_agreement import LEFT_OUT
from calibration.ink.own_mesh_windows import WINDOWS
from calibration.survey.scroll_survey import (
    CELL,
    cells_of,
    letter_band,
    open_columns,
)
from calibration.text.spot_tests import piece_stack
from ink.model.stock_model import SURE
from ink.model.tile_reader import TileReader
from ink.reading.piece_reading import WEIGHTS, window
from sampling import column_sampling as sampler
from storage.data_roots import BLIND_PAIRS, INK_RECORDS, ROUTE_OUTPUTS, TRAINED_WEIGHTS, stage_folder

SAMPLE, VOLUME = "PHerc0211", "20250821151803"
UM = 9.362
FIELD = 1440
SMALLEST_AREA_MM2 = 400
MOST_SCRAMBLED = 0.01
TAIL, RANDOM = 8, 8
SEED = 20261107
NAMED = ("rows-0022-0034", 74)
KNOWN = {
    "PHerc0841": ("22814-9.366um-1.2m-113keV-0002",),
    "PHerc0500P2": (
        "81859-9.362um-1.2m-113keV-0005",
        "81859-9.362um-1.2m-113keV-0006",
        "81859-9.362um-1.2m-113keV-0026",
    ),
}
OUT = BLIND_PAIRS
KEY = OUT / "key.json"
ANSWERS = {"L": "a", "R": "b", "B": "both", "N": "neither"}


def looked_pieces():
    """Pieces of the sheet with what the full read saw on each face."""
    base = ROUTE_OUTPUTS / SAMPLE / VOLUME
    faces = stage_folder(base, "faces") / "level-0"
    rows = []
    for p in json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]:
        record = faces / p["unit"] / f"piece-{p['piece']:04d}-taught-side.json"
        if record.exists():
            looks = json.loads(record.read_text())
            forward, back = ("far", "near") if looks["written"] == "far" else ("near", "far")
            rows.append(
                p
                | {
                    "forward_face": forward,
                    "forward": looks[forward],
                    "reversed": looks[back],
                    "scrambled": looks.get("scrambled", 0.0),
                    "fibres": looks.get("fibres", {}).get("face"),
                }
            )
    return rows


def chosen(rows):
    """The groups of the protocol, each row marked with its group."""
    side_mm = FIELD * UM / 1000
    pool = [
        r
        for r in rows
        if r["area"] >= SMALLEST_AREA_MM2
        and min(r["width_mm"], r["height_mm"]) >= side_mm
        and r["forward"] + r["reversed"] > 0
        and r["scrambled"] <= MOST_SCRAMBLED
    ]
    pool.sort(key=lambda r: (r["forward"] - r["reversed"], r["unit"], r["piece"]))
    low, middle, high = pool[:TAIL], pool[TAIL:-TAIL], pool[-TAIL:]
    drawn = [middle[k] for k in sorted(np.random.default_rng(SEED).choice(len(middle), RANDOM, replace=False))]
    named = [r for r in rows if (r["unit"], r["piece"]) == NAMED and r not in low + drawn + high]
    return [
        r | {"group": group}
        for group, part in (("forward tail", high), ("reversed tail", low), ("random", drawn), ("named", named))
        for r in part
    ]


def fullest_square(sheet, side=FIELD, step=64):
    """Corner of the square most covered by a mask."""
    h, w = sheet.shape
    mass = np.pad(sheet.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    spots = [(y, x) for y in range(0, max(h - side, 0) + 1, step) for x in range(0, max(w - side, 0) + 1, step)]
    return max(
        spots,
        key=lambda s: (
            mass[s[0] + side, s[1] + side] - mass[s[0], s[1] + side] - mass[s[0] + side, s[1]] + mass[s[0], s[1]]
        ),
    )


def faces_read(stack, reader):
    """Near and far of a stack, read alike and whole."""
    return reader.read(window(stack)), reader.read(window(stack[::-1]))


def measures(forward, reversed_read, forward_stack, voxel_mm):
    """Sure share forward and reversed on open columns and their letter-band ratio, as the ruler takes them."""
    seen = forward_stack[forward_stack.shape[0] // 2] > 0
    gap = seen & open_columns(forward_stack)
    if not gap.any():
        return {"open": 0.0, "forward": 0.0, "reversed": 0.0, "band_ratio": None}
    whole, opened = cells_of(seen.astype(np.float32)) > 0.9, cells_of(gap.astype(np.float32)) > 0.9
    energy = [
        float((letter_band(cells_of(read), whole, voxel_mm * CELL)[opened] ** 2).sum())
        for read in (forward, reversed_read)
    ]
    return {
        "open": float(gap.sum() / max(seen.sum(), 1)),
        "forward": float((forward[gap] >= SURE).mean()),
        "reversed": float((reversed_read[gap] >= SURE).mean()),
        "band_ratio": float(np.sqrt(energy[0] / energy[1])) if energy[1] > 0 else None,
    }


def views(near, far, sheet, corner, flip_rows=False, flip_cols=False):
    """Each face's square as it looks from that face, on the map scale of a piece's read."""
    y, x = corner
    shown = []
    for read in (near, far):
        part = np.where(
            sheet[y : y + FIELD, x : x + FIELD], np.rint(read[y : y + FIELD, x : x + FIELD] * 254) + 1, 0
        ).astype(np.uint8)
        part = part[::-1] if flip_rows else part
        shown.append(part[:, ::-1] if flip_cols else part)
    return {"near": shown[0], "far": shown[1][:, ::-1]}


def scroll_pair(row, reader):
    stack = np.asarray(piece_stack(SAMPLE, VOLUME, row["unit"], row["piece"], row["turn"]))
    near, far = faces_read(stack, reader)
    sheet = stack[stack.shape[0] // 2] > 0
    corner = fullest_square(sheet)
    y, x = corner
    forward, back = (far, near) if row["forward_face"] == "far" else (near, far)
    crop = (slice(None), slice(y, y + FIELD), slice(x, x + FIELD))
    forward_stack = stack[::-1] if row["forward_face"] == "far" else stack
    numbers = measures(forward[crop[1:]], back[crop[1:]], forward_stack[crop], UM / 1000)
    entry = {
        "kind": row["group"],
        "scan": SAMPLE,
        "unit": row["unit"],
        "piece": row["piece"],
        "x_mm": row["x_mm"],
        "y_mm": row["y_mm"],
        "corner": [int(y), int(x)],
        "forward_face": row["forward_face"],
        "fibres": row["fibres"],
        "looks": [row["forward"], row["reversed"]],
        **numbers,
    }
    return views(near, far, sheet, corner, row["flip_rows"], row["flip_cols"]), entry


def known_pair(sample, path, reader):
    known = np.load(path)
    stack, ink = known["stack"], known["ink"]
    rows, columns = known["rows"], known["columns"]
    near, far = faces_read(stack, reader)
    labelled = np.zeros(stack.shape[1:], dtype=bool)
    labelled[rows[ink], columns[ink]] = True
    y, x = fullest_square(labelled)
    scores = {
        "near": rank_auc(near[rows, columns][ink], near[rows, columns][~ink]),
        "far": rank_auc(far[rows, columns][ink], far[rows, columns][~ink]),
    }
    face = max(scores, key=scores.get)
    forward, back = (far, near) if face == "far" else (near, far)
    forward_stack = stack[::-1] if face == "far" else stack
    numbers = measures(
        forward[y : y + FIELD, x : x + FIELD],
        back[y : y + FIELD, x : x + FIELD],
        forward_stack[:, y : y + FIELD, x : x + FIELD],
        UM / 1000,
    )
    entry = {
        "kind": "known text",
        "scan": sample,
        "window": path.stem,
        "corner": [int(y), int(x)],
        "forward_face": face,
        "auc": {k: float(v) for k, v in scores.items()},
        **numbers,
    }
    return views(near, far, stack[stack.shape[0] // 2] > 0, (y, x)), entry


def sealed(made):
    """The pairs shuffled and their sides drawn."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)
    pairs, key = [], []
    for number, k in enumerate(rng.permutation(len(made)), 1):
        faces, entry = made[k]
        left = "near" if rng.random() < 0.5 else "far"
        sides = {"a": left, "b": "far" if left == "near" else "near"}
        names = {side: f"{number:02d}-{side}.png" for side in sides}
        for side, face in sides.items():
            Image.fromarray(faces[face]).save(OUT / names[side], compress_level=3)
        pairs.append({"number": number, **names})
        key.append({"number": number, "sides": sides, **entry})
    (OUT / "pairs.json").write_text(json.dumps(pairs))
    KEY.write_text(json.dumps(key, indent=1))


def fetch():
    """Each chosen piece's stack, kept."""
    rows = chosen(looked_pieces())
    for k, row in enumerate(rows, 1):
        piece_stack(SAMPLE, VOLUME, row["unit"], row["piece"], row["turn"])
        print(k, "of", len(rows), row["unit"], row["piece"], flush=True)


def read():
    """Every pair read and sealed."""
    reader = TileReader(WEIGHTS)
    made = [scroll_pair(row, reader) for row in chosen(looked_pieces())]
    print("scroll pairs", len(made), flush=True)
    for sample, stems in KNOWN.items():
        reader = TileReader(TRAINED_WEIGHTS / LEFT_OUT[sample][0])
        for path in (p for p in sorted((WINDOWS / sampler.SAMPLING / sample).glob("*.npz")) if p.stem.endswith(stems)):
            made.append(known_pair(sample, path, reader))
    print("all pairs", len(made), flush=True)
    sealed(made)


def scored(answers):
    """A reader's answers, like '1L 2R 3N 4B', against the key."""
    said = {int(item[:-1]): ANSWERS[item[-1].upper()] for item in answers.split()}
    key = json.loads(KEY.read_text())
    for pair in key:
        pick = said.get(pair["number"])
        pair["picked"] = pair["sides"].get(pick) if pick in ("a", "b") else pick
    text = [p for p in key if p["kind"] == "known text"]
    scroll = [p for p in key if p["kind"] != "known text"]
    clear = [p for p in scroll if p["picked"] in ("near", "far") and p.get("fibres") in ("near", "far")]
    agree = sum(p["picked"] == p["fibres"] for p in clear)
    return {
        "known_text": {"right": sum(p["picked"] == p["forward_face"] for p in text), "of": len(text)},
        "eye_passes": sum(p["picked"] == p["forward_face"] for p in text) >= 3,
        "scroll_picks": {
            group: {
                "forward": sum(p["picked"] == p["forward_face"] for p in part),
                "reversed": sum(p["picked"] in ("near", "far") and p["picked"] != p["forward_face"] for p in part),
                "both": sum(p["picked"] == "both" for p in part),
                "neither": sum(p["picked"] == "neither" for p in part),
            }
            for group in sorted({p["kind"] for p in scroll})
            for part in [[p for p in scroll if p["kind"] == group]]
        },
        "fibres": {
            "agree": agree,
            "of": len(clear),
            "p": float(binomtest(agree, len(clear), 0.5, alternative="greater").pvalue) if clear else None,
        },
        "ranked": sorted(key, key=lambda p: -(p["forward"] - p["reversed"])),
    }


def run(action, *answers):
    if action == "score":
        (INK_RECORDS / "blind-pairs.json").write_text(json.dumps(scored(" ".join(answers)), indent=1))
    else:
        {"fetch": fetch, "read": read}[action]()


if __name__ == "__main__":
    run(*sys.argv[1:])
