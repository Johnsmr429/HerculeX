"""Whether the second reader on the ink map adds rows of its own."""

import itertools
import json
import sys

import numpy as np
from scipy import ndimage as ni

from calibration.text import line_spectrum
from calibration.text.line_spectrum import PERIODS, spectrum
from ink.model.stock_model import SURE
from ink.training import map_context_reader as map_context
from storage.data_roots import calibration_output

CELL_MM = 0.075
INK_LEVEL = 0.5
BLANK_MM = 4.0
TEXT_MM = 1.0
SMALLEST_BLANK = 3000
LETTER_BAND_MM = (0.4, 3.0)
STRIP_MM, BAND_MM = 3.0, (2.0, 4.0)
ROW_PERIODS = (PERIODS >= 4.0) & (PERIODS <= 7.0)
ROW_WINDOW_MM = (16.0, 10.0)
LONGEST_STROKE_CELLS, TOP_SHARE = 13, 0.05
CENSUS = calibration_output("height_census")
OUT = map_context.FOLDER / "invention"


def letter_band(image, seen):
    weight = seen.astype(np.float32)
    values = np.where(seen, image, 0).astype(np.float32)
    fine, coarse = (
        ni.gaussian_filter(values, mm / CELL_MM) / np.maximum(ni.gaussian_filter(weight, mm / CELL_MM), 1e-3)
        for mm in LETTER_BAND_MM
    )
    return fine - coarse


def band_energy(image, seen):
    return float(np.sqrt((letter_band(image, seen)[seen] ** 2).mean()))


def strokes_per_cm2(image, seen):
    """Long thin bright marks per cm2 among the top share of a map's letter band."""
    band = letter_band(image, seen)
    marks, _count = ni.label((band > np.quantile(band[seen], 1 - TOP_SHARE)) & seen)
    found = 0
    for box in ni.find_objects(marks):
        height, width = box[0].stop - box[0].start, box[1].stop - box[1].start
        found += max(height, width) >= LONGEST_STROKE_CELLS and max(height, width) >= 2.5 * min(height, width)
    return found / max(seen.sum() * CELL_MM**2 / 100, 1e-6)


def blank_papyrus(net, sample):
    """Papyrus the published map calls blank, read alone so no ink is in reach, against text alone."""
    rows = []
    for number, (first, published, sheet) in enumerate(map_context.windows_of([sample])):
        distance = ni.distance_transform_edt(published < INK_LEVEL) * CELL_MM
        seen = sheet > 0.5
        blank, text = seen & (distance >= BLANK_MM), seen & (distance <= TEXT_MM)
        if blank.sum() < SMALLEST_BLANK or text.sum() < SMALLEST_BLANK:
            print(
                "window", number, "blank cells", int(blank.sum()), "text cells", int(text.sum()), "too few", flush=True
            )
            continue
        row = {"window": number, "blank_cells": int(blank.sum()), "text_cells": int(text.sum())}
        for kind, where in (("blank", blank), ("text", text)):
            second = map_context.read(net, first, where.astype(np.float32))
            row[kind] = {
                "first_energy": band_energy(first, where),
                "second_energy": band_energy(second, where),
                "first_strokes": strokes_per_cm2(first, where),
                "second_strokes": strokes_per_cm2(second, where),
                "first_sure": float((first[where] >= SURE).mean()),
                "second_sure": float((second[where] >= 0.5).mean()),
            }
        rows.append(row)
        print(
            number, json.dumps({k: {a: round(b, 3) for a, b in row[k].items()} for k in ("blank", "text")}), flush=True
        )
    return rows


def rows_destroyed(image, seen, rng):
    """Strips along the fibres' cross direction cut into bands of random height and the bands shuffled."""
    out, cover = image.copy(), seen.copy()
    strip = int(STRIP_MM / CELL_MM)
    for x in range(0, image.shape[1], strip):
        edges = [0]
        while edges[-1] < image.shape[0]:
            edges.append(edges[-1] + int(rng.uniform(*BAND_MM) / CELL_MM))
        edges[-1] = image.shape[0]
        bands = [(a, b) for a, b in itertools.pairwise(edges)]
        rng.shuffle(bands)
        y = 0
        for a, b in bands:
            out[y : y + b - a, x : x + strip] = image[a:b, x : x + strip]
            cover[y : y + b - a, x : x + strip] = seen[a:b, x : x + strip]
            y += b - a
    return out, cover


def row_power(image, cover):
    """Mean line spectrum across and along the rows, by period, or None when no window fits."""
    found = spectrum(image, cover)
    return (
        (np.mean(found["across"], axis=0), np.mean(found["along"], axis=0))
        if found["across"] and found["along"]
        else None
    )


def gain(before, after):
    """Output power over input power by period, across the rows and, as the control, along them."""
    across, along = after[0] / before[0], after[1] / before[1]
    return {
        "across_rows": float(across[ROW_PERIODS].mean() / across[~ROW_PERIODS].mean()),
        "along_rows": float(along[ROW_PERIODS].mean() / along[~ROW_PERIODS].mean()),
        "across_by_period": [round(float(v), 3) for v in across],
    }


def census_maps(limit):
    """First-reader maps of PHerc0211 pieces of the census on cells, with their sheet."""
    for path in sorted(CENSUS.glob("rows-*.npz"))[:limit]:
        kept = np.load(path)
        face = json.loads(path.with_suffix(".json").read_text())["side"]["face"]
        image = kept["far" if face == "far" else "near"].astype(np.float32) / 255
        yield map_context.cells_of(image), map_context.cells_of(kept["sheet"].astype(np.float32))


def known_maps(sample):
    """First-reader maps of published windows of a left-out scroll, with their sheet."""
    for first, _, sheet in map_context.windows_of([sample]):
        yield first, sheet


def row_tests(net, maps):
    """Row gain of real maps and of the same maps with their rows destroyed."""
    line_spectrum.WINDOW_MM = ROW_WINDOW_MM
    rng = np.random.default_rng(11)
    totals = {"real": [], "destroyed": []}
    for first, sheet in maps:
        for kind in totals:
            image, cover = (first, sheet) if kind == "real" else rows_destroyed(first, sheet, rng)
            before = row_power(image, cover)
            if before is None:
                continue
            after = row_power(map_context.read(net, image, (cover > 0.5).astype(np.float32)), cover)
            if after is not None:
                totals[kind].append((before, after))
    out = {}
    for kind, pairs in totals.items():
        if pairs:
            before = tuple(np.mean([p[0][i] for p in pairs], axis=0) for i in (0, 1))
            after = tuple(np.mean([p[1][i] for p in pairs], axis=0) for i in (0, 1))
            out[kind] = dict(gain(before, after), maps=len(pairs))
            print(kind, {k: v for k, v in out[kind].items() if k != "across_by_period"}, flush=True)
    return out


def run(left_out, levels, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    net = map_context.loaded(left_out, levels)
    print("known text", flush=True)
    known = row_tests(net, known_maps(left_out))
    print("PHerc0211", flush=True)
    unread = row_tests(net, census_maps(limit))
    result = {
        "left_out": left_out,
        "levels": levels,
        "blank_papyrus": blank_papyrus(net, left_out),
        "rows_known": known,
        "rows_unread": unread,
        "periods": [float(p) for p in PERIODS],
    }
    (OUT / f"{left_out}-levels-{levels}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40)
