"""Prize scrolls ranked for where to look for writing, by a protocol fixed before the reads: excess sure ink over a
blank offset on pieces open on both faces, a first look and a replicate on a broader sample, each confirmed with
Benjamini-Hochberg at 0.10, and a prespecified row test, with PHerc0841 as the known-text control."""

import json
import sys

import numpy as np
from scipy import ndimage as ni

from calibration.survey.scroll_survey import (
    CELL,
    REPLICATED,
    RULER,
    SECOND_MODEL,
    SURVEYED,
    cells_of,
    letter_band,
    open_columns,
)
from calibration.survey.survey_rim_excess import (
    RECORD as RIM_RECORD,
)
from calibration.survey.survey_row_power import (
    RECORD as ROW_RECORD,
)
from ink.model.neural_engine import EngineReader
from ink.model.stock_model import SURE
from ink.reading.piece_reading import window
from ink.training.map_context_reader import LEFT_OUT, PUBLISHED, WINDOWS
from scans.catalog.scan_catalog import load, survey_volumes
from storage.data_roots import ROUTE_OUTPUTS

ELIGIBLE = {
    "PHerc0125": "20250821151825/band-0032-0044",
    "PHerc0175A": "20250521115057/band-0024-0036",
    "PHerc0175B": "20250521125822/band-0024-0036",
    "PHerc0191": "20250821151635/band-0032-0044",
    "PHerc0211": "20250821151803/rows-0022-0034",
    "PHerc0257": "20250821151750/band-0032-0044",
    "PHerc0268": "20251110183117/band-0024-0036",
    "PHerc0358": "20250821151737/band-0024-0036",
    "PHerc0483A": "20250521140913/band-0032-0044",
    "PHerc0483B": "20251124083638/band-0020-0032",
    "PHerc0490A": "20250521151210/band-0020-0032",
    "PHerc0490B": "20250521151215/band-0016-0028",
    "PHerc0800": "20250521135224/band-0040-0052",
    "PHerc0813": "20250821151723/band-0028-0040",
    "PHerc0826": "20250821151701/band-0024-0036",
    "PHerc0846A": "20250728152254/band-0016-0028",
    "PHerc0846B": "20250804142305/band-0028-0040",
    "PHerc1203": "20250820131727/band-0032-0044",
    "PHerc1218": "20250521120456/band-0040-0052",
    "PHerc1545": "20250821151648/band-0028-0040",
}
CONTROLS = {
    "PHerc0841": "20250821151531/rows-0050-0062",
    "PHerc0343": "20250521140437/band-0028-0040",
    "PHerc0306B": "20250521133212/band-0028-0040",
}
BLANK_SAMPLES = ("PHerc0009B", "PHerc0500P2")
CELL_MM = 0.075
INK_LEVEL = 0.5
BLANK_MM, TEXT_MM = 3.0, 0.5
FEWEST_CELLS, FEWEST_OPEN_CELLS = 1000, 300
DRAWS = 20000
BOTH_FACES = " both faces"
TARGET_OPEN_SHARE = 0.25
NULL_RECORD = SURVEYED / "ruler-null.json"
BOARD = SURVEYED / "leaderboard.json"


def ruler_cells(near, far, where, cell_mm):
    """The log band ratio and sure excess of two maps over a set of cells, the band taken inside those cells so nothing leaks in from text beside them."""
    forward, backward = (letter_band(cells_of(m), where, cell_mm)[where] for m in (near, far))
    sure = [cells_of((m >= SURE).astype(np.float32))[where].mean() for m in (near, far)]
    return 0.5 * float(np.log(max((forward**2).sum(), 1e-12) / max((backward**2).sum(), 1e-12))), float(
        sure[0] - sure[1]
    )


def blank_windows(sample):
    """Published windows of a scroll holding enough papyrus far from any ink, as (stack, published cells)."""
    for path in sorted(WINDOWS.glob(f"{sample}-*.npz")):
        kept = np.load(path)
        distance = ni.distance_transform_edt(kept["published"] < INK_LEVEL) * CELL_MM
        if ((kept["sheet"] > 0.5) & (distance >= BLANK_MM)).sum() >= FEWEST_CELLS:
            name = path.stem.removeprefix(f"{sample}-")
            tag = next(t for t in PUBLISHED if name.startswith(t))
            yield PUBLISHED[tag] / sample / f"{name.removeprefix(tag + '-')}.npy", kept["published"]


def window_rows(reader, stack_path, published):
    """A blank window read both ways."""
    stack = np.load(stack_path)
    near, far = (reader.read(window(cube)) for cube in (stack, stack[::-1]))
    height, width = published.shape[0] * CELL, published.shape[1] * CELL
    near, far, face = near[:height, :width], far[:height, :width], stack[:, :height, :width]
    seen = cells_of((face[face.shape[0] // 2] > 0).astype(np.float32)) > 0.9
    opened = (cells_of(open_columns(face).astype(np.float32)) > 0.9) & seen
    behind = (cells_of(open_columns(face[::-1]).astype(np.float32)) > 0.9) & seen
    distance = ni.distance_transform_edt(published < INK_LEVEL) * CELL_MM
    rows = []
    for name, where in (("blank", distance >= BLANK_MM), ("text", distance <= TEXT_MM), ("all", seen)):
        for suffix, columns in (("", opened), (BOTH_FACES, opened & behind)):
            if (where & columns).sum() >= FEWEST_OPEN_CELLS:
                ratio, excess = ruler_cells(near, far, where & columns, CELL_MM)
                rows.append(
                    {
                        "window": stack_path.stem,
                        "class": name + suffix,
                        "log_ratio": ratio,
                        "excess": excess,
                        "cells": int((where & columns).sum()),
                    }
                )
    return rows


def measured_null():
    """The ruler on papyrus far from any ink of the published maps, on text and on all open columns, each scroll read by the model that left it out."""
    rows = []
    for sample in BLANK_SAMPLES:
        reader = EngineReader(LEFT_OUT[sample])
        for stack_path, published in blank_windows(sample):
            rows += window_rows(reader, stack_path, published)
            print(rows[-1], flush=True)
    rng = np.random.default_rng(1)
    out = {"rows": rows}
    out |= measured_classes(rows, rng)
    NULL_RECORD.write_text(json.dumps(out, indent=1))
    return out


def measured_classes(rows, rng):
    """Each class of papyrus over its windows, weighted by open cells as pieces are by open pixels."""
    out = {}
    for name in ("blank", "text", "all", "blank" + BOTH_FACES, "text" + BOTH_FACES, "all" + BOTH_FACES):
        part = [r for r in rows if r["class"] == name]
        cells = np.array([r["cells"] for r in part], dtype=np.float64)
        if part:
            out[name] = {"windows": len(part)} | {
                key: summary_of(np.array([r[key] for r in part]), rng, cells) for key in ("log_ratio", "excess")
            }
    return out


def summary_of(values, rng, weights=None):
    """Weighted mean of pieces with its 95% interval, standard error and lower 90% bound, by bootstrap over pieces."""
    weights = np.ones(len(values)) if weights is None else weights
    draws = rng.integers(0, len(values), (DRAWS, len(values)))
    means = (values[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
    low, high, bound = np.quantile(means, [0.025, 0.975, 0.10])
    return {
        "mean": float((values * weights).sum() / weights.sum()),
        "low": float(low),
        "high": float(high),
        "bound": float(bound),
        "error": float(means.std()),
    }


def sign_flip(values, weights, zero, rng):
    """One-sided chance of a weighted mean this far above zero if each piece's sign were arbitrary."""
    centred = (values - zero) * weights
    signs = rng.choice((-1.0, 1.0), (DRAWS, len(values)))
    return float((1 + ((signs * centred).sum(axis=1) >= centred.sum()).sum()) / (DRAWS + 1))


def discoveries(chances, rate=0.10):
    """Benjamini-Hochberg over a dict of chances."""
    names = sorted(chances, key=chances.get)
    passed = [k for k, name in enumerate(names, 1) if chances[name] <= rate * k / len(names)]
    return set(names[: max(passed)]) if passed else set()


def shrunken(means, errors):
    """Empirical Bayes means and errors, the spread between scrolls by the method of moments."""
    means, errors = np.asarray(means), np.asarray(errors)
    weight = 1 / errors**2
    centre = (weight * means).sum() / weight.sum()
    excess = (weight * (means - centre) ** 2).sum() - (len(means) - 1)
    spread = max(excess / (weight.sum() - (weight**2).sum() / weight.sum()), 0.0)
    keep = spread / (spread + errors**2)
    return centre + keep * (means - centre), np.sqrt(keep) * errors


def pieces_in(folder):
    """Rows of a stage's pieces that had open columns, read by today's ruler."""
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("piece-*.json"))]
    return [r for r in rows if r.get("pixels", 0) > 0 and r.get("ruler") == RULER]


def statistics_of(rows):
    """Each piece's log band ratio, sure excess and open pixels."""
    ratio = np.array([0.5 * np.log(max(r["band_forward"], 1e-12) / max(r["band_reversed"], 1e-12)) for r in rows])
    return (
        ratio,
        np.array([(r["forward"] - r["reversed"]) / r["pixels"] for r in rows]),
        np.array([r["pixels"] for r in rows], dtype=np.float64),
    )


def open_share(folder):
    """Share of the windows' sheet looking onto a gap."""
    opened = sheet = 0.0
    for path in folder.glob("piece-*.npz"):
        kept = np.load(path)
        for k in range(2):
            if f"sheet{k}" in kept:
                seen = kept[f"sheet{k}"].astype(np.float32) > 0.5
                opened += float((seen & (kept[f"open{k}"].astype(np.float32) > 0.5)).sum())
                sheet += float(seen.sum())
    return opened / sheet if sheet else None


def scan_of(item, volumes):
    """A scroll's scan and the sheet its band holds."""
    sample, volume, band = item.split("/")
    row = volumes.get((sample, volume), {})
    out = {"volume": volume, "band": band, "voxel_um": row.get("voxel_um"), "energy_kev": row.get("energy_kev")}
    path = ROUTE_OUTPUTS / sample / volume / "pieces" / band / "record.json"
    if path.exists():
        record = json.loads(path.read_text())
        out |= {"pieces_4cm2": record["pieces_4cm2"], "area_cm2": round(record["area_cm2"], 1)}
    return out


def both_faces(rows, rng):
    """Forward over reversed on columns where both faces look onto a gap, by pieces weighted by those columns."""
    counts = np.array(
        [r["faces_open"]["both"] for r in rows if r.get("faces_open", {}).get("both", [0])[0] > 0], dtype=np.float64
    )
    return summary_of((counts[:, 1] - counts[:, 2]) / counts[:, 0], rng, counts[:, 0]) if len(counts) >= 2 else None


def paired_rows(rows, rng):
    """Row power forward against reversed on the same pieces."""
    pairs = np.array([(r["ratio"], r["reversed"]["ratio"]) for r in rows if r.get("reversed")])
    if len(pairs) < 3:
        return None
    gap = np.log(pairs[:, 0] / pairs[:, 1])
    return {
        "pieces": len(pairs),
        "forward": float(np.median(pairs[:, 0])),
        "reversed": float(np.median(pairs[:, 1])),
        "higher": int((gap > 0).sum()),
        "chance": sign_flip(gap, np.ones(len(gap)), 0.0, rng),
    }


def row_power(sample, rng):
    """Row power of a scroll's pieces in the two stages, and alone on the third sample, as fixed and with rims left out."""
    kept = (json.loads(ROW_RECORD.read_text()).get(sample) or {}) if ROW_RECORD.exists() else {}
    stages = [r for name in ("first", "replicate") for r in kept.get(name, {}).values()]
    third = list(kept.get("row_sample", {}).values())
    out = {}
    for name, rows in (("stages", stages), ("third_sample", third)):
        out[name] = paired_rows([r for r in rows if "ratio" in r], rng)
        out[f"{name}_no_rim"] = paired_rows([r["no_rim"] for r in rows if "no_rim" in r], rng)
    return out


def rim_out(sample, rng):
    """The replicate's excess with 1 mm of rim left out, by pieces weighted by what is left."""
    kept = (json.loads(RIM_RECORD.read_text()).get(sample) or {}) if RIM_RECORD.exists() else {}
    counts = np.array([r["1.0"] for r in kept.get("replicate", {}).values() if r["1.0"][0] > 0], dtype=np.float64)
    return summary_of((counts[:, 1] - counts[:, 2]) / counts[:, 0], rng, counts[:, 0]) if len(counts) >= 2 else None


def entry(sample, scan, zero, rng, volumes):
    """A scroll's rows by stage."""
    first, second = pieces_in(SURVEYED / sample), pieces_in(REPLICATED / sample)
    if not first:
        return None
    fair = REPLICATED if len(second) >= 2 else SURVEYED
    out = {
        "sample": sample,
        "open_share": open_share(fair / sample),
        "first_pieces": len(first),
        "replicate_pieces": len(second),
    } | scan_of(f"{sample}/{scan}", volumes)
    ratio, excess, pixels = statistics_of(first + second)
    out |= {
        "pooled": summary_of(excess, rng, pixels),
        "pooled_band_ratio": summary_of(ratio, rng, pixels),
        "chance": sign_flip(excess, pixels, zero, rng),
    }
    out["first"] = summary_of(statistics_of(first)[1], rng, statistics_of(first)[2])
    if len(second) >= 2:
        out["replicate"] = summary_of(statistics_of(second)[1], rng, statistics_of(second)[2])
        out["confirmed"] = out["replicate"]["bound"] > zero
    else:
        out["confirmed"] = None
    out["row_power"], out["both_faces"], out["rim_out"] = (
        row_power(sample, rng),
        both_faces(first + second, rng),
        rim_out(sample, rng),
    )
    model = pieces_in(SECOND_MODEL / sample)
    if len(model) >= 2:
        out["second_model"] = summary_of(statistics_of(model)[1], rng, statistics_of(model)[2])
    return out


def board():
    """The table of the protocol."""
    zero = measured_classes(json.loads(NULL_RECORD.read_text())["rows"], np.random.default_rng(1))["blank"]["excess"][
        "mean"
    ]
    rng = np.random.default_rng(2)
    volumes = {(r["sample"], r["volume"]): r for r in survey_volumes(load())}
    prize, controls = (
        [row for row in (entry(name, scan, zero, rng, volumes) for name, scan in group.items()) if row]
        for group in (ELIGIBLE, CONTROLS)
    )
    for r, mean, error in zip(
        prize, *shrunken([r["pooled"]["mean"] for r in prize], [r["pooled"]["error"] for r in prize])
    ):
        r["shrunken"], r["shrunken_error"] = float(mean), float(error)
    found = discoveries({r["sample"]: r["chance"] for r in prize})
    for r in prize:
        r["discovered"] = r["sample"] in found
        r["target"] = (
            bool(r["confirmed"]) and (r["open_share"] or 0) >= TARGET_OPEN_SHARE and r.get("pieces_4cm2", 0) >= 1
        )
    prize.sort(key=lambda r: (not r["confirmed"], -r["shrunken"]))
    out = {
        "zero": zero,
        "prize": prize,
        "controls": controls,
        "unread": sorted(set(ELIGIBLE) - {r["sample"] for r in prize}),
    }
    BOARD.write_text(json.dumps(out, indent=1))
    return out


def interval(row, key):
    return f"{row[key]['mean']:+.3f} [{row[key]['low']:+.3f}, {row[key]['high']:+.3f}]" if key in row else ""


def share_of(row):
    return f"{row['open_share']:.2f}" if row["open_share"] is not None else ""


def both_of(row):
    return f"{row['both_faces']['mean']:+.3f}" if row.get("both_faces") else ""


def rows_of(row, key):
    found = (row.get("row_power") or {}).get(key)
    return (
        f"{found['forward']:.2f} / {found['reversed']:.2f} ({found['higher']} of {found['pieces']}, {found['chance']:.3f})"
        if found
        else ""
    )


def table(out):
    """The board as Markdown."""
    lines = [
        "| Rank | Scroll | Pieces 1+2 | Open | First look | Replicate | Pooled, shrunken | Second model | Both faces open | Replicate, rim out | Rows, forward / reversed | Rows, rim out | Confirmed | Target |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for k, r in enumerate(out["prize"], 1):
        confirmed = {True: "yes", False: "no", None: "pending"}[r["confirmed"]]
        lines.append(
            f"| {k} | {r['sample']} | {r['first_pieces']}+{r['replicate_pieces']} | {share_of(r)} | {interval(r, 'first')} | {interval(r, 'replicate')} | {r['shrunken']:+.3f} ± {r['shrunken_error']:.3f} | {interval(r, 'second_model')} | {both_of(r)} | {interval(r, 'rim_out') if r.get('rim_out') else ''} | {rows_of(r, 'stages')} | {rows_of(r, 'stages_no_rim')} | {confirmed} | {'yes' if r['target'] else ''} |"
        )
    return "\n".join(lines)


def calibration(out):
    """The ruler on scrolls that are not candidates, as Markdown."""
    lines = [
        "| Control | Pieces 1+2 | Open | First look | Replicate | Both faces open | Replicate, rim out | Rows, forward / reversed | Rows, rim out |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['sample']} | {r['first_pieces']}+{r['replicate_pieces']} | {share_of(r)} | {interval(r, 'first')} | {interval(r, 'replicate')} | {both_of(r)} | {interval(r, 'rim_out') if r.get('rim_out') else ''} | {rows_of(r, 'stages')} | {rows_of(r, 'stages_no_rim')} |"
        for r in out["controls"]
    ]
    return "\n".join(lines)


def third_sample(out):
    """The row test alone on the third sample, as Markdown."""
    lines = ["| Scroll | Rows, forward / reversed | Rows, rim out |", "| --- | ---: | ---: |"]
    lines += [
        f"| {r['sample']} | {rows_of(r, 'third_sample')} | {rows_of(r, 'third_sample_no_rim')} |"
        for r in out["controls"] + out["prize"]
        if (r.get("row_power") or {}).get("third_sample")
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:] == ["null"]:
        result = measured_null()
        for name in ("blank", "text", "all"):
            print(
                name,
                {k: v if k == "windows" else {a: round(b, 3) for a, b in v.items()} for k, v in result[name].items()},
            )
    else:
        result = board()
        print(
            table(result),
            calibration(result),
            third_sample(result),
            f"Unread: {', '.join(result['unread'])}",
            sep="\n\n",
        )
