"""Every ink training dataset checked against its schema, with a manifest of what each holds."""

import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np

from ink.datasets.depth_mean_targets import OWN_PIECES
from ink.datasets.letter_spots import SHOWN
from ink.datasets.own_reads import READS
from storage.data_roots import (
    AGREED_TARGETS,
    DEPTH_MEAN_TARGETS,
    HAND_MARKS,
    INFRARED_FRAGMENTS,
    INK_TRAINING_DATA,
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    LETTER_SPOTS,
    OWN_MESH_FINE_LETTERS,
    OWN_MESH_LABELS,
    OWN_MESH_SMOOTHING,
    OWN_MESH_THIN_MARKS,
    OWN_SURFACE_HOLDOUT,
    PUBLISHED_PAIRS,
    PUBLISHED_PAIRS_WRITING,
    TARGET_STACKS,
)
from website.site_server import standing

CASE_KEYS = ("case_id", "sample", "voxel_um", "box", "truth", "partition")
PARTITIONS = ("development", "holdout")
LAYERS = ("bricks.bin", "meta.json", "offsets.npy")
FRAGMENT_IMAGES = ("ir.png", "mask.png", "inklabels.png")
ROUTES = ("grown_route", "grown_route_fresh_nodes", "level_route")
WINDOW = re.compile(r"^\d{14}-\d+-\d+$")
PIECE = re.compile(r"^piece-\d{4}\.npy$")


def readable(path):
    """An array whose header and size agree."""
    try:
        np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return True


def labelled_cases(problems):
    """Folders holding a case record, its layers, its ink and where it is supervised."""
    found = Counter()
    for folder in sorted(p for p in LABELLED_CASES.iterdir() if p.is_dir()):
        missing = [n for n in ("case.json", "ink.npy", "supervised.npy") if not (folder / n).exists()]
        missing += [f"layers/{n}" for n in LAYERS if not (folder / "layers" / n).exists()]
        if missing:
            problems.append(f"labelled_cases/{folder.name}: missing {', '.join(missing)}")
            continue
        case = json.loads((folder / "case.json").read_text())
        if case.get("case_id") != folder.name or any(k not in case for k in CASE_KEYS):
            problems.append(f"labelled_cases/{folder.name}: case.json lacks {CASE_KEYS} or names another case")
        elif case["partition"] not in PARTITIONS:
            problems.append(f"labelled_cases/{folder.name}: partition is not one of {PARTITIONS}")
        if not readable(folder / "ink.npy") or not readable(folder / "supervised.npy"):
            problems.append(f"labelled_cases/{folder.name}: unreadable labels")
        found[case.get("sample", "?")] += 1
    return found


def labelled_case_stacks(problems):
    """One stack per labelled case, named by it."""
    found = Counter()
    for path in sorted(LABELLED_CASE_STACKS.glob("*.npy")):
        case = LABELLED_CASES / path.stem / "case.json"
        if not case.exists():
            problems.append(f"labelled_case_stacks/{path.name}: no such labelled case")
            continue
        if not readable(path):
            problems.append(f"labelled_case_stacks/{path.name}: unreadable")
        found[json.loads(case.read_text())["sample"]] += 1
    return found


def infrared_fragments(problems):
    """An infrared photograph, its mask and its ink labels per fragment."""
    found = Counter()
    for folder in sorted(p for p in INFRARED_FRAGMENTS.iterdir() if p.is_dir()):
        missing = [n for n in FRAGMENT_IMAGES if not (folder / n).exists()]
        if missing:
            problems.append(f"infrared_fragments/{folder.name}: missing {', '.join(missing)}")
        found[folder.name] += 1
    return found


def published_windows(root, problems):
    """Windows named SEGMENT-Y-X: a surface stack and the published map of the same place."""
    found = Counter()
    for folder in sorted(root.glob("PHerc*")):
        maps = {p.stem for p in folder.glob("*.png") if WINDOW.match(p.stem)}
        stacks = {p.stem for p in folder.glob("*.npy") if not p.name.endswith(".part.npy")}
        if stacks - maps:
            problems.append(f"{root.name}/{folder.name}: {len(stacks - maps)} stacks without their map")
        if maps - stacks:
            problems.append(f"{root.name}/{folder.name}: {len(maps - stacks)} maps without their stack")
        found[folder.name] += len(maps & stacks)
    return found


def published_pairs(problems):
    return published_windows(PUBLISHED_PAIRS, problems)


def published_pairs_writing(problems):
    return published_windows(PUBLISHED_PAIRS_WRITING, problems)


def own_mesh_labels(problems):
    """Per route and scroll: a stack, its labels and its record, and the census each route was built from."""
    found = Counter()
    for folder in sorted(p for p in OWN_MESH_LABELS.iterdir() if p.is_dir()):
        if folder.name not in ROUTES:
            problems.append(f"own_mesh_labels/{folder.name}: not one of {ROUTES}")
            continue
        for path in sorted(folder.glob("PHerc*/*.npy")):
            for kept in (path.with_name(f"{path.stem}-labels.npz"), path.with_suffix(".json")):
                if not kept.exists():
                    problems.append(f"own_mesh_labels/{folder.name}/{path.parent.name}/{kept.name}: missing")
            found[f"{folder.name}/{path.parent.name}"] += 1
        for read in sorted(folder.glob(f"PHerc*/{READS}/*.npy")):
            if not (read.parent.parent / read.name).exists():
                problems.append(
                    f"own_mesh_labels/{folder.name}/{read.parent.parent.name}/{READS}/{read.name}: no stack"
                )
    return found


def letter_spots(problems):
    """Per route and scroll: each kept window's stack, the second reader's read, the first read and the record."""
    found = Counter()
    for path in sorted(LETTER_SPOTS.glob("*/PHerc*/*.npy")):
        place = path.parent.relative_to(LETTER_SPOTS)
        for kept in (path.parent / READS / path.name, path.parent / SHOWN / path.name, path.with_suffix(".json")):
            if not kept.exists():
                problems.append(f"letter_spots/{place}/{kept.relative_to(path.parent)}: missing")
        found[str(place)] += 1
    return found


def eye_letters(problems):
    """Letters named by eye: one ink, label and supervised window per row of letters.json, classes of 24."""
    root = INK_TRAINING_DATA / "eye_letters"
    if not (root / "letters.json").exists():
        return Counter()
    rows = json.loads((root / "letters.json").read_text())
    arrays = np.load(root / "letters.npz")
    for key in ("ink", "own", "any", "known"):
        if key not in arrays or len(arrays[key]) != len(rows):
            problems.append(f"eye_letters/letters.npz: {key} missing or not one per letter")
    for r in rows:
        if not 0 <= r["class"] < 24 or r["partition"] not in PARTITIONS or not 0 <= (r["auc"] or 0) <= 1:
            problems.append(f"eye_letters: {r['case']} body {r['body']} out of schema")
    return Counter(r["source"] for r in rows)


def source_of(kind, name):
    """The stack a kept target was made on."""
    if kind == "labelled_cases":
        return LABELLED_CASE_STACKS / f"{name}.npy"
    if kind == "own_pieces":
        return OWN_PIECES / f"{name}.npy"
    if kind.startswith("published_pairs/"):
        return PUBLISHED_PAIRS / kind.split("/", 1)[1] / f"{name}.npy"
    return OWN_MESH_LABELS / kind.split("/", 1)[1] / f"{name}.npy"


def targets(root, label, same_size, problems):
    """Per kind: each kept target a byte a pixel on an existing stack, no larger than it or, for maps of whole windows, as large."""
    found = Counter()
    for path in sorted(root.glob("**/*.npy")):
        if path.name.endswith(".part.npy"):
            continue
        kind = str(path.parent.relative_to(root))
        source = source_of(kind, path.stem)
        target = np.load(path, mmap_mode="r")
        if not source.exists():
            problems.append(f"{label}/{kind}/{path.name}: no stack")
        else:
            shape = np.load(source, mmap_mode="r").shape[1:]
            if target.dtype != np.uint8 or (
                target.shape != shape if same_size else any(a > b for a, b in zip(target.shape, shape))
            ):
                problems.append(f"{label}/{kind}/{path.name}: {target.dtype} {target.shape} on a stack of {shape}")
        found[kind] += 1
    return found


def agreed_targets(problems):
    return targets(AGREED_TARGETS, "agreed_targets", False, problems)


def depth_mean_targets(problems):
    return targets(DEPTH_MEAN_TARGETS, "depth_mean_targets", True, problems)


def target_stacks(problems):
    """Stacks of the target's pieces by scroll and unit."""
    found = Counter()
    for path in sorted(TARGET_STACKS.glob("*/*/*")):
        if not PIECE.match(path.name):
            problems.append(f"target_stacks/{path.relative_to(TARGET_STACKS)}: not piece-NNNN.npy")
        found[path.parts[-3]] += 1
    return found


def hand_marks(problems):
    """Marks still standing by kind, each drawn mark with the layout it was drawn on."""
    found = {}
    for log in sorted(HAND_MARKS.glob("**/marks.jsonl")):
        place = log.parent.relative_to(HAND_MARKS)
        layouts = {p.stem: p for p in (log.parent / "layouts").glob("*.json")}
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        referenced = {e["layout"] for e in entries if "layout" in e}
        orphans = referenced - layouts.keys()
        if orphans:
            problems.append(f"hand_marks/{place}: marks on layouts not kept: {sorted(orphans)}")
        for digest in sorted(referenced & layouts.keys()):
            if hashlib.sha256(layouts[digest].read_bytes()).hexdigest()[:12] != digest:
                problems.append(f"hand_marks/{place}: layout digest mismatch: {digest}")
        kinds = Counter(mark["kind"] for mark in standing(log))
        found[str(place)] = dict(sorted(kinds.items())) | {"drawn": sum("points" in e for e in entries)}
    return found


DATASETS = {
    "labelled_cases": labelled_cases,
    "labelled_case_stacks": labelled_case_stacks,
    "infrared_fragments": infrared_fragments,
    "published_pairs": published_pairs,
    "published_pairs_writing": published_pairs_writing,
    "own_mesh_labels": own_mesh_labels,
    "letter_spots": letter_spots,
    "eye_letters": eye_letters,
    "agreed_targets": agreed_targets,
    "depth_mean_targets": depth_mean_targets,
    "target_stacks": target_stacks,
    "hand_marks": hand_marks,
}

STUDY_DATASETS = {
    "own_mesh_fine_letters": OWN_MESH_FINE_LETTERS,
    "own_mesh_smoothing": OWN_MESH_SMOOTHING,
    "own_mesh_thin_marks": OWN_MESH_THIN_MARKS,
    "own_surface_holdout": OWN_SURFACE_HOLDOUT,
}


def total(counts):
    """Cases, windows or stacks of a dataset; of hand marks, those still standing."""
    return sum(
        sum(n for kind, n in count.items() if kind != "drawn") if isinstance(count, dict) else count
        for count in counts.values()
    )


def check():
    """The manifest written beside the datasets; exit 1 when any breaks its schema."""
    problems = []
    manifest = {name: dict(sorted(read(problems).items())) for name, read in DATASETS.items()}
    manifest["studies"] = {
        name: {"folder": root.name, "files": sum(p.is_file() for p in root.rglob("*"))}
        for name, root in STUDY_DATASETS.items()
        if root.is_dir()
    }
    manifest["problems"] = problems
    pending = INK_TRAINING_DATA / "manifest.json.part"
    pending.write_text(json.dumps(manifest, indent=1) + "\n")
    pending.replace(INK_TRAINING_DATA / "manifest.json")
    for problem in problems:
        print(problem)
    print({name: total(manifest[name]) for name in DATASETS})
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check())
