"""Ink targets kept only where two truths agree: stroke cores as ink, papyrus away from any writing as blank, the rest unknown."""

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from calibration.ink.stroke_widths import fine_map
from ink.datasets.labelled_cases import load_truth
from ink.datasets.text_evidence import text_known
from storage.data_roots import AGREED_TARGETS, LABELLED_CASE_STACKS, LABELLED_CASES, OWN_MESH_LABELS, PUBLISHED_PAIRS

CORE, WRITING, BLANK = 200, 128, 64
RIM_MM = 0.1
TEXT_CUT = 2.0
ROUTES = ("level_route", "grown_route")
UNKNOWN_CODE, BLANK_CODE, INK_CODE = 0, 1, 2


def path_of(kind, name):
    return AGREED_TARGETS / kind / f"{name}.npy"


class Coded:
    """Pixels of an agreed target at or above one code, read a slice at a time from the mapped file."""

    def __init__(self, codes, least):
        self.codes, self.least, self.shape = codes, least, codes.shape

    def __getitem__(self, where):
        return self.codes[where] >= self.least


def targets_of(kind, name):
    """(ink, known, inked pixels, known pixels) of a kept agreed target, the masks read lazily."""
    codes = np.load(path_of(kind, name), mmap_mode="r")
    whole = np.asarray(codes)
    return (
        Coded(codes, INK_CODE),
        Coded(codes, BLANK_CODE),
        np.argwhere(whole == INK_CODE),
        np.argwhere(whole >= BLANK_CODE),
    )


def cases():
    """Hand-labelled cases of scans near 9 um with a fine-scan map of their segment."""
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        maps = PUBLISHED_PAIRS / "maps" / f"{case['sample']}-{case['segment']}.png"
        if 7 <= case["voxel_um"] <= 10 and maps.exists() and (LABELLED_CASE_STACKS / f"{case['case_id']}.npy").exists():
            yield case


def of_case(case):
    """A case's ink where the label holds a stroke core of the fine map, blank where neither shows writing near."""
    stack = np.load(LABELLED_CASE_STACKS / f"{case['case_id']}.npy", mmap_mode="r")
    label, known = load_truth(case)
    fine = fine_map(case)
    height, width = (min(a.shape[i] for a in (stack[0], label, fine)) for i in (0, 1))
    label, known, fine = label[:height, :width], known[:height, :width], fine[:height, :width]
    known &= np.asarray(stack[stack.shape[0] // 2, :height, :width]) > 0
    rim = RIM_MM * 1000 / case["voxel_um"]
    writing = label | (fine >= WRITING)
    ink = known & label & (fine >= CORE)
    blank = known & (ni.distance_transform_edt(~writing) > rim) & (fine <= BLANK) & (fine > 0)
    return ink, ink | blank


def of_published(path):
    """A published window's ink where its fine map holds a stroke core inside tiles of writing, blank away from any writing there."""
    image = np.asarray(Image.open(path.with_suffix(".png")))
    segment, y, x = path.stem.rsplit("-", 2)
    text = text_known(path.parent.name, segment, int(y), int(x), image.shape[0], cut=TEXT_CUT)
    rim = RIM_MM * 1000 / 9.36
    ink = text & (image >= CORE)
    blank = text & (ni.distance_transform_edt(image < WRITING) > rim) & (image <= BLANK) & (image > 0)
    return ink, ink | blank


def of_own(path, case_targets):
    """A labelled case's agreed targets carried onto one of our own pieces through each pixel's published pixel."""
    labels = np.load(path.with_name(f"{path.stem}-labels.npz"))
    ys, xs, rows, columns = labels["published"]
    ink, known = case_targets
    inside = (rows < ink.shape[0]) & (columns < ink.shape[1])
    ys, xs, rows, columns = ys[inside], xs[inside], rows[inside], columns[inside]
    own_ink, own_known = np.zeros((2, *labels["known"].shape), dtype=bool)
    own_ink[ys, xs], own_known[ys, xs] = ink[rows, columns], known[rows, columns]
    return own_ink, own_known


def save(kind, name, ink, known):
    """One target as a byte a pixel: unknown, blank or ink."""
    out = path_of(kind, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    codes = np.where(ink & known, INK_CODE, np.where(known, BLANK_CODE, UNKNOWN_CODE)).astype(np.uint8)
    np.save(out.with_suffix(".part.npy"), codes)
    out.with_suffix(".part.npy").replace(out)
    print(kind, name, "ink", int((codes == INK_CODE).sum()), "blank", int((codes == BLANK_CODE).sum()), flush=True)


def build():
    """Agreed targets for every labelled case, every published window and every carried own-mesh window."""
    by_case = {}
    for case in cases():
        by_case[case["case_id"]] = of_case(case)
        save("labelled_cases", case["case_id"], *by_case[case["case_id"]])
    for path in sorted(PUBLISHED_PAIRS.glob("PHerc*/*.npy")):
        if not path.name.endswith(".part.npy") and path.with_suffix(".png").exists():
            save(f"published_pairs/{path.parent.name}", path.stem, *of_published(path))
    for route in ROUTES:
        for path in sorted((OWN_MESH_LABELS / route).glob("PHerc*/*.npy")):
            row = json.loads(path.with_suffix(".json").read_text())
            if row["case"] in by_case and "published" in np.load(path.with_name(f"{path.stem}-labels.npz")):
                save(f"own_mesh_labels/{route}/{path.parent.name}", path.stem, *of_own(path, by_case[row["case"]]))


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
