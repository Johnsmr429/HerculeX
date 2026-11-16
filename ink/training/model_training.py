"""Ink model training for scans near 9 µm on labelled cases, published pairs and blank controls."""

import copy
import importlib
import json
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ni

from ink.datasets.labelled_cases import load_truth
from ink.datasets.own_reads import read_path
from ink.datasets.stroke_core import core_of_patch
from ink.datasets.text_evidence import text_known
from ink.model import stock_model, tile_reader
from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of
from sampling.sheet_images import largest_mosaic
from storage.brick_store import BrickStore
from storage.data_roots import (
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    OWN_MESH_LABELS,
    PUBLISHED_PAIRS,
    PUBLISHED_PAIRS_WRITING,
    ROUTE_OUTPUTS,
    TARGET_STACKS,
    TRAINED_WEIGHTS,
    stage_folder,
)

PATCH = 256
BATCH = 2
LEARNING_RATE = 1e-4
STEPS = 8000
JITTER = 3
NEAR_INK = 0.7
BACK_OF_SHEET = 0.3
PUBLISHED = 0.6
SCRAMBLED = 0.1
SURE_INK, SURE_BLANK, TEXT_CUT = 160, 64, 2.0
STEM_LEARNS = 1
WANDERING = 0.0
WANDER_LAYERS, WANDER_PX, WANDER_REACH = 1.5, 16, 4
BLANK_SHEET = 0.0
BLANK_SHEET_UNITS = ("rows-0000-0012", "rows-0011-0023", "rows-0022-0034")
QUARTER_TURNS = 4
WRITING_WINDOWS = 0
WARPED = 0.0
WARP_TURN, WARP_SCALE, WARP_SHEAR = 0.35, 0.15, 0.1
MOVED = 0.0
MOVED_STRENGTH = (0.5, 1.2)
MOVED_DEPTH = 3
MOVED_RING_PX = 24
MEAN_BLOCK = 32
SHARES = {"PHerc0500P2": 0.25, "PHerc0841": 0.25, "PHerc0009B": 0.25, "PHerc0139": 0.2, "PHerc0814": 0.05}
LEAST_CORRELATION, LEAST_OVER_SHUFFLED = 0.2, 4.0
STROKE_CORE = 0
OWN_READS = None
READ_INK, READ_BLANK = 166, 89
PIXEL_MM = 0.00936
NEVER_TAUGHT = ("PHerc0841-20260220214732-9.366um-1.2m-113keV-0086",)


def cases():
    """Every hand-labelled case of a scan near 9 um."""
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        if case["sample"] in SHARES and 7 <= case["voxel_um"] <= 10:
            yield case


class Labelled:
    """A case's layers and its labels, kept as one array on disc."""

    def __init__(self, case):
        self.sample, self.name, self.fixed = case["sample"], case["case_id"], {}
        path = LABELLED_CASE_STACKS / f"{case['case_id']}.npy"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            store = BrickStore(LABELLED_CASES / case["case_id"] / "layers")
            y0, y1, x0, x1 = case["box"]
            np.save(path, store.read((0, store.shape[0]), (y0, y1), (x0, x1)))
        self.stack = np.load(path, mmap_mode="r")
        self.ink, known = load_truth(case)
        self.known = known & (np.asarray(self.stack[self.stack.shape[0] // 2]) > 0)
        self.start, self.back_start = face_layer_of(self.stack), face_layer_of(self.stack[::-1])
        self.inked = np.argwhere(self.ink & self.known)[::101].copy()
        self.spots = np.argwhere(self.known)[::509].copy()


class OwnLabelled:
    """Hand labels carried onto one of our own pieces, its stack turned as the published one runs."""

    def __init__(self, path):
        self.sample, self.name, self.fixed = path.parent.name, path.stem, {}
        self.stack = np.load(path, mmap_mode="r")
        self.ink, self.known = self.labels(path)
        self.start, self.back_start = face_layer_of(self.stack), face_layer_of(self.stack[::-1])
        self.inked = np.argwhere(self.ink & self.known)[::101].copy()
        self.spots = np.argwhere(self.known)[::509].copy()

    def labels(self, path):
        labels = np.load(path.with_name(f"{path.stem}-labels.npz"))
        return labels["ink"], labels["known"]


class OwnRead(OwnLabelled):
    """One of our own pieces with a reader's sure ink and sure blank as its labels, no hand in them."""

    def labels(self, path):
        read = np.load(read_path(path))
        on_sheet = np.asarray(self.stack[self.stack.shape[0] // 2]) > 0
        return read >= READ_INK, ((read >= READ_INK) | (read <= READ_BLANK)) & on_sheet


def own_reads(folders, held_out):
    """Windows of our own pieces with their kept reads, those named after what is held out left aside."""
    paths = [p for folder in folders for p in sorted(folder.glob("*.npy"))]
    found = [OwnRead(p) for p in paths if held_out not in p.stem and p.stem not in NEVER_TAUGHT]
    print("own reads", len(found), "windows, known", sum(int(case.known.sum()) for case in found), flush=True)
    return found


def own_cases(left_out):
    """Carried labels that raw scan values place on their sheet."""
    found = []
    for path in sorted((OWN_MESH_LABELS / "grown_route").glob("PHerc*/*.npy")):
        row = json.loads(path.with_suffix(".json").read_text())
        if (
            path.parent.name.split("-")[0] != left_out
            and path.stem not in NEVER_TAUGHT
            and row["correlation"] >= max(LEAST_CORRELATION, LEAST_OVER_SHUFFLED * row["shuffled"])
        ):
            found.append(OwnLabelled(path))
    print("own-mesh labels", {case.name[-30:]: int(case.known.sum()) for case in found}, flush=True)
    return found


class Above:
    """A map read as a mask a slice at a time, so hundreds of windows stay small."""

    def __init__(self, image):
        self.image = image
        self.shape = image.shape

    def __getitem__(self, where):
        return self.image[where] >= SURE_INK


class Known(Above):
    """What a published map is sure of, on the sheet, and only where it shows writing at all."""

    def __init__(self, image, text):
        super().__init__(image)
        self.text = text

    def __getitem__(self, where):
        part = self.image[where]
        return ((part >= SURE_INK) | ((part <= SURE_BLANK) & (part > 0))) & self.text[where]


class PublishedWindow:
    """A window of a 9 um surface volume with the published map of the same surface from a better scan."""

    def __init__(self, path):
        self.sample, self.name, self.fixed = path.parent.name, path.stem, {}
        self.stack = np.load(path, mmap_mode="r")
        image = np.asarray(Image.open(path.with_suffix(".png")))
        segment, y, x = path.stem.rsplit("-", 2)
        text = text_known(self.sample, segment, int(y), int(x), image.shape[0], cut=TEXT_CUT)
        self.ink, self.known = Above(image), Known(image, text)
        faces = path.with_suffix(".json")
        if not faces.exists():
            faces.write_text(json.dumps([face_layer_of(self.stack), face_layer_of(self.stack[::-1])]))
        self.start, self.back_start = json.loads(faces.read_text())
        self.inked = np.argwhere((image >= SURE_INK) & text)[::211].copy()
        self.spots = np.argwhere((image > 0) & text)[::1009].copy()


def published_of(left_out):
    """Every downloaded window with writing, by scroll."""
    found = {}
    for path in sorted(
        p
        for folder in (PUBLISHED_PAIRS, PUBLISHED_PAIRS_WRITING)[: 1 + WRITING_WINDOWS]
        for p in folder.glob("PHerc*/*.npy")
    ):
        if path.parent.name != left_out and not path.name.endswith(".part.npy"):
            window = PublishedWindow(path)
            if len(window.inked) >= 20:
                found.setdefault(path.parent.name, []).append(window)
    print("published windows", {k: len(v) for k, v in found.items()}, flush=True)
    return found


class Background:
    """An unlabelled stack of the target."""

    def __init__(self, stack):
        self.stack, self.fixed = stack, {}
        self.start = face_layer_of(stack)
        middle = np.asarray(stack[stack.shape[0] // 2, ::16, ::16]) > 0
        self.spots = np.argwhere(ni.binary_erosion(middle, iterations=6)) * 16

    def crop(self, y, x, offset):
        h, w = self.stack.shape[1:]
        y, x = min(max(y, 0), max(h - PATCH, 0)), min(max(x, 0), max(w - PATCH, 0))
        layer = int(np.clip(self.start + offset, 0, self.stack.shape[0] - DEPTH))
        block = np.zeros((DEPTH, PATCH, PATCH), dtype=np.float32)
        part = self.stack[layer : layer + DEPTH, y : y + PATCH, x : x + PATCH]
        block[:, : part.shape[1], : part.shape[2]] = part
        return block


def backgrounds_of(target):
    """Every stack kept of the scroll to be read, or a left-out scroll's crops with their labels unused."""
    if target == "PHerc0211":
        return [
            Background(np.load(path, mmap_mode="r"))
            for path in sorted((TARGET_STACKS / "PHerc0211").glob("*/piece-*.npy"))
        ]
    return [Background(Labelled(case).stack) for case in cases() if case["sample"] == target]


def crop_of(case, y, x, back):
    """(layers, ink, known) of a patch of a labelled surface."""
    h, w = case.known.shape
    y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + PATCH, h), min(x + PATCH, w)
    stack = case.stack[::-1] if back else case.stack
    start = case.back_start if back else case.start
    layer = int(np.clip(start + PAST_FACE + np.random.randint(-JITTER, JITTER + 1), 0, stack.shape[0] - DEPTH))
    block = np.zeros((DEPTH, PATCH, PATCH), dtype=np.float32)
    ink, known = np.zeros((2, PATCH, PATCH), dtype=np.float32)
    here = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    there = (slice(y0, y1), slice(x0, x1))
    block[:, here[0], here[1]] = (
        wandering(stack, layer, there)
        if np.random.rand() < WANDERING
        else stack[layer : layer + DEPTH, there[0], there[1]]
    )
    drawn = core_of_patch(case.ink, case.known.shape, there, PIXEL_MM) if STROKE_CORE else case.ink[there]
    ink[here], known[here] = (0 if back else drawn), case.known[there]
    return block, ink, known


def wandering(stack, layer, there):
    """A patch's layers along a surface that wanders round the true one, as our meshes do round the published ones."""
    low, high = max(layer - WANDER_REACH, 0), min(layer + DEPTH + WANDER_REACH, stack.shape[0])
    wide = np.asarray(stack[low:high, there[0], there[1]], dtype=np.float32)
    field = ni.gaussian_filter(np.random.normal(size=wide.shape[1:]).astype(np.float32), WANDER_PX)
    field *= np.float32(np.random.uniform(0, WANDER_LAYERS) / max(float(field.std()), 1e-6))
    depth = np.clip(np.arange(DEPTH, dtype=np.float32)[:, None, None] + (layer - low) + field, 0, wide.shape[0] - 1.001)
    whole = depth.astype(np.int64)
    part = depth - whole
    return np.take_along_axis(wide, whole, 0) * (1 - part) + np.take_along_axis(wide, whole + 1, 0) * part


def colour_change():
    """A change of grey levels as scans differ, gain and gamma, as a function of raw values."""
    gain, lift, power = (
        np.float32(v)
        for v in (np.exp(np.random.normal(0, 0.15)), np.random.normal(0, 0.03), np.exp(np.random.normal(0, 0.2)))
    )
    return lambda values: (
        np.clip(np.asarray(values, dtype=np.float32) * (gain / np.float32(255)) + lift, 0, 1) ** power * np.float32(255)
    )


def recoloured(block, change):
    """A patch changed in grey levels, with a little noise and softness."""
    a = change(block)
    a = a + np.random.normal(0, 255 * 0.012 * np.random.rand(), a.shape).astype(np.float32)
    if np.random.rand() < 0.3:
        a = ni.gaussian_filter(a, (0, np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0)))
    return np.clip(a, 0, 255)


def changed_scale(scale, change):
    """A stack's raw scale after a change of grey levels."""
    low, high, centre, spread = scale
    low, high, centre, above, below = change([low, high, centre, centre + spread, centre - spread])
    return float(low), float(high), float(centre), max(float(above - below) / 2, 1.0)


def scrambled_patch(backgrounds):
    """A patch of the target with every stroke and fibre destroyed and its spectrum, depth profile and grey levels kept."""
    target = backgrounds[np.random.randint(len(backgrounds))]
    spot = target.spots[np.random.randint(len(target.spots))]
    block = target.crop(
        int(spot[0]) - PATCH // 2, int(spot[1]) - PATCH // 2, PAST_FACE + np.random.randint(-JITTER, JITTER + 1)
    )
    seen = block[DEPTH // 2] > 0
    means = np.array([layer[seen].mean() if seen.any() else 0.0 for layer in block], dtype=np.float32)[:, None, None]
    phases = np.exp(2j * np.pi * np.random.rand(PATCH, PATCH // 2 + 1))
    out = np.fft.irfft2(np.fft.rfft2(np.where(seen, block - means, 0)) * phases, s=(PATCH, PATCH)) + means
    return (
        target,
        np.clip(out, 1, 255).astype(np.float32),
        np.zeros((PATCH, PATCH), dtype=np.float32),
        seen.astype(np.float32),
    )


def blank_sheets(sample="PHerc0211", volume="20250821151803"):
    """Both faces of the kept stacks of a layout's largest area, in the units taught on."""
    members = largest_mosaic(
        json.loads((stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / "layout.json").read_text())["pieces"]
    )
    paths = [
        TARGET_STACKS / sample / p["unit"] / f"piece-{p['piece']:04d}.npy"
        for p in members
        if p["unit"] in BLANK_SHEET_UNITS
    ]
    stacks = [np.load(path, mmap_mode="r") for path in paths if path.exists()]
    return [Background(s) for s in stacks] + [Background(s[::-1]) for s in stacks]


def blank_patch(sheets):
    sheet = sheets[np.random.randint(len(sheets))]
    spot = sheet.spots[np.random.randint(len(sheet.spots))]
    block = sheet.crop(
        int(spot[0]) - PATCH // 2, int(spot[1]) - PATCH // 2, PAST_FACE + np.random.randint(-JITTER, JITTER + 1)
    )
    return sheet, block, np.zeros((PATCH, PATCH), dtype=np.float32), (block[DEPTH // 2] > 0).astype(np.float32)


def fine_part(block):
    """What a patch's layers hold above their slow part."""
    slow = block.reshape(block.shape[0], PATCH // MEAN_BLOCK, MEAN_BLOCK, PATCH // MEAN_BLOCK, MEAN_BLOCK).mean(
        axis=(2, 4)
    )
    return block - ni.zoom(slow, (1, MEAN_BLOCK, MEAN_BLOCK), order=1, grid_mode=True, mode="nearest")


def moved_patch(labelled, shares, backgrounds):
    """Letters of a labelled scroll moved layer by layer into the target's papyrus at some depth and strength."""
    names = list(labelled)
    group = labelled[names[np.random.choice(len(names), p=shares)]]
    case = group[np.random.randint(len(group))]
    if not len(case.inked):
        return None
    y, x = case.inked[np.random.randint(len(case.inked))] - np.random.randint(PATCH // 8, PATCH - PATCH // 8, size=2)
    source, ink, known = crop_of(case, int(y), int(x), False)
    mask = (ink > 0) & (known > 0)
    if mask.mean() < 0.02:
        return None
    real = np.random.rand() < 0.5
    if not real:
        for _ in range(8):
            y, x = case.spots[np.random.randint(len(case.spots))] - PATCH // 2
            source, other, seen = crop_of(case, int(y), int(x), False)
            if other[mask].mean() < 0.02 and seen[mask].mean() > 0.9:
                break
        else:
            return None
    target = backgrounds[np.random.randint(len(backgrounds))]
    spot = target.spots[np.random.randint(len(target.spots))]
    block = target.crop(int(spot[0]) - PATCH // 2, int(spot[1]) - PATCH // 2, PAST_FACE)
    sheet = block[DEPTH // 2] > 0
    if sheet.mean() < 0.5:
        return None
    soft = ni.gaussian_filter(mask.astype(np.float32), 2.0) * sheet
    fine = np.roll(fine_part(source), np.random.randint(-MOVED_DEPTH, MOVED_DEPTH + 1), axis=0)
    out = np.clip(block + np.random.uniform(*MOVED_STRENGTH) * fine * soft, 1, 255).astype(np.float32)
    ring = ni.binary_dilation(mask, iterations=MOVED_RING_PX) & sheet
    return (
        target,
        out,
        (mask & sheet).astype(np.float32) if real else np.zeros((PATCH, PATCH), dtype=np.float32),
        ring.astype(np.float32),
    )


def sample_of(labelled, shares, published, backgrounds, sheets):
    """One patch."""
    names = list(labelled)
    while True:
        back = False
        moved = moved_patch(labelled, shares, backgrounds) if np.random.rand() < MOVED else None
        if moved:
            case, block, ink, known = moved
        elif np.random.rand() < BLANK_SHEET:
            case, block, ink, known = blank_patch(sheets)
        elif np.random.rand() < SCRAMBLED:
            case, block, ink, known = scrambled_patch(backgrounds)
        else:
            group = (
                labelled["own-reads"]
                if OWN_READS and np.random.rand() < OWN_READS[0]
                else list(published.values())[np.random.randint(len(published))]
                if np.random.rand() < PUBLISHED
                else labelled[names[np.random.choice(len(names), p=shares)]]
            )
            case = group[np.random.randint(len(group))]
            spots = case.inked if np.random.rand() < NEAR_INK and len(case.inked) else case.spots
            y, x = spots[np.random.randint(len(spots))] - np.random.randint(PATCH // 8, PATCH - PATCH // 8, size=2)
            back = case.stack.shape[0] > DEPTH and np.random.rand() < BACK_OF_SHEET
            block, ink, known = crop_of(case, int(y), int(x), back)
        if known.mean() >= 0.2:
            break
    if np.random.rand() < 0.5:
        block, ink, known = block[:, ::-1], ink[::-1], known[::-1]
    if np.random.rand() < 0.5:
        block, ink, known = block[:, :, ::-1], ink[:, ::-1], known[:, ::-1]
    turns = np.random.randint(QUARTER_TURNS)
    block, ink, known = np.rot90(block, turns, (1, 2)), np.rot90(ink, turns), np.rot90(known, turns)
    change = colour_change()
    return (
        case,
        back,
        change,
        np.ascontiguousarray(recoloured(block, change)),
        np.ascontiguousarray(ink),
        np.ascontiguousarray(known),
    )


def batch_of(*arguments):
    return [sample_of(*arguments) for _ in range(BATCH)]


def statistics_of(stem, case, back):
    """A surface's stem statistics as the reader will take them, kept with it."""
    if back not in case.fixed:
        stack, start = (case.stack[::-1], case.back_start) if back else (case.stack, case.start)
        layer = int(np.clip(start + PAST_FACE, 0, stack.shape[0] - DEPTH))
        case.fixed[back] = stem.statistics(stack[layer : layer + DEPTH])
    return case.fixed[back]


def warped(torch, block, ink, known):
    """A patch and its labels turned, scaled and sheared a little, as hands, letter sizes and scans differ."""
    if np.random.rand() >= WARPED:
        return block, ink, known
    angle, size = np.random.uniform(-WARP_TURN, WARP_TURN), np.exp(np.random.uniform(-WARP_SCALE, WARP_SCALE))
    shear = np.random.uniform(-WARP_SHEAR, WARP_SHEAR)
    matrix = torch.tensor(
        [
            [
                [size * np.cos(angle), size * (shear - np.sin(angle)), 0.0],
                [size * np.sin(angle), size * np.cos(angle), 0.0],
            ]
        ],
        dtype=torch.float32,
        device=block.device,
    )
    grid = torch.nn.functional.affine_grid(matrix, (1, 1, PATCH, PATCH), align_corners=False)
    both = torch.nn.functional.grid_sample(
        torch.cat([block, ink, known])[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    return both[:DEPTH], both[DEPTH : DEPTH + 1], (both[DEPTH + 1 :] > 0.99).float()


def fused_batch(torch, stem, samples, device, learning):
    """The stem's output for a batch of patches, each on its own stack's statistics, and its labels."""
    parts, inks, knowns = [], [], []
    for case, back, change, block, ink, known in samples:
        scale, weights = statistics_of(stem, case, back)
        moved = changed_scale(scale, change)
        block, ink, known = warped(
            torch,
            torch.from_numpy(block).to(device),
            torch.from_numpy(ink)[None].to(device),
            torch.from_numpy(known)[None].to(device),
        )
        image = tile_reader.scaled(block, moved)[None]
        parts.append(learning(image[:, None].float()) if STEM_LEARNS else stem(image, (moved, weights)))
        inks.append(ink)
        knowns.append(known)
    return torch.cat(parts).float(), torch.stack(inks), torch.stack(knowns)


def loss_of(torch, logits, ink, known):
    """The stock training loss on what is known."""
    cross = torch.nn.functional.binary_cross_entropy_with_logits(logits, ink * 0.5 + 0.25, reduction="none")
    cross = (cross * known).sum() / known.sum().clamp(min=1.0)
    p, t = logits.sigmoid() * known, ink * known
    return cross + (1 - (2 * (p * t).sum(dim=(1, 2, 3)) + 1) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1)).mean()


def learning_as_folded(torch, conv):
    """A 3x3x3 convolution run as the 2-D one of stock_model.folded with its live weights."""

    def forward(x):
        depth = x.shape[2]
        padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1))
        stacked = torch.cat([padded[:, :, k : k + depth] for k in range(3)], dim=1).permute(0, 2, 1, 3, 4).flatten(0, 1)
        weight = conv.weight.permute(0, 2, 1, 3, 4).flatten(1, 2)
        flat = torch.nn.functional.conv2d(stacked, weight, padding=1)
        return flat.reshape(x.shape[0], depth, -1, x.shape[3], x.shape[4]).permute(0, 2, 1, 3, 4)

    conv.forward = forward


def optimizer_of(torch, parameters):
    return torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=1e-4, fused=True)


def taught(name, target, left_out=None, seed=42, steps=STEPS, own_mesh=0.0):
    """Teach one model from its stock seed."""
    import torch

    sys.path.insert(0, str(stock_model.source()))
    api = importlib.import_module("vesuvius.ink_detection.inference.infer")
    device = torch.device("mps")
    configured = api.configure_model(
        Namespace(checkpoint=stock_model.checkpoint(f"hybrid_3d2d-seed{seed}/step-075000.pth"), amp_dtype="default")
    )
    net = configured.model.to(device).float()
    net.model.depth_fusion.requires_grad_(bool(STEM_LEARNS))
    stem = tile_reader.DepthStem(copy.deepcopy(net.model.depth_fusion), "mps")
    if STEM_LEARNS:
        for at in (0, 3):
            learning_as_folded(torch, net.model.depth_fusion.features[at])
    labelled = {}
    for case in cases():
        if case["sample"] != left_out:
            labelled.setdefault(case["sample"], []).append(Labelled(case))
    shares = np.array([SHARES[n] for n in labelled])
    shares /= shares.sum()
    if own_mesh:
        labelled["own-mesh"] = own_cases(left_out)
        shares = np.append(shares * (1 - own_mesh), own_mesh)
    if OWN_READS:
        labelled["own-reads"] = own_reads(*OWN_READS[1:])
        shares = np.append(shares, 0.0)
    published, backgrounds = published_of(left_out), backgrounds_of(target)
    net.train()
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = optimizer_of(torch, parameters)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    arguments = (labelled, shares, published, backgrounds, blank_sheets() if BLANK_SHEET else [])
    ahead = ThreadPoolExecutor(3)
    coming = [ahead.submit(batch_of, *arguments) for _ in range(6)]
    start, losses = time.time(), []
    for step in range(1, steps + 1):
        samples = coming.pop(0).result()
        coming.append(ahead.submit(batch_of, *arguments))
        fused, ink, known = fused_batch(torch, stem, samples, device, net.model.depth_fusion)
        logits = net.model.network(fused)["ink"]
        loss = loss_of(torch, logits[0] if isinstance(logits, (list, tuple)) else logits, ink, known)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0:
            print(
                name,
                "step",
                step,
                "loss",
                round(float(np.mean(losses[-500:])), 4),
                round(step * BATCH / (time.time() - start), 1),
                "patches/s",
                flush=True,
            )
    out = TRAINED_WEIGHTS / f"{name}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
    del net, optimizer
    torch.mps.empty_cache()
    return out


if __name__ == "__main__":
    print(taught(sys.argv[1], sys.argv[2], *sys.argv[3:4], *map(int, sys.argv[4:6])), flush=True)
