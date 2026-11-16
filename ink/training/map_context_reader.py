"""A second reader that learns the published map from a long reach of the first reader's map."""

import json
import sys
import time

import numpy as np
from PIL import Image

from calibration.ink.ink_yield import rank_auc
from ink.datasets.labelled_cases import load_truth
from storage.data_roots import (
    LABELLED_CASE_STACKS,
    LABELLED_CASES,
    PUBLISHED_PAIRS,
    PUBLISHED_PAIRS_WRITING,
    TRAINED_WEIGHTS,
    calibration_output,
)

CELL = 8
CROP = 192
BATCH = 6
STEPS = 1500
WIDTH = 16
LEVELS = 4
THREADS = 4
FOLDER = calibration_output("map_context")
WINDOWS = FOLDER / "windows"
MAPS = calibration_output("left_out_maps")
LEFT_OUT = {
    name: TRAINED_WEIGHTS / f"F-without-{name.removeprefix('PHerc')}.pth"
    for name in ("PHerc0841", "PHerc0500P2", "PHerc0009B")
}
PUBLISHED = {"d-pairs": PUBLISHED_PAIRS, "writing": PUBLISHED_PAIRS_WRITING}


def cells_of(image):
    h, w = image.shape[0] // CELL * CELL, image.shape[1] // CELL * CELL
    return image[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL).mean(axis=(1, 3)).astype(np.float32)


def read_published(sample):
    """Published 9 um windows of a scroll read by the model that left it out, kept beside the published map on the same cells."""
    from ink.model import tile_reader
    from ink.training.model_training import DEPTH, PAST_FACE, face_layer_of

    reader = tile_reader.TileReader(LEFT_OUT[sample])
    WINDOWS.mkdir(parents=True, exist_ok=True)
    for tag, root in PUBLISHED.items():
        for path in sorted((root / sample).glob("*.npy")):
            out = WINDOWS / f"{sample}-{tag}-{path.stem}.npz"
            png = path.with_suffix(".png")
            if out.exists() or path.name.endswith(".part.npy") or not png.exists():
                continue
            stack = np.load(path)
            first = min(face_layer_of(stack) + PAST_FACE, stack.shape[0] - DEPTH)
            read = reader.read(np.ascontiguousarray(stack[first : first + DEPTH]))
            published = np.asarray(Image.open(png), dtype=np.float32) / 255
            h, w = min(read.shape[0], published.shape[0]), min(read.shape[1], published.shape[1])
            sheet = (stack[stack.shape[0] // 2] > 0)[:h, :w]
            np.savez_compressed(
                out,
                read=cells_of(read[:h, :w]),
                published=cells_of(published[:h, :w]),
                sheet=cells_of(sheet.astype(np.float32)),
            )
            print(out.name, stack.shape, flush=True)


def windows_of(samples):
    """(read, published, sheet) of the kept windows of some scrolls that are large enough and mostly sheet."""
    found = []
    for sample in samples:
        for path in sorted(WINDOWS.glob(f"{sample}-*.npz")):
            kept = np.load(path)
            sheet = (kept["sheet"] > 0.5).astype(np.float32)
            if sheet.mean() > 0.3 and min(sheet.shape) >= CROP:
                found.append((kept["read"].astype(np.float32), kept["published"].astype(np.float32), sheet))
    return found


def labelled_maps(sample):
    """(case, read, ink, known, sheet) on cells, for each labelled window read by the model that left its scroll out."""
    found = []
    for path in sorted(MAPS.glob(f"{sample}-*-near.npy")):
        name = path.name[: -len("-near.npy")]
        record = json.loads((LABELLED_CASES / name / "case.json").read_text())
        stack = np.load(LABELLED_CASE_STACKS / f"{name}.npy", mmap_mode="r")
        read = np.load(path).astype(np.float32) / 255
        ink, known = load_truth(record)
        h, w = min(read.shape[0], ink.shape[0], stack.shape[1]), min(read.shape[1], ink.shape[1], stack.shape[2])
        sheet = np.asarray(stack[stack.shape[0] // 2])[:h, :w] > 0
        found.append(
            (
                name,
                cells_of(read[:h, :w]),
                cells_of(ink[:h, :w].astype(np.float32)),
                cells_of((known[:h, :w] & sheet).astype(np.float32)) > 0.9,
                cells_of(sheet.astype(np.float32)),
            )
        )
    return found


def network(levels=LEVELS, width=WIDTH):
    """A small U-Net on a map and its sheet."""
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = [width * 2**i for i in range(levels + 1)]
            self.down = nn.ModuleList([block(2, w[0])] + [block(w[i], w[i + 1]) for i in range(levels)])
            self.up = nn.ModuleList([block(w[i + 1] + w[i], w[i]) for i in reversed(range(levels))])
            self.out = nn.Conv2d(w[0], 1, 1)

        def forward(self, x):
            kept = []
            for i, down in enumerate(self.down):
                x = down(x if i == 0 else nn.functional.avg_pool2d(x, 2))
                kept.append(x)
            kept.pop()
            for up in self.up:
                skip = kept.pop()
                x = up(
                    torch.cat(
                        [nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False), skip],
                        1,
                    )
                )
            return self.out(x)

    return Net()


def batch_of(windows, rng):
    x, target, seen = (
        np.zeros((BATCH, 2, CROP, CROP), np.float32),
        np.zeros((BATCH, 1, CROP, CROP), np.float32),
        np.zeros((BATCH, 1, CROP, CROP), np.float32),
    )
    for i in range(BATCH):
        read, published, sheet = windows[rng.integers(len(windows))]
        y, c = rng.integers(0, read.shape[0] - CROP + 1), rng.integers(0, read.shape[1] - CROP + 1)
        a, b, s = (part[y : y + CROP, c : c + CROP] for part in (read, published, sheet))
        if rng.random() < 0.5:
            a, b, s = a[::-1], b[::-1], s[::-1]
        if rng.random() < 0.5:
            a, b, s = a[:, ::-1], b[:, ::-1], s[:, ::-1]
        x[i, 0], x[i, 1], target[i, 0], seen[i, 0] = (
            np.clip(a * rng.uniform(0.7, 1.3) + rng.uniform(-0.05, 0.05), 0, 1) * s,
            s,
            b,
            s,
        )
    return x, target, seen


def taught(windows, levels=LEVELS, steps=STEPS, seed=0):
    """A second reader taught on windows."""
    import torch

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = network(levels)
    optimizer = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    start = time.time()
    for step in range(steps):
        x, target, seen = (torch.from_numpy(part) for part in batch_of(windows, rng))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), target, reduction="none")
        loss = (loss * seen).sum() / seen.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0:
            print(step, round(loss.item(), 4), round(time.time() - start), "s", flush=True)
    return net.eval()


def read(net, image, sheet):
    """The second reader's map of a first reader's map on cells."""
    import torch

    h, w = image.shape
    x = np.zeros((1, 2, (h + 31) // 32 * 32, (w + 31) // 32 * 32), np.float32)
    x[0, 0, :h, :w], x[0, 1, :h, :w] = image * sheet, sheet
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(x)))[0, 0, :h, :w].numpy()


def saved(left_out, levels=LEVELS):
    return FOLDER / f"without-{left_out}-levels-{levels}.pth"


def loaded(left_out, levels=LEVELS):
    import torch

    net = network(levels)
    net.load_state_dict(torch.load(saved(left_out, levels)))
    return net.eval()


def study(levels=LEVELS):
    """Each scroll left out in turn."""
    import torch

    rows = []
    for left_out in LEFT_OUT:
        windows = windows_of([s for s in LEFT_OUT if s != left_out])
        net = taught(windows, levels)
        torch.save(net.state_dict(), saved(left_out, levels))
        for name, first, ink, known, sheet in labelled_maps(left_out):
            second = read(net, first, sheet)
            ink_cells, bare_cells = known & (ink > 0.5), known & (ink < 0.05)
            row = {
                "left_out": left_out,
                "levels": levels,
                "case": name,
                "windows": len(windows),
                "first": round(rank_auc(first[ink_cells], first[bare_cells]), 4),
                "second": round(rank_auc(second[ink_cells], second[bare_cells]), 4),
            }
            rows.append(row)
            print(row, flush=True)
    out = FOLDER / f"study-levels-{levels}.json"
    out.write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "windows":
        read_published(sys.argv[2])
    else:
        study(int(sys.argv[2]) if len(sys.argv) > 2 else LEVELS)
