"""Where a published ink map holds rows of writing, scored on a 1.6 mm grid."""

import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from ink.datasets.labelled_cases import load_truth
from storage.data_roots import LABELLED_CASES, PUBLISHED_PAIRS, calibration_output

EVIDENCE = calibration_output("text_evidence")
SEGMENTS = EVIDENCE / "segments.json"
MAPS = PUBLISHED_PAIRS / "maps"
SCROLLS = ("PHerc0009B", "PHerc0139", "PHerc0500P2", "PHerc0814", "PHerc0841")
FLOODED = ("PHerc0139", "20250108000005")
CELL_MM = 0.1
TILE = 64
WIDE = 88
STEP = 16
SCALES = (0.8, 1.0, 1.3)
LEVEL, TURNED = [0, 1, 4, 5], [2, 3, 6, 7]
TEXT = 2.0
Image.MAX_IMAGE_PIXELS = None


def turn(a, k):
    """The kth of the eight symmetries on the last two axes."""
    a = np.rot90(a, k // 2, axes=(-2, -1))
    return a[..., ::-1] if k % 2 else a


def segments():
    return json.loads(SEGMENTS.read_text())


def pixel_mm(row, image):
    return row["shape"][1] * row["steps"][1] / 1000 / image.size[1]


def at_cell(image, px, cell):
    size = (max(int(image.size[0] * px / cell), 1), max(int(image.size[1] * px / cell), 1))
    return np.asarray(image.resize(size, Image.BOX))


def labelled_tiles():
    """Each labelled case at each scale."""
    rows = {(r["sample"], r["segment"]): r for r in segments()}
    found = []
    for path in sorted(LABELLED_CASES.glob("*/case.json")):
        case = json.loads(path.read_text())
        if not 7 <= case["voxel_um"] <= 10 or case["sample"] not in SCROLLS:
            continue
        key = (case["sample"], case["case_id"].split("-")[1])
        file = MAPS / f"{key[0]}-{key[1]}.png"
        if key not in rows or not file.exists() or key == FLOODED:
            continue
        image = Image.open(file).convert("L")
        px = pixel_mm(rows[key], image)
        ink, _ = load_truth(case)
        voxel = case["voxel_um"] / 1000
        for scale in SCALES:
            cell = CELL_MM * scale
            a = at_cell(image, px, cell)
            f = max(int(round(cell / voxel)), 1)
            h, w = ink.shape[0] // f * f, ink.shape[1] // f * f
            share = ink[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
            total = np.pad(share, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            top, left = int(case["box"][0] * voxel / cell), int(case["box"][2] * voxel / cell)
            spots = []
            for y in range(0, share.shape[0] - TILE, 8):
                for x in range(0, share.shape[1] - TILE, 8):
                    mean = (total[y + TILE, x + TILE] - total[y, x + TILE] - total[y + TILE, x] + total[y, x]) / TILE**2
                    corner_y, corner_x = top + y - (WIDE - TILE) // 2, left + x - (WIDE - TILE) // 2
                    if (
                        mean > 0.06
                        and corner_y >= 0
                        and corner_x >= 0
                        and corner_y + WIDE <= a.shape[0]
                        and corner_x + WIDE <= a.shape[1]
                    ):
                        spots.append((corner_y, corner_x))
            if spots:
                found.append({"sample": case["sample"], "map": a, "spots": np.array(spots)})
    return found


class RowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 1
        for out, kernel in ((16, 5), (32, 3), (64, 3), (64, 3)):
            layers += [
                torch.nn.Conv2d(channels, out, kernel, stride=2, padding=kernel // 2),
                torch.nn.GroupNorm(4, out),
                torch.nn.ReLU(),
            ]
            channels = out
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.head(self.body(x))


def normal(block):
    block = block.astype(np.float32)
    block = block - block.mean(axis=(-2, -1), keepdims=True)
    return block / (block.std(axis=(-2, -1), keepdims=True) + 1e-3)


def tilted(wide, angle):
    margin = (WIDE - TILE) // 2
    return np.asarray(Image.fromarray(np.ascontiguousarray(wide)).rotate(angle, Image.BILINEAR))[
        margin : margin + TILE, margin : margin + TILE
    ]


def batch(pool, size, rng):
    images, labels = [], []
    for _ in range(size):
        item = pool[rng.integers(len(pool))]
        y, x = item["spots"][rng.integers(len(item["spots"]))]
        g = int(rng.integers(8))
        images.append(
            np.ascontiguousarray(turn(tilted(item["map"][y : y + WIDE, x : x + WIDE], rng.uniform(-8, 8)), g))
        )
        labels.append(g)
    return torch.from_numpy(normal(np.stack(images))[:, None]), torch.tensor(labels)


def taught(pool, steps=1200, seed=11, lr=2e-3):
    torch.manual_seed(seed)
    net = RowNet()
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=steps)
    net.train()
    for _ in range(steps):
        x, y = batch(pool, 96, rng)
        loss = torch.nn.functional.cross_entropy(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return net.eval()


def level_minus_turned(net, tiles):
    with torch.inference_mode():
        p = torch.log_softmax(net(torch.from_numpy(normal(tiles))[:, None]), 1)
        return (torch.logsumexp(p[:, LEVEL], 1) - torch.logsumexp(p[:, TURNED], 1)).numpy()


def grid_of(net, a):
    """Row evidence of every tile with enough of the sheet and some strong ink, NaN elsewhere."""
    ny, nx = (a.shape[0] - TILE) // STEP + 1, (a.shape[1] - TILE) // STEP + 1
    out = np.full((max(ny, 0), max(nx, 0)), np.nan, dtype=np.float32)
    spots = []
    for i in range(ny):
        for j in range(nx):
            tile = a[i * STEP : i * STEP + TILE, j * STEP : j * STEP + TILE]
            if (tile > 0).mean() > 0.9 and (tile >= 128).mean() > 0.03:
                spots.append((i, j))
    for k in range(0, len(spots), 256):
        part = spots[k : k + 256]
        tiles = np.stack([a[i * STEP : i * STEP + TILE, j * STEP : j * STEP + TILE] for i, j in part])
        turned = np.ascontiguousarray(np.rot90(tiles, 1, axes=(1, 2)))
        for (i, j), value in zip(part, 0.5 * (level_minus_turned(net, tiles) - level_minus_turned(net, turned))):
            out[i, j] = value
    return out


def net_for(sample):
    """The network that never saw this scroll."""
    net = RowNet()
    net.load_state_dict(torch.load(EVIDENCE / f"without-{sample if sample in SCROLLS else 'PHerc0139'}.pth"))
    return net.eval()


def text_known(sample, segment, y, x, size, cut=TEXT):
    """Which pixels of a window of the segment's scan lie in a tile of writing."""
    grid = np.load(EVIDENCE / f"{sample}-{segment}.npz")
    evidence = np.nan_to_num(grid["evidence"].astype(np.float32), nan=-9.0)
    voxel, cell = float(grid["voxel_mm"]), float(grid["cell_mm"])
    span = float(grid["step_cells"]) * cell
    offset = float(grid["tile_cells"]) * cell / 2 - span / 2
    pixels = np.arange(size)
    rows = np.clip(np.floor(((y + pixels) * voxel - offset) / span).astype(int), 0, evidence.shape[0] - 1)
    cols = np.clip(np.floor(((x + pixels) * voxel - offset) / span).astype(int), 0, evidence.shape[1] - 1)
    return evidence[np.ix_(rows, cols)] >= cut


def score_all():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    nets, start = {}, time.time()
    for row in segments():
        file = MAPS / f"{row['sample']}-{row['segment']}.png"
        if not file.exists():
            continue
        image = Image.open(file).convert("L")
        a = at_cell(image, pixel_mm(row, image), CELL_MM)
        if min(a.shape) <= TILE + STEP:
            continue
        net = nets.setdefault(row["sample"], net_for(row["sample"]))
        grid = grid_of(net, a)
        np.savez_compressed(
            EVIDENCE / f"{row['sample']}-{row['segment']}.npz",
            evidence=grid.astype(np.float16),
            cell_mm=CELL_MM,
            step_cells=STEP,
            tile_cells=TILE,
            segment_rows=row["shape"][1],
            segment_cols=row["shape"][2],
            voxel_mm=row["steps"][1] / 1000,
        )
        print(
            row["sample"],
            row["segment"],
            int(np.isfinite(grid).sum()),
            int((grid >= TEXT).sum()),
            round(time.time() - start),
            "s",
            flush=True,
        )


def teach_all():
    pool = labelled_tiles()
    for left in SCROLLS:
        torch.save(
            taught([item for item in pool if item["sample"] != left]).state_dict(), EVIDENCE / f"without-{left}.pth"
        )
        print("taught without", left, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    if "teach" in sys.argv:
        teach_all()
    score_all()
