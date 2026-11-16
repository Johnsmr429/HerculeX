"""A small network that names a Greek letter from a window of ink map, taught on the letters named by eye.

Data: datasets/eye_letters (ink/datasets/eye_letters.py), each letter a window SIDE reference heights on a side of the
published ink map (ink bright), the reference height being the median letter height of its row. The network sees the
middle VIEW reference heights at IN pixels, so a letter of the row's height fills about half of it.

Teaching: each letter twice, as its ink window and as its labelled ink blurred by SHAPE_BLUR pixels (the clean shape
of the same letter and its neighbours); every window drawn afresh each step with a random affine (scale, turn, shear,
shift), stroke thickening or thinning, blur, contrast and noise; classes drawn evenly. ENSEMBLE networks from
different seeds, their probabilities averaged, each over the plain view and changed views of the window.

Holdout cases (taught on 415 development letters; 122 holdout letters at AUC >= 0.9 / all 162):
    one network                      top-1 0.76-0.80 / 0.69-0.72   top-3 0.92 / 0.86
    + label shapes                   top-1 0.83 / 0.72             top-3 0.93 / 0.89
    + label shapes, three networks   top-1 0.85 / 0.74             top-3 0.93 / 0.89
The template fit of case_letters.py scored top-1 0.55 on the atlas's confirmed letters.

./run.sh ink.training.letter_classifier holdout      taught on the development cases, tested on the holdout cases
./run.sh ink.training.letter_classifier scrolls      each scroll tested with every letter of its scroll left out
./run.sh ink.training.letter_classifier final        taught on every letter; weights to weights/trained/letter-classifier.pth
Each test reports accuracy on the published-map letters at AUC >= 0.9 and on all of them.

Reading with the taught weights: cut() a window of any ink map round a letter, then Reader()(windows).
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ni

from ink.datasets.eye_letters import GREEK, PX, SIDE
from ink.datasets.eye_letters import OUT as LETTERS
from storage.data_roots import TRAINED_WEIGHTS

VIEW, IN = 2.0, 64
EPOCHS, BATCH, LR = 120, 64, 2e-3
SHAPE_BLUR, ENSEMBLE = 2.0, 3
GOOD_AUC = 0.9
CLASSES = len(GREEK)
WEIGHTS = TRAINED_WEIGHTS / "letter-classifier.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load():
    """Ink windows and label-shape windows (n, PX, PX) as 0..1, and each letter's row."""
    d = np.load(LETTERS / "letters.npz")
    rows = json.loads((LETTERS / "letters.json").read_text())
    shapes = np.stack([ni.gaussian_filter(m.astype(np.float32), SHAPE_BLUR) for m in d["any"]])
    return d["ink"].astype(np.float32) / 255, shapes, rows


class Net(nn.Module):
    def __init__(self, classes=CLASSES, width=32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            )  # fmt: skip

        w = width
        self.features = nn.Sequential(
            block(1, w), nn.MaxPool2d(2), block(w, 2 * w), nn.MaxPool2d(2), block(2 * w, 4 * w), nn.MaxPool2d(2),
            block(4 * w, 8 * w),
        )  # fmt: skip
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(8 * w, classes))

    def forward(self, x):
        return self.head(self.features(x))


def standardised(x):
    """Each window to zero mean and unit spread, so maps of any contrast look alike."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
    return (x - m) / s


def views(x, augment, generator=None):
    """The middle VIEW/SIDE of each stored window at IN pixels, with the teaching changes when augment is set."""
    n = x.shape[0]
    k = VIEW / SIDE
    theta = torch.zeros(n, 2, 3, device=x.device)
    if augment:

        def u(lo, hi):
            return torch.empty(n, device=x.device).uniform_(lo, hi, generator=generator)

        scale = torch.exp(u(np.log(0.8), np.log(1.25)))
        turn = u(-0.15, 0.15)
        shear = u(-0.2, 0.2)
        aspect = torch.exp(u(np.log(0.85), np.log(1.15)))
        sx, sy = k * scale * aspect.sqrt(), k * scale / aspect.sqrt()
        theta[:, 0, 0] = sx * torch.cos(turn)
        theta[:, 0, 1] = -sy * torch.sin(turn) + sx * shear
        theta[:, 1, 0] = sx * torch.sin(turn)
        theta[:, 1, 1] = sy * torch.cos(turn)
        theta[:, 0, 2] = u(-0.12, 0.12) * k
        theta[:, 1, 2] = u(-0.12, 0.12) * k
    else:
        theta[:, 0, 0] = theta[:, 1, 1] = k
    grid = F.affine_grid(theta, (n, 1, IN, IN), align_corners=False)
    v = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if augment:
        # stroke width: grey dilation or erosion by one pixel for a third of the windows each
        r = torch.rand(n, 1, 1, 1, device=x.device, generator=generator)
        wide = F.max_pool2d(v, 3, 1, 1)
        thin = -F.max_pool2d(-v, 3, 1, 1)
        v = torch.where(r < 1 / 3, wide, torch.where(r < 2 / 3, thin, v))
        # blur
        sigma = torch.rand(n, device=x.device, generator=generator) * 1.5
        size = torch.arange(-3, 4, device=x.device, dtype=torch.float32)
        kernel = torch.exp(-(size[None] ** 2) / (2 * sigma[:, None].clamp_min(0.05) ** 2))
        kernel = kernel / kernel.sum(1, keepdim=True)
        v = F.conv2d(v.transpose(0, 1), kernel[:, None, None, :], padding=(0, 3), groups=n)
        v = F.conv2d(v, kernel[:, None, :, None], padding=(3, 0), groups=n).transpose(0, 1)
        # contrast, gamma and noise
        gamma = torch.exp(torch.empty(n, 1, 1, 1, device=x.device).uniform_(-0.4, 0.4, generator=generator))
        v = v.clamp(0, 1) ** gamma
        v = (
            v
            + torch.randn(v.shape, device=x.device, generator=generator)
            * torch.rand(n, 1, 1, 1, device=x.device, generator=generator)
            * 0.08
        )
    return standardised(v)


def teach(x, y, epochs=EPOCHS, seed=0, log=None):
    """A network taught on windows x (n, 1, H, W) with classes y."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    net = Net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=5e-4)
    steps = epochs * max(1, len(y) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    counts = np.bincount(y, minlength=len(GREEK)).astype(np.float64)
    weight = torch.tensor(1 / np.maximum(counts[y], 1), dtype=torch.float64)
    xs, ys = x.to(DEVICE), torch.tensor(y, device=DEVICE)
    net.train()
    for step in range(steps):
        idx = torch.multinomial(weight, BATCH, replacement=True).to(DEVICE)
        logits = net(views(xs[idx], True, g))
        loss = F.cross_entropy(logits, ys[idx], label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 200 == 0:
            print(f"  step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def probabilities(nets, x, passes=8, seed=1):
    """Class probabilities of windows x (n, 1, PX, PX), the mean over the networks and, for each, over the plain view
    and passes-1 changed ones."""
    xs = x.to(DEVICE)
    total = 0
    for net in nets:
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        p = F.softmax(net(views(xs, False)), 1)
        for _ in range(passes - 1):
            p = p + F.softmax(net(views(xs, True, g)), 1)
        total = total + p / passes
    return (total / len(nets)).cpu().numpy()


def report(name, p, rows):
    y = np.array([r["class"] for r in rows])
    good = np.array([r["source"] == "published" and (r["auc"] or 0) >= GOOD_AUC for r in rows])
    published = np.array([r["source"] == "published" for r in rows])
    top = np.argsort(-p, 1)
    out = {"name": name}
    for label, keep in (("auc>=0.9", good), ("published", published)):
        if keep.any():
            out[label] = {
                "n": int(keep.sum()),
                "top1": round(float((top[keep, 0] == y[keep]).mean()), 3),
                "top3": round(float((top[keep, :3] == y[keep, None]).any(1).mean()), 3),
            }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def tensor(images, idx):
    return torch.from_numpy(images[idx])[:, None]


def taught(images, shapes, rows, idx, epochs=EPOCHS, log=False):
    """ENSEMBLE networks taught on the letters idx, each letter as its ink window and its label shape."""
    x = torch.cat([tensor(images, idx), tensor(shapes, idx)])
    y = np.array([rows[i]["class"] for i in idx] * 2)
    return [teach(x, y, int(epochs), seed=seed, log=log) for seed in range(ENSEMBLE)]


def holdout(epochs=EPOCHS):
    images, shapes, rows = load()
    train = [i for i, r in enumerate(rows) if r["partition"] != "holdout"]
    test = [i for i, r in enumerate(rows) if r["partition"] == "holdout"]
    start = time.time()
    nets = taught(images, shapes, rows, train, epochs, log=True)
    p = probabilities(nets, tensor(images, test))
    result = report("holdout cases", p, [rows[i] for i in test])
    print("seconds", round(time.time() - start), flush=True)
    return result, p, test


def scrolls(epochs=EPOCHS):
    images, shapes, rows = load()
    names = sorted({r["sample"] for r in rows if r["source"] == "published"})
    p_all = np.zeros((len(rows), len(GREEK)))
    tested, found = [], {}
    for name in names:
        test = [i for i, r in enumerate(rows) if r["sample"] == name]
        train = [i for i, r in enumerate(rows) if r["sample"] != name]
        p_all[test] = probabilities(taught(images, shapes, rows, train, epochs), tensor(images, test))
        found[name] = report(f"{name} left out", p_all[test], [rows[i] for i in test])
        tested += test
    found["every scroll"] = report("every scroll left out", p_all[tested], [rows[i] for i in tested])
    LETTERS.joinpath("scrolls-left-out.json").write_text(json.dumps(found, ensure_ascii=False, indent=1))
    np.save(LETTERS / "scrolls-left-out.npy", p_all)
    return found, p_all


def final(epochs=EPOCHS):
    images, shapes, rows = load()
    nets = taught(images, shapes, rows, list(range(len(rows))), epochs, log=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"states": [n.state_dict() for n in nets], "view": VIEW, "side": SIDE, "px": PX, "in": IN, "letters": GREEK},
        WEIGHTS,
    )
    print("weights", WEIGHTS)


def cut(v, cy, cx, reference_px, px=PX):
    """A window SIDE reference heights on a side round (cy, cx) of an ink map v (ink bright, 0..1), at px pixels, zeros
    outside the map: the window the network was taught on, a letter of the row's height filling a third of its side."""
    half = max(int(round(SIDE * reference_px / 2)), 1)
    part = np.zeros((2 * half, 2 * half), np.float32)
    a0, a1 = max(cy - half, 0), min(cy + half, v.shape[0])
    b0, b1 = max(cx - half, 0), min(cx + half, v.shape[1])
    if a1 > a0 and b1 > b0:
        part[a0 - (cy - half) : a1 - (cy - half), b0 - (cx - half) : b1 - (cx - half)] = v[a0:a1, b0:b1]
    part = (np.clip(np.nan_to_num(part), 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(part).resize((px, px), Image.LANCZOS), np.float32) / 255


class Reader:
    """The taught networks: windows (n, PX, PX) from cut() to (n, 24) probabilities in the order of GREEK."""

    def __init__(self, path=WEIGHTS):
        saved = torch.load(path, map_location="cpu")
        self.nets = []
        for state in saved["states"]:
            net = Net()
            net.load_state_dict(state)
            self.nets.append(net.to(DEVICE).eval())

    def __call__(self, windows, passes=8):
        out = []
        for k in range(0, len(windows), 256):
            out.append(
                probabilities(
                    self.nets, torch.from_numpy(np.asarray(windows[k : k + 256], np.float32))[:, None], passes
                )
            )
        return np.concatenate(out) if out else np.zeros((0, len(GREEK)))


if __name__ == "__main__":
    {"holdout": holdout, "scrolls": scrolls, "final": final}[sys.argv[1]](*sys.argv[2:])
