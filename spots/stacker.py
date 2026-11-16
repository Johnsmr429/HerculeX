"""A stacked classifier: one pixel's reads by every reader at every depth shift, to ink or blank.

./run.sh spots.stacker [TEST_SCROLL]        default PHerc0841; trains on the other scrolls' control features

Features are those spots.controls keeps: per labelled pixel, each reader's probability at each of nine window shifts
round the production window, on a surface (original or m7). The classifier sees their logits, standardised; it is
logistic regression and a two-layer network (64 hidden, dropout), both with balanced classes, taught on every scroll
but the test scroll and scored there by rank AUC. Beside it, each reader alone at shift 0 and as its five-depth mean,
on the same pixels. The network and its standardisation are kept for reading a target.

The test scroll should be one the readers were not taught on: PHerc0841 is held out by the student, stroke-core,
own-reads-stroke-core and thin-marks; own-mesh was taught on every labelled scroll and is reported but not trusted
there.
"""

import json
import sys

import numpy as np
import torch

from calibration.ink.ink_yield import rank_auc
from storage.data_roots import SPOT_CANDIDATES

FEATURES = SPOT_CANDIDATES / "controls" / "features"
OUT = SPOT_CANDIDATES / "controls" / "stacker"
EPOCHS, BATCH, SEED = 12, 8192, 20261112


def loaded(surface):
    """Per scroll: features (n, readers, shifts) as logits, and ink labels."""
    by_scroll, names, shifts = {}, None, None
    for path in sorted(FEATURES.glob(f"*/*-{surface}.npz")):
        data = np.load(path)
        names, shifts = list(data["readers"]), list(data["shifts"])
        scroll = path.name.split("-")[0]
        f = np.clip(data["features"].astype(np.float32), 1e-4, 1 - 1e-4)
        by_scroll.setdefault(scroll, ([], []))
        by_scroll[scroll][0].append(np.log(f / (1 - f)))
        by_scroll[scroll][1].append(data["ink"].astype(bool))
    return {s: (np.concatenate(f), np.concatenate(y)) for s, (f, y) in by_scroll.items()}, names, shifts


class Net(torch.nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, x):
        return self.body(x).squeeze(-1)


def taught(x, y, model):
    torch.manual_seed(SEED)
    x, y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    weight = torch.tensor(float((1 - y.mean()) / max(y.mean(), 1e-6)))
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        order = torch.randperm(len(y))
        model.train()
        for k in range(0, len(y), BATCH):
            idx = order[k : k + BATCH]
            opt.zero_grad()
            loss(model(x[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    return model


def scores(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def run(test="PHerc0841"):
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for surface in ("original", "m7"):
        data, names, shifts = loaded(surface)
        if test not in data or len(data) < 2:
            print(surface, "not enough scrolls yet:", sorted(data), flush=True)
            continue
        train = [s for s in data if s != test]
        xtr = np.concatenate([data[s][0] for s in train]).reshape(-1, len(names) * len(shifts))
        ytr = np.concatenate([data[s][1] for s in train])
        xte, yte = data[test][0], data[test][1]
        mean, std = xtr.mean(0), xtr.std(0) + 1e-6
        ztr, zte = (xtr - mean) / std, (xte.reshape(len(yte), -1) - mean) / std
        row = {
            "train": train,
            "test": test,
            "train_pixels": int(len(ytr)),
            "test_pixels": int(len(yte)),
            "test_ink": round(float(yte.mean()), 3),
        }
        zero, band = shifts.index(0), [shifts.index(s) for s in (-2, -1, 0, 1, 2)]
        for k, name in enumerate(names):
            row[f"{name} shift 0"] = round(float(rank_auc(xte[yte, k, zero], xte[~yte, k, zero])), 4)
            m = xte[:, k][:, band].mean(1)
            row[f"{name} 5 depths"] = round(float(rank_auc(m[yte], m[~yte])), 4)
        mean_all = xte.mean(axis=(1, 2))
        row["mean of all readers and depths"] = round(float(rank_auc(mean_all[yte], mean_all[~yte])), 4)
        logistic = taught(
            ztr.astype(np.float32), ytr, torch.nn.Sequential(torch.nn.Linear(ztr.shape[1], 1), torch.nn.Flatten(0))
        )
        s = scores(logistic, zte.astype(np.float32))
        row["stacker logistic"] = round(float(rank_auc(s[yte], s[~yte])), 4)
        net = taught(ztr.astype(np.float32), ytr, Net(ztr.shape[1]))
        s = scores(net, zte.astype(np.float32))
        row["stacker network"] = round(float(rank_auc(s[yte], s[~yte])), 4)
        torch.save(
            {
                "state": net.state_dict(),
                "mean": mean,
                "std": std,
                "readers": names,
                "shifts": shifts,
                "surface": surface,
                "test": test,
            },
            OUT / f"net-{surface}-without-{test}.pt",
        )
        report[surface] = row
        print(surface, json.dumps(row, indent=1), flush=True)
    (OUT / f"report-without-{test}.json").write_text(json.dumps(report, indent=1))
    return report


def surfaces():
    """Per scroll and surface, each reader's AUC at the production window, at the scroll's best fixed shift, and as the
    mean of all nine shifts, on pairs both surfaces scored (pixels pooled within a scroll)."""
    pairs = {}
    for path in FEATURES.glob("*/*.npz"):
        name, surface = path.stem.rsplit("-", 1)
        pairs.setdefault(name, {})[surface] = path
    both = {n: s for n, s in pairs.items() if {"original", "m7"} <= set(s)}
    report = {}
    for scroll in sorted({n.split("-")[0] for n in both}):
        row = {"pairs": sum(n.startswith(scroll) for n in both)}
        for surface in ("original", "m7"):
            data = [np.load(both[n][surface]) for n in both if n.startswith(scroll)]
            f = np.concatenate([d["features"].astype(np.float32) for d in data])
            y = np.concatenate([d["ink"].astype(bool) for d in data])
            names, shifts = list(data[0]["readers"]), list(data[0]["shifts"])
            for k, reader in enumerate(names):
                per_shift = [float(rank_auc(f[y, k, s], f[~y, k, s])) for s in range(len(shifts))]
                mean = f[:, k].mean(1)
                row[f"{surface} {reader}"] = {
                    "window": round(per_shift[shifts.index(0)], 4),
                    "best": round(max(per_shift), 4),
                    "best_shift": int(shifts[int(np.argmax(per_shift))]),
                    "mean_9": round(float(rank_auc(mean[y], mean[~y])), 4),
                }
        report[scroll] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "surfaces.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    if sys.argv[1:2] == ["surfaces"]:
        print(json.dumps(surfaces(), indent=1))
    else:
        run(*sys.argv[1:2])
