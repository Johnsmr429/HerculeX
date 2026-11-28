import json
import math
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    out = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(re.sub(r"\bnan\b", "NaN", line)))
    except OSError:
        pass
    return out


def main():
    root, png = sys.argv[1], sys.argv[2]
    stages = [s for s in ("pre", "mid") if load(f"{root}/{s}/train.jsonl")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), facecolor="white")
    off = 0
    for s in stages:
        t = [d for d in load(f"{root}/{s}/train.jsonl") if "losses" in d]
        v = load(f"{root}/{s}/val.jsonl")
        x = [d["step"] + off for d in t]
        for k in t[-1]["losses"]:
            y = [d["losses"].get(k, math.nan) for d in t]
            if any(a > 0 for a in y if not math.isnan(a)):
                ax[0, 0].plot(x, y, lw=0.8, label=f"{s} {k}")
        ax[0, 1].plot(x, [d["gradient_norm"] for d in t], lw=0.8, label=s)
        ax[0, 2].plot(x, [d["learning_rate"] for d in t], lw=0.8, label=s)
        ax[1, 2].plot(x, [d["steps_per_second"] for d in t], lw=0.8, label=s)
        for d in v:
            d["_x"] = d["step"] + off
        for key, a in (("recon_mse", ax[1, 0]), ("sheet_dice", ax[1, 1]), ("next_turn_accuracy", ax[1, 1]),
                       ("relative_accuracy", ax[1, 1]), ("lines_without_jumps", ax[1, 1])):
            pts = [(d["_x"], d["all"][key]) for d in v if key in d.get("all", {}) and not math.isnan(d["all"][key])]
            if pts:
                a.plot(*zip(*pts), "o-", ms=3, label=f"{s} {key}")
        off += t[-1]["step"]
    titles = ["training losses", "gradient norm", "learning rate", "held-out recon MSE", "held-out labelled measures",
              "steps per second"]
    for a, ti in zip(ax.flat, titles):
        a.set_title(ti, fontsize=10)
        a.set_xlabel("step: pre 0-50k, then mid (mid step + 50k)", fontsize=8)
        a.grid(alpha=0.3)
        if a.lines:
            a.legend(fontsize=7)
    ax[0, 0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(png, dpi=110)


main()
