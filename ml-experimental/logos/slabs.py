import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {0: "unlabelled CT (held-out boxes)", 1: "PHercParis4 instance cubes", 2: "fibre cubes (Dataset001)",
         3: "PHercParis4 winding points", 4: "PHercParis4 eval fibres", 5: "PHerc0332 slice lines",
         6: "PHerc0332 manual fibres", 7: "Kaggle surface volumes"}
PT = np.dtype([("z", "<f4"), ("y", "<f4"), ("x", "<f4"), ("set", "<i4"), ("kind", "<i4"), ("w", "<f4")])


def read(path):
    b = open(path, "rb").read()
    d, h, w, src = np.frombuffer(b, "<i4", 4)
    v = int(d * h * w)
    o = 16
    ct, sh, u = (np.frombuffer(b, "<f4", v, o + k * 4 * v).reshape(d, h, w) for k in range(3))
    o += 12 * v
    n = int(np.frombuffer(b, "<i4", 1, o)[0])
    o += 4
    pts = np.frombuffer(b, PT, n, o)
    o += n * PT.itemsize
    lab = np.frombuffer(b, "u1", v, o).reshape(d, h, w) if len(b) >= o + v else None
    return int(src), ct, sh, u, pts, lab


def show(a, img, title, **kw):
    a.imshow(img, interpolation="nearest", **kw)
    a.set_title(title, fontsize=8)
    a.set_xticks([])
    a.set_yticks([])


def winding(ct, sh, u):
    lo, hi = np.percentile(ct, [1, 99])
    g = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.repeat(g[..., None] * 0.55, 3, -1)
    c = plt.get_cmap("hsv")(np.mod(u, 6.0) / 6.0)[..., :3]
    m = (sh > 0.5)[..., None]
    return np.where(m, 0.15 * rgb + 0.85 * c, rgb)


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    by = {}
    for p in sorted(glob.glob(f"{src_dir}/slab-*.bin")):
        r = read(p)
        by.setdefault(r[0], []).append((os.path.basename(p), r))
    for src, rows in sorted(by.items()):
        fig, ax = plt.subplots(len(rows), 6, figsize=(18, 3.1 * len(rows)), facecolor="white", squeeze=False)
        for i, (name, (_, ct, sh, u, pts, lab)) in enumerate(rows):
            z = ct.shape[0] // 2
            if len(pts):
                z = int(np.clip(np.round(np.median(pts["z"])), 0, ct.shape[0] - 1))
            y = ct.shape[1] // 2
            show(ax[i, 0], ct[z], f"{name} CT z={z}", cmap="gray")
            show(ax[i, 1], sh[z], "sheet probability", cmap="magma", vmin=0, vmax=1)
            if lab is not None and (lab != 255).any():
                show(ax[i, 2], np.where(lab[z] == 255, 0.5, lab[z]), "label (grey = unknown)", cmap="gray", vmin=0, vmax=1)
            else:
                show(ax[i, 2], ct[z] * 0, "no voxel label", cmap="gray")
            show(ax[i, 3], winding(ct[z], sh[z], u[z]), "winding u on sheet (hue repeats every 6 turns)")
            near = pts[np.abs(pts["z"] - z) <= 2] if len(pts) else pts
            if len(near):
                ax[i, 3].scatter(near["x"], near["y"], s=6, c="white", lw=0)
            show(ax[i, 4], ct[:, y, :], f"side view CT y={y}", cmap="gray", aspect="auto")
            show(ax[i, 5], winding(ct[:, y, :], sh[:, y, :], u[:, y, :]), "side view winding", aspect="auto")
        fig.suptitle(NAMES.get(src, f"source {src}"), fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/source{src}.png", dpi=90)
        plt.close(fig)
        print(f"{out_dir}/source{src}.png", len(rows))


main()
