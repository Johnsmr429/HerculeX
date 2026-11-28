import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERIOD = 6.0


def layers(ct, sh, u):
    g = np.where(ct > 0, (ct.astype(np.float32) - 1) / 254, 0.0)
    p = np.where(sh > 0, (sh.astype(np.float32) - 1) / 254, np.nan)
    base = np.repeat(g[..., None], 3, -1)
    on = (p > 0.5)[..., None]
    col = plt.get_cmap("hsv")(np.mod(np.nan_to_num(u.astype(np.float32)), PERIOD) / PERIOD)[..., :3]
    wind = np.where(on, 0.15 * base + 0.85 * col, 0.45 * base)
    return g, p, wind


def panel(a, img, title, **kw):
    a.imshow(img, interpolation="nearest", **kw)
    a.set_title(title, fontsize=9)
    a.set_xticks([])
    a.set_yticks([])


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for path in sys.argv[2:]:
        d = np.load(path)
        ct, sh, u = d["ct"], d["sheet"], d["u"]
        vox = float(d["voxel"])
        name = os.path.basename(path)[:-4]
        z = ct.shape[0] // 2
        y = ct.shape[1] // 2
        mm = [s * vox / 1000 for s in ct.shape]
        fig = plt.figure(figsize=(17, 9.2), facecolor="white")
        gs = fig.add_gridspec(2, 3, height_ratios=[ct.shape[1], ct.shape[0] * 1.0 + 1e-3], hspace=0.12, wspace=0.03)
        g, p, w = layers(ct[z], sh[z], u[z])
        panel(fig.add_subplot(gs[0, 0]), g, f"CT, slice z+{z}", cmap="gray")
        panel(fig.add_subplot(gs[0, 1]), p, "sheet probability", cmap="magma", vmin=0, vmax=1)
        panel(fig.add_subplot(gs[0, 2]), w, f"winding u on sheet (colour repeats every {PERIOD:g} turns)")
        g, p, w = layers(ct[:, y], sh[:, y], u[:, y])
        panel(fig.add_subplot(gs[1, 0]), g, f"side view at y+{y} (z down)", cmap="gray")
        panel(fig.add_subplot(gs[1, 1]), p, "sheet probability", cmap="magma", vmin=0, vmax=1)
        panel(fig.add_subplot(gs[1, 2]), w, "winding u on sheet")
        o = d["origin"].tolist()
        fig.suptitle(f"{name}: held-out box at scan voxel z,y,x = {o}, {ct.shape[0]}x{ct.shape[1]}x{ct.shape[2]} voxels "
                     f"({mm[0]:.1f} x {mm[1]:.1f} x {mm[2]:.1f} mm at {vox:g} um), curl {int(d['curl'])}", fontsize=11)
        png = f"{out}/{name}.png"
        fig.savefig(png, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(png)


main()
