import glob
import math
import os
import sys

import numpy as np

D, H, W = 64, 128, 128
V = D * H * W
PT = 24


def read(path):
    b = open(path, "rb").read()
    src = int(np.frombuffer(b, "<i4", 4)[3])
    o = 16
    ct, sh, u = (np.frombuffer(b, "<f4", V, o + k * 4 * V).reshape(D, H, W) for k in range(3))
    o += 12 * V
    n = int(np.frombuffer(b, "<i4", 1, o)[0])
    o += 4 + n * PT + V
    tail = np.frombuffer(b, "<f4", 6, o)
    ctr = np.frombuffer(b, "<f4", 128, o + 24).reshape(D, 2)
    return dict(src=src, ct=ct, sh=sh, u=u, origin=tail[:3].astype(int), voxel=float(tail[3]),
                curl=int(round(tail[4])), ref=float(tail[5]), ctr=ctr)


def groups(slabs):
    left = set(range(len(slabs)))
    out = []
    while left:
        seed = left.pop()
        g, todo = [seed], [seed]
        while todo:
            a = slabs[todo.pop()]["origin"]
            near = [j for j in left if np.all(np.abs(slabs[j]["origin"] - a) <= (D, H, W))
                    and np.sum(np.abs(slabs[j]["origin"] - a) > 0) == 1]
            for j in near:
                left.discard(j)
                g.append(j)
                todo.append(j)
        out.append(g)
    return out


def wrap(a):
    return a - 2 * math.pi * np.floor((a + math.pi) / (2 * math.pi))


def main():
    dump, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    slabs = [read(p) for p in sorted(glob.glob(f"{dump}/slab-*.bin"))]
    for gi, g in enumerate(sorted(groups(slabs), key=len, reverse=True)):
        o0 = np.min([slabs[i]["origin"] for i in g], 0)
        o1 = np.max([slabs[i]["origin"] for i in g], 0) + (D, H, W)
        shape = tuple(int(x) for x in o1 - o0)
        ct = np.full(shape, np.nan, np.float32)
        sh = np.full(shape, np.nan, np.float32)
        u = np.full(shape, np.nan, np.float32)
        mid = (o0 + o1) / 2
        best = min(g, key=lambda i: np.sum((slabs[i]["origin"] + (D / 2, H / 2, W / 2) - mid) ** 2))
        ref_box = slabs[best]["ref"]
        for i in g:
            s = slabs[i]
            z0, y0, x0 = (s["origin"] - o0).astype(int)
            zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
            cy = s["ctr"][:, 0][:, None, None]
            cx = s["ctr"][:, 1][:, None, None]
            th = np.arctan2(yy - cy, xx - cx)
            a_slab = s["ref"] + wrap(th - s["ref"])
            a_box = ref_box + wrap(th - ref_box)
            k = np.round((a_slab - a_box) / (2 * math.pi))
            ct[z0:z0 + D, y0:y0 + H, x0:x0 + W] = s["ct"]
            sh[z0:z0 + D, y0:y0 + H, x0:x0 + W] = s["sh"]
            u[z0:z0 + D, y0:y0 + H, x0:x0 + W] = s["u"] + s["curl"] * k
        lo, hi = np.nanpercentile(ct, [0.5, 99.5])
        ct8 = np.where(np.isnan(ct), 0, np.clip((ct - lo) / (hi - lo + 1e-6) * 254 + 1, 1, 255)).astype(np.uint8)
        sh8 = np.where(np.isnan(sh), 0, np.round(sh * 254) + 1).astype(np.uint8)
        path = f"{out}/{name}-box{gi}.npz"
        np.savez_compressed(path, ct=ct8, sheet=sh8, u=u.astype(np.float16), origin=o0, voxel=slabs[g[0]]["voxel"],
                            curl=slabs[g[0]]["curl"], slabs=len(g))
        print(path, shape, "slabs", len(g), "origin", o0.tolist())


main()
