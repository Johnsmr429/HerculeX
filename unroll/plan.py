import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import fetch_box


def papyrus_mask(m7, depth):
    mask = np.zeros_like(m7)
    for z in range(m7.shape[0]):
        s = m7[z]
        if s.sum() < 50:
            continue
        f = ndi.binary_fill_holes(ndi.binary_closing(s, iterations=2))
        lab, n = ndi.label(f)
        if n == 0:
            continue
        sizes = ndi.sum(f, lab, range(1, n + 1))
        f = lab == (1 + int(np.argmax(sizes)))
        mask[z] = ndi.distance_transform_edt(f) >= depth
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m7-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=13.0)
    ap.add_argument("--z-anchor", type=int, default=0)
    ap.add_argument("--unit-z", type=int, default=768)
    ap.add_argument("--step", type=int, default=576)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    level4 = args.m7_url.rstrip("/").rsplit("/", 1)[0] + "/4"
    path = os.path.join(args.out, "m7_L4.npy")
    if not os.path.exists(path):
        shape = json.load(urllib.request.urlopen(level4 + "/.zarray"))["shape"]
        fetch_box(level4, ((0, shape[0]), (0, shape[1]), (0, shape[2])), path, workers=16)
    m7 = np.load(path) > 0
    mask = papyrus_mask(m7, args.depth)
    np.save(os.path.join(args.out, "papyrus_mask_L4.npy"), mask)
    np.savez_compressed(
        os.path.join(args.out, "papyrus_mask_L4.npz"),
        mask=np.packbits(mask, axis=-1),
        shape=np.array(mask.shape),
    )
    mask.astype(np.uint8).tofile(os.path.join(args.out, "papyrus_mask_L4.u8"))
    full = json.load(urllib.request.urlopen(args.m7_url.rstrip("/") + "/.zarray"))[
        "shape"
    ]
    full_z = int(full[0])
    units = []
    k = 0
    while args.z_anchor + args.step * k > 0:
        k -= 1
    while True:
        z0 = max(0, args.z_anchor + args.step * k)
        z1 = min(full_z, args.z_anchor + args.step * k + args.unit_z)
        if z0 >= full_z:
            break
        sl = mask[z0 // 16 : (z1 + 15) // 16]
        ys = np.nonzero(sl.any(axis=(0, 2)))[0]
        xs = np.nonzero(sl.any(axis=(0, 1)))[0]
        if len(ys) and len(xs):
            y0 = max(0, int(ys.min()) * 16 - 32)
            y1 = min(int(full[1]), int(ys.max()) * 16 + 48)
            x0 = max(0, int(xs.min()) * 16 - 32)
            x1 = min(int(full[2]), int(xs.max()) * 16 + 48)
            units.append(
                {
                    "k": k,
                    "name": f"unit_z{z0:05d}_{z1:05d}",
                    "z0": z0,
                    "z1": z1,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                }
            )
        if z1 >= full_z:
            break
        k += 1
    Path(os.path.join(args.out, "units_plan.json")).write_text(
        json.dumps(units, indent=1)
    )
    print(
        f"{len(units)} units, mask keeps {int((m7 & mask).sum())} of {int(m7.sum())} m7 voxels at level 4"
    )


if __name__ == "__main__":
    main()
