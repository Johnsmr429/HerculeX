"""The scorer checked on surfaces whose answer is known, made from a package's own meshes.

  oracle-w      the meshes with their w.tif                       everything right, distance 0
  oracle-int    the meshes with only an integer winding          the same, through the fitted cut angle
  oracle-local  the meshes with no winding (local mode)           the same, each surface its own offset
  shift80       the meshes moved 80 µm towards the axis           recall falls, distance rises
  offby1        odd wraps labelled one winding high               about half the truth wrong
  switch        one surface on one wrap, then on the next         half of it wrong and a wrong join at the seam

Each case is scored on the dev zone and held to the bounds below; the command exits 1 if any bound fails.
Surfaces are written under the data root's scratch/score-check/NAME and removed afterwards unless --keep.
"""

import argparse
import json
import shutil
import sys

import numpy as np
import tifffile

from calibration.bench import package as packages
from calibration.bench import score as scorer
from storage.data_roots import SCRATCH

BOUNDS = {
    "oracle-w": {"truth_recall": (0.999, 1), "winding.right_share": (0.999, 1), "area_cm2.wrong": (0, 0.01)},
    "oracle-int": {"truth_recall": (0.999, 1), "winding.right_share": (0.999, 1), "area_cm2.wrong": (0, 0.01)},
    "oracle-local": {"truth_recall": (0.999, 1), "winding.right_share": (0.999, 1), "area_cm2.wrong": (0, 0.01)},
    "shift80": {"truth_recall": (0, 0.6)},
    "offby1": {"winding.right_share": (0.3, 0.75)},
    "switch": {"winding.right_share": (0.2, 0.8), "wrong_join_share": (1e-4, 1)},
}


def value(out, dotted):
    for part in dotted.split("."):
        out = out[part]
    return out


def write(folder, xyz, valid, meta, w=None):
    folder.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate("xyz"):
        tifffile.imwrite(folder / f"{a}.tif", np.where(valid, xyz[..., i], -1).astype(np.float32))
    if w is not None:
        tifffile.imwrite(folder / "w.tif", w.astype(np.float32))
    (folder / "meta.json").write_text(json.dumps(meta))


def make(name, root):
    meshes = sorted((packages.PACKAGES / name / "meshes").glob("*.tifxyz"))
    knots = np.array(json.loads((packages.PACKAGES / name / "axis.json").read_text())["knots_z_x_y"])
    voxel_um = float(np.load(packages.PACKAGES / name / "truth.npz")["voxel_um"])
    grids = []
    for m in meshes:
        meta = json.loads((m / "meta.json").read_text())
        xyz = np.stack([tifffile.imread(m / f"{a}.tif") for a in "xyz"], axis=-1).astype(np.float64)
        grids.append((m.name, xyz, np.all(xyz > 0, axis=-1), tifffile.imread(m / "w.tif").astype(np.float64), meta))
    for label, xyz, valid, w, meta in grids:
        wrap = meta["wrap_name"]
        write(root / "oracle-w" / label, xyz, valid, {"winding": wrap}, w)
        write(root / "oracle-int" / label, xyz, valid, {"winding": wrap})
        write(root / "oracle-local" / label, xyz, valid, {})
        write(root / "offby1" / label, xyz, valid, {"winding": wrap}, w + (1 if wrap % 2 else 0))
        n = packages.grid_normals(xyz, valid)
        cx = np.interp(xyz[..., 2], knots[:, 0], knots[:, 1])
        cy = np.interp(xyz[..., 2], knots[:, 0], knots[:, 2])
        n[((cx - xyz[..., 0]) * n[..., 0] + (cy - xyz[..., 1]) * n[..., 1]) < 0] *= -1
        moved = valid & np.all(np.isfinite(n), axis=-1)
        write(root / "shift80" / label, xyz + np.nan_to_num(n) * 80 / voxel_um, moved, {"winding": wrap}, w)
    by_size = sorted(grids, key=lambda g: -g[2].sum())
    (label_a, a, va, wa, ma) = by_size[0]
    partner = [g for g in grids if g[4]["wrap_name"] == ma["wrap_name"] + 1] or [by_size[1]]
    _, b, vb, _, _ = partner[0]
    rows, cols = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    xyz, valid, w = a[:rows, :cols].copy(), va[:rows, :cols].copy(), wa[:rows, :cols].copy()
    left = np.cumsum(va[:rows, :cols].sum(axis=0))
    right = np.cumsum(vb[:rows, :cols].sum(axis=0)[::-1])[::-1]
    half = int(np.argmin(np.abs(left - right)))
    xyz[:, half:], valid[:, half:] = b[:rows, half:cols], vb[:rows, half:cols]
    write(root / "switch" / label_a, xyz, valid, {"winding": ma["wrap_name"]}, w)


def run(name, keep=False):
    root = SCRATCH / "score-check" / name
    shutil.rmtree(root, ignore_errors=True)
    make(name, root)
    failed, rows = [], []
    for case, bounds in BOUNDS.items():
        paths = sorted((root / case).glob("*.tifxyz"))
        out = scorer.score(name, paths, ("dev",), out_dir=root / f"score-{case}", label=f"check-{case}")
        row = {"case": case}
        for key, (low, high) in bounds.items():
            v = value(out, key)
            row[key] = v
            if v is None or not low <= v <= high:
                failed.append(f"{case}: {key} = {v} outside [{low}, {high}]")
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", default="PHerc0139-dev")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    failed = run(args.name, args.keep)
    print("all bounds hold" if not failed else "\n".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
