"""The original surface and its m7 snap combined: a gated surface, and reads of both pooled and fused per pixel.

gated(xyz, moved, limit)     each node moved towards its m7 place by a weight: whole where the snap moved it at most
                             `limit` voxels and as its neighbours moved, none past 2 x limit or where it moved unlike
                             them (by more than limit from their 3 x 3 median move), linear between; the weights
                             smoothed over 3 x 3 nodes so a gate's edge leaves no step. Nodes missing in either grid
                             keep the original's value. The snap keeps each node's (row, column), so the gated grid is
                             on the same grid and pixel frame as both.
following(xyz, other)        how closely a grid follows another surface: node distances to its nearest point
toward(xyz, other, limit)    xyz moved towards another surface on a different grid (a spiral winding, a level piece)
                             by the same gate, in xyz's frame
pooled(values, how)          a reader's values over depth shifts per pixel: "d0", "mean" over -2..+2, "max" over -4..+4
fused(maps)                  per pixel, the mean of maps each z-scored on its own pixels (the same pixels on each)

Scoring on the controls (spots/controls.py keeps every reader at every shift at the same pixels of each surface):

./run.sh spots.combine score BAND [BAND ...]     writes SPOT_CANDIDATES/controls/combine.json

For every pair with its surfaces kept at the same pixels: each reader's rank AUC of ink against blank under each
pooling, on each surface alone, on the original and m7 fused per pixel, and with the surface chosen per pair by the
readers' agreement (label-free). Reported as the median over pairs and the mean weighted by kept pixels.
"""

import json
import sys
import warnings

import numpy as np
from scipy import ndimage as ni

SHIFTS = tuple(range(-4, 5))
POOLINGS = {"d0": (0,), "mean": (-2, -1, 0, 1, 2), "max": SHIFTS}


def _median3(field):
    """3 x 3 median of a field with NaN for missing nodes, NaN where none of the nine is known."""
    padded = np.pad(field, 1, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=(-2, -1))


def gate_weights(xyz, moved, limit):
    """Per node, the share of its snap move kept (0..1), and where both grids hold a node."""
    both = (xyz[0] > 0) & (moved[0] > 0)
    shift = np.where(both[None], moved - xyz, np.nan).astype(np.float64)
    size = np.linalg.norm(np.nan_to_num(shift), axis=0)
    median = np.stack([_median3(s) for s in shift])
    unlike = np.linalg.norm(np.nan_to_num(shift - median), axis=0)
    weight = np.clip((2 * limit - size) / limit, 0, 1) * np.clip((limit - unlike) / (limit / 2), 0, 1)
    weight = np.where(both, weight, 0.0)
    known = ni.uniform_filter(both.astype(np.float64), 3)
    smooth = ni.uniform_filter(weight, 3) / np.maximum(known, 1e-9)
    return np.where(both, np.clip(smooth, 0, 1), 0.0), both


def gated(xyz, moved, limit):
    """(3, H, W) grid: each node moved towards its m7 place by its gate weight (see the module's note)."""
    weight, both = gate_weights(xyz, moved, limit)
    out = xyz.astype(np.float32).copy()
    out[:, both] = (xyz[:, both] + weight[both][None] * (moved[:, both] - xyz[:, both])).astype(np.float32)
    return out


def gate_report(xyz, moved, limit):
    """What a gate did: shares of nodes moved whole, partly and not at all, and the move kept."""
    weight, both = gate_weights(xyz, moved, limit)
    w = weight[both]
    size = np.linalg.norm((moved - xyz)[:, both], axis=0)
    return {
        "limit_vox": limit,
        "nodes": int(both.sum()),
        "whole": round(float((w >= 0.999).mean()), 4) if w.size else 0.0,
        "partly": round(float(((w > 0.001) & (w < 0.999)).mean()), 4) if w.size else 0.0,
        "kept_original": round(float((w <= 0.001).mean()), 4) if w.size else 0.0,
        "median_snap_move_vox": round(float(np.median(size)), 2) if w.size else None,
        "median_kept_move_vox": round(float(np.median(w * size)), 2) if w.size else None,
    }


def following(xyz, other):
    """How closely a grid follows another surface (any (3, H, W) grid, or (n, 3) points): each node's distance to
    the other's nearest point, in voxels."""
    from scipy.spatial import cKDTree

    pts = other[:, other[0] > 0].T if other.ndim == 3 else other
    mine = xyz[:, xyz[0] > 0].T
    if not len(mine) or not len(pts):
        return {"nodes": int(len(mine))}
    d, _ = cKDTree(pts).query(mine)
    return {
        "nodes": int(len(mine)),
        "median_vox": round(float(np.median(d)), 2),
        "p90_vox": round(float(np.percentile(d, 90)), 2),
        **{f"within_{k}_vox": round(float((d <= k).mean()), 4) for k in (2, 4, 8)},
    }


def toward(xyz, other, limit):
    """xyz's grid with each node moved towards the nearest point of another surface ((3, H, W) grid or (n, 3)
    points) by the gate of gated(): whole within `limit` voxels and alike its neighbours, none past 2 x limit. For
    combining two surfaces on different grids (a spiral winding, a level piece) in the frame of the first."""
    from scipy.spatial import cKDTree

    pts = other[:, other[0] > 0].T if other.ndim == 3 else other
    valid = xyz[0] > 0
    moved = xyz.astype(np.float32).copy()
    _, i = cKDTree(pts).query(xyz[:, valid].T)
    moved[:, valid] = pts[i].T
    return gated(xyz, moved, limit)


def zscored(values):
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def pooled(values, how):
    """values (pixels, shifts) over SHIFTS to (pixels,): one shift, the mean of a band, or the max over all."""
    idx = [SHIFTS.index(s) for s in POOLINGS[how]]
    part = values[:, idx]
    return part.max(1) if how == "max" else part.mean(1)


def fused(maps):
    """Per pixel, the mean of the maps each z-scored on its own pixels."""
    return np.mean([zscored(m) for m in maps], axis=0)


def agreement(features, how):
    """Mean pairwise correlation of the readers' pooled values: a label-free sign of a well-placed surface."""
    pooled_all = np.stack([pooled(features[:, r], how) for r in range(features.shape[1])])
    c = np.corrcoef(pooled_all)
    return float(c[np.triu_indices(len(c), 1)].mean())


def score(*bands):
    from calibration.ink.ink_yield import rank_auc
    from storage.data_roots import SPOT_CANDIDATES

    root = SPOT_CANDIDATES / "controls"
    results, per_pair = {}, []
    for band in bands:
        folder = root / "features" / band
        for path in sorted(folder.glob("*-original.npz")):
            name = path.name.removesuffix("-original.npz")
            loaded = {}
            for surface in ("original", "m7", "gated"):
                p = folder / f"{name}-{surface}.npz"
                if p.exists():
                    loaded[surface] = np.load(p)
            if "m7" not in loaded or any("ys" not in d for d in loaded.values()):
                continue
            ref = loaded["original"]
            same = all(
                np.array_equal(d["ys"], ref["ys"]) and np.array_equal(d["xs"], ref["xs"]) for d in loaded.values()
            )
            if not same:
                continue
            ink = ref["ink"].astype(bool)
            if ink.all() or not ink.any():
                continue
            readers = [str(r) for r in ref["readers"]]
            feats = {s: d["features"].astype(np.float32) for s, d in loaded.items()}
            row = {"band": band, "pair": name, "pixels": int(len(ink)), "auc": {}}
            for how in POOLINGS:
                for r, reader in enumerate(readers):
                    maps = {s: pooled(f[:, r], how) for s, f in feats.items()}
                    for s, m in maps.items():
                        row["auc"][f"{s}/{reader}/{how}"] = rank_auc(m[ink], m[~ink])
                    f2 = fused([maps["original"], maps["m7"]])
                    row["auc"][f"fused/{reader}/{how}"] = rank_auc(f2[ink], f2[~ink])
                    if "gated" in maps:
                        f3 = fused(list(maps.values()))
                        row["auc"][f"fused3/{reader}/{how}"] = rank_auc(f3[ink], f3[~ink])
                choice = max(("original", "m7"), key=lambda s: agreement(feats[s], how))
                row[f"agreement_choice/{how}"] = choice
                for reader in readers:
                    row["auc"][f"agreement/{reader}/{how}"] = row["auc"][f"{choice}/{reader}/{how}"]
                # every reader together, z-scored, on each surface and fused
                for s in feats:
                    m = fused([pooled(feats[s][:, r], how) for r in range(len(readers))])
                    row["auc"][f"{s}/all-readers/{how}"] = rank_auc(m[ink], m[~ink])
                m = fused([pooled(feats[s][:, r], how) for s in ("original", "m7") for r in range(len(readers))])
                row["auc"][f"fused/all-readers/{how}"] = rank_auc(m[ink], m[~ink])
            per_pair.append(row)
    keys = sorted({k for r in per_pair for k in r["auc"]})
    for k in keys:
        values = np.array([r["auc"][k] for r in per_pair if k in r["auc"]], dtype=np.float64)
        weights = np.array([r["pixels"] for r in per_pair if k in r["auc"]], dtype=np.float64)
        keep = np.isfinite(values)
        if keep.any():
            results[k] = {
                "pairs": int(keep.sum()),
                "median": round(float(np.median(values[keep])), 4),
                "weighted_mean": round(float(np.average(values[keep], weights=weights[keep])), 4),
            }
    out = root / "combine.json"
    out.write_text(
        json.dumps({"bands": list(bands), "pairs": len(per_pair), "scores": results, "per_pair": per_pair}, indent=1)
    )
    best = sorted(results.items(), key=lambda kv: -kv[1]["weighted_mean"])[:15]
    for k, v in best:
        print(f"{k:45s} pairs {v['pairs']:3d} median {v['median']:.4f} weighted {v['weighted_mean']:.4f}", flush=True)
    print("written", out, flush=True)
    return out


if __name__ == "__main__":
    {"score": score}[sys.argv[1]](*sys.argv[2:])
