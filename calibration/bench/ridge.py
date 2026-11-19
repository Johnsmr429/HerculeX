"""The truth against the CT: how far each published mesh vertex lies from the papyrus ridge along its normal.

The truth's own spread is the floor for any "distance to truth": a surface cannot be scored closer to the meshes
than the meshes lie to the papyrus. For a sample of truth vertices of a package (bench.package), the level-0 CT is
read in whole chunks around them and sampled along each vertex's normal (towards the axis is +) from -12 to +12
voxels in half-voxel steps. The ridge is the local maximum nearest to the vertex after smoothing at the sheet's
scale (Gaussian sigma 3 voxels: a 9.4 µm sheet is about 7 voxels wide at half height); sigma 1 is reported too but
CT texture then puts a maximum within a few voxels of any point. Every figure has a null beside it: the same search
from points 6 voxels off the mesh on either side. The mean profile shows where the papyrus is brightest.

Writes BENCH/checks/NAME-ridge.json (summary by zone, pressed and loose) and NAME-ridge.npz (truth index, offsets,
profiles) under the data root. Only chunks already in the store are read; a vertex whose window reaches a chunk
never fetched is skipped.
"""

import argparse
import json

import numpy as np
from scipy import ndimage

from calibration.bench import package as packages
from scans.chunks.volume import Volume
from storage.data_roots import SURFACE_BENCH as BENCH

HALF = 12.0
STEP = 0.5
SIGMAS = (3.0, 1.0)
NULL_VOX = 6.0
MARGIN = 16
CHECKS = BENCH / "checks"


def ridges(raw, offsets_t, sigma, reference=0.0):
    """Signed offset of the local maximum nearest to `reference` per profile, after Gaussian smoothing; NaN if none."""
    prof = ndimage.gaussian_filter1d(raw, sigma / STEP, axis=1)
    peak = (prof[:, 1:-1] > prof[:, :-2]) & (prof[:, 1:-1] >= prof[:, 2:])
    peak = np.pad(peak, ((0, 0), (1, 1)))
    distance = np.where(peak, np.abs(offsets_t - reference)[None, :], np.inf)
    nearest = np.argmin(distance, axis=1)
    has = np.isfinite(distance[np.arange(len(raw)), nearest])
    return np.where(has, offsets_t[nearest] - reference, np.nan)


def summarise(raw, offsets_t, groups):
    out = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        entry = {"vertices": int(mask.sum())}
        for sigma in SIGMAS:
            found = ridges(raw[mask], offsets_t, sigma)
            ok = np.isfinite(found)
            a = np.abs(found[ok])
            row = {
                "abs_p50_p90_vox": [round(float(v), 2) for v in np.percentile(a, (50, 90))],
                "signed_p50_vox": round(float(np.median(found[ok])), 2),
                "within_2_vox": round(float((a <= 2).mean()), 4),
                "within_4_vox": round(float((a <= 4).mean()), 4),
            }
            for side in (NULL_VOX, -NULL_VOX):
                null = ridges(raw[mask], offsets_t, sigma, side)
                b = np.abs(null[np.isfinite(null)])
                row[f"null_{side:+.0f}_abs_p50_p90_vox"] = [round(float(v), 2) for v in np.percentile(b, (50, 90))]
            entry[f"sigma_{sigma:g}"] = row
        mean = raw[mask].mean(axis=0)
        entry["mean_profile_peak_vox"] = round(float(offsets_t[int(np.argmax(mean))]), 2)
        entry["mean_profile_every_vox"] = [round(float(v), 1) for v in mean[:: int(round(1 / STEP))]]
        out[name] = entry
    return out


def check(name, chunks_per_zone=240, seed=0):
    truth = packages.load(name)
    record = truth["record"]
    entry = record["scan"]
    rng = np.random.default_rng(seed)
    unique = (truth["flags"] & packages.DUPLICATE) == 0
    unique &= np.all(np.isfinite(truth["normals"]), axis=1)
    points, normals = truth["points"].astype(np.float64), truth["normals"].astype(np.float64)
    offsets_t = np.arange(-HALF, HALF + STEP / 2, STEP)
    with Volume.for_scan(entry["sample"], entry["scan"], "ct", 0) as volume:
        size = np.asarray(volume.chunks)
        key = np.floor(points[:, ::-1] / size).astype(np.int64)
        picks = []
        for z in np.unique(truth["zone"]):
            inside = np.flatnonzero((truth["zone"] == z) & unique)
            cells, inverse = np.unique(key[inside], axis=0, return_inverse=True)
            chosen = rng.choice(len(cells), min(chunks_per_zone, len(cells)), replace=False)
            picks.append((cells[chosen], [inside[inverse.ravel() == c] for c in chosen]))
        rows, profiles = [], []
        for cells, members in picks:
            for cell, idx in zip(cells, members, strict=True):
                low = cell * size - MARGIN
                high = (cell + 1) * size + MARGIN
                box = (low[0], high[0], low[1], high[1], low[2], high[2])
                data, unknown = volume.read_with_unknown(box)
                if unknown.any():
                    continue
                data = data.astype(np.float32)
                p = points[idx][:, None, :] + offsets_t[None, :, None] * normals[idx][:, None, :]
                coords = np.stack([p[..., 2] - low[0], p[..., 1] - low[1], p[..., 0] - low[2]])
                prof = ndimage.map_coordinates(data, coords.reshape(3, -1), order=1, mode="nearest").reshape(
                    len(idx), -1
                )
                rows.append(idx)
                profiles.append(prof.astype(np.float32))
    rows, raw = np.concatenate(rows), np.concatenate(profiles)
    zone = truth["zone"][rows]
    pressed = (truth["flags"][rows] & packages.PRESSED) > 0
    groups = {"all": np.ones(len(rows), dtype=bool)}
    for z in np.unique(zone):
        groups[packages.ZONES[z]] = zone == z
    groups["pressed"] = pressed
    groups["loose"] = ~pressed
    out = {
        "control": name,
        "scan": entry,
        "window_vox": [-HALF, HALF],
        "step_vox": STEP,
        "smoothing_sigma_vox": list(SIGMAS),
        "null_offset_vox": NULL_VOX,
        "sign": "+ is towards the scroll axis along the truth normal",
        "chunks_per_zone": chunks_per_zone,
        "by_group": summarise(raw, offsets_t, groups),
    }
    CHECKS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKS / f"{name}-ridge.npz",
        truth_index=rows,
        ridge_sigma3_vox=ridges(raw, offsets_t, SIGMAS[0]).astype(np.float32),
        profiles=raw.astype(np.float16),
        offsets_vox=offsets_t.astype(np.float32),
    )
    (CHECKS / f"{name}-ridge.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--chunks", type=int, default=240, help="CT chunks sampled per zone")
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(check(name, args.chunks), indent=1))


if __name__ == "__main__":
    main()
