"""Plan and build LOGOS labelled shards from HerculeX training bench controls."""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

SD, SH, SW = 64, 128, 128
CTX, CTXN, MAXPTS = 256, 4, 4096
LAYOUT_LABELLED = 0


def find_controls_json(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        sys.exit(f"bench_to_shard: missing {p}")
    here = Path(__file__).resolve()
    candidates = [here.parent / "controls.json"]
    for p in (here.parent, *here.parents):
        candidates.append(p / "calibrate" / "bench" / "controls.json")
    for c in candidates:
        if c.is_file():
            return c
    sys.exit("bench_to_shard: cannot find controls.json; pass --controls-json")


def default_controls_dir(controls_json):
    env_dir = os.environ.get("MODEL_BENCH_CONTROLS")
    if env_dir:
        return Path(env_dir)
    outputs = os.environ.get("HERCULEX_OUTPUTS")
    if outputs:
        return Path(outputs) / "bench" / "controls"
    for p in (controls_json.parent, *controls_json.parents):
        cand = p / "data" / "outputs" / "bench" / "controls"
        if cand.is_dir():
            return cand
    return Path(os.environ.get("MODEL_ROOT", "/content/model")) / "bench-controls"


def verify_train_control(manifest, control_name):
    controls = manifest.get("controls", {})
    if control_name not in controls:
        sys.exit(f"bench_to_shard: unknown control {control_name}")
    spec = controls[control_name]
    scan_key = spec["scan"]
    scan_info = manifest["scans"][scan_key]
    zones = scan_info["zones"]
    has_train = any(z[2] == "train" for z in zones)
    has_holdout = any(z[2].startswith("holdout") for z in zones)
    if has_holdout or not has_train:
        sys.exit(f"bench_to_shard: refusing non-training control {control_name} ({scan_key})")
    return spec, scan_info


def load_train_truth(controls_dir, control_name, scan_info):
    truth_path = controls_dir / control_name / "truth.npz"
    if not truth_path.is_file():
        sys.exit(f"bench_to_shard: missing {truth_path}")
    tr = np.load(truth_path)
    pts_xyz = tr["points"].astype(np.float32)
    pts = np.stack([pts_xyz[:, 2], pts_xyz[:, 1], pts_xyz[:, 0]], axis=1)
    meshes = tr["mesh"].astype(np.int32)
    windings = tr["winding"].astype(np.float64)
    knots = tr["knots"].astype(np.float64)
    curl = int(tr["sense"])
    voxel_um = float(tr["voxel_um"])
    keep = np.zeros(len(pts), dtype=bool)
    for z0, z1, label in scan_info["zones"]:
        if label == "train":
            keep |= (pts[:, 0] >= float(z0)) & (pts[:, 0] < float(z1))
    pts = pts[keep]
    meshes = meshes[keep]
    windings = windings[keep]
    if len(pts) == 0:
        sys.exit(f"bench_to_shard: no train-zone points in {control_name}")
    return pts, meshes, windings, knots, curl, voxel_um


def interp_axis(knots, zs):
    kz = knots[:, 0]
    kx = knots[:, 1]
    ky = knots[:, 2]
    cy = np.interp(zs, kz, ky).astype(np.float32)
    cx = np.interp(zs, kz, kx).astype(np.float32)
    return cy, cx


def build_context_slab(ctx, factor, oz, oy, ox):
    out = np.zeros((CTXN, CTX, CTX), dtype=np.uint8)
    step = 1
    while ctx.shape[1] // step > CTX or ctx.shape[2] // step > CTX:
        step += 1
    scale = float(factor * step)
    offs = (-24, -8, 8, 24)
    ny = min(CTX, (ctx.shape[1] + step - 1) // step)
    nx = min(CTX, (ctx.shape[2] + step - 1) // step)
    for si, off in enumerate(offs):
        cz = int(math.floor((oz + SD / 2 + off) / factor))
        cz = max(0, min(ctx.shape[0] - 1, cz))
        out[si, :ny, :nx] = ctx[cz, : ny * step : step, : nx * step : step]
    box = (
        float(oy / scale),
        float(ox / scale),
        float((oy + SH) / scale),
        float((ox + SW) / scale),
        scale,
    )
    return out, box


def write_record(f, meta, ct16, sheet, inst, ctx_slab, centre, pts):
    h = bytearray(128)
    h[0:4] = b"LGS1"
    struct.pack_into(
        "<5I2f3i5fI",
        h,
        4,
        1,
        meta["source"],
        meta["split"],
        meta["curl"] & 0xFFFFFFFF,
        meta["ct_bits"],
        meta["voxel_um"],
        meta["confidence"],
        meta["oz"],
        meta["oy"],
        meta["ox"],
        meta["box"][0],
        meta["box"][1],
        meta["box"][2],
        meta["box"][3],
        meta["box"][4],
        len(pts),
    )
    f.write(h)
    f.write(np.ascontiguousarray(ct16, dtype="<u2").tobytes())
    f.write(np.ascontiguousarray(sheet, dtype=np.uint8).tobytes())
    f.write(np.ascontiguousarray(inst, dtype="<u2").tobytes())
    f.write(np.ascontiguousarray(ctx_slab, dtype=np.uint8).tobytes())
    f.write(np.ascontiguousarray(centre, dtype="<f4").tobytes())
    ptbuf = bytearray(24 * MAXPTS)
    for i, (pz, py, px, sid, kind, wind) in enumerate(pts[:MAXPTS]):
        struct.pack_into("<3f2if", ptbuf, 24 * i, pz, py, px, sid, kind, wind)
    f.write(ptbuf)


def rasterize_slab_points(pts_zyx, meshes, windings, oz, oy, ox, dense_sheet, half_thick):
    z_rel = pts_zyx[:, 0] - oz
    y_rel = pts_zyx[:, 1] - oy
    x_rel = pts_zyx[:, 2] - ox
    sheet = np.full((SD, SH, SW), 0 if dense_sheet else 255, dtype=np.uint8)
    inst = np.zeros((SD, SH, SW), dtype=np.uint16)
    if dense_sheet and len(pts_zyx):
        zi = np.clip(np.rint(z_rel).astype(np.int32), 0, SD - 1)
        yi = np.clip(np.rint(y_rel).astype(np.int32), 0, SH - 1)
        xi = np.clip(np.rint(x_rel).astype(np.int32), 0, SW - 1)
        ids = (meshes.astype(np.int32) + 1).astype(np.uint16)
        for dz in range(-half_thick, half_thick + 1):
            zz = np.clip(zi + dz, 0, SD - 1)
            for dy in range(-half_thick, half_thick + 1):
                yy = np.clip(yi + dy, 0, SH - 1)
                for dx in range(-half_thick, half_thick + 1):
                    if dz * dz + dy * dy + dx * dx > half_thick * half_thick:
                        continue
                    xx = np.clip(xi + dx, 0, SW - 1)
                    sheet[zz, yy, xx] = 1
                    inst[zz, yy, xx] = ids
    records = []
    n = len(pts_zyx)
    step = max(1, n // (MAXPTS // 2))
    sel = np.arange(0, n, step)
    for idx in sel:
        pz, py, px = float(z_rel[idx]), float(y_rel[idx]), float(x_rel[idx])
        mid = int(meshes[idx])
        w = float(windings[idx])
        records.append((pz, py, px, 1000000 + mid, 0, math.nan))
        if math.isfinite(w):
            records.append((pz, py, px, 2000000, 1, w))
        if len(records) >= MAXPTS - 1:
            break
    return sheet, inst, records[:MAXPTS]


def bucket_grid(pts):
    oz = (np.floor(pts[:, 0] / SD).astype(np.int64)) * SD
    oy = (np.floor(pts[:, 1] / SH).astype(np.int64)) * SH
    ox = (np.floor(pts[:, 2] / SW).astype(np.int64)) * SW
    valid = (oz >= 0) & (oy >= 0) & (ox >= 0)
    idx = np.nonzero(valid)[0]
    keys = np.stack([oz[idx], oy[idx], ox[idx]], axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    idx_s = idx[order]
    uniq, starts, counts = np.unique(keys_s, axis=0, return_index=True, return_counts=True)
    return uniq, starts, counts, idx_s


def cmd_plan(args):
    cj = find_controls_json(args.controls_json)
    manifest = json.loads(cj.read_text())
    _, scan_info = verify_train_control(manifest, args.control)
    cdir = Path(args.controls_dir) if args.controls_dir else default_controls_dir(cj)
    pts, _, _, _, _, _ = load_train_truth(cdir, args.control, scan_info)
    uniq, _, counts, _ = bucket_grid(pts)
    ok = counts >= args.min_points
    boxes = uniq[ok]
    cnts = counts[ok]
    if args.max_slabs > 0 and len(boxes) > args.max_slabs:
        sel = np.linspace(0, len(boxes) - 1, args.max_slabs, dtype=np.int64)
        boxes = boxes[sel]
        cnts = cnts[sel]
    out_p = Path(args.plan_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for i, (oz, oy, ox) in enumerate(boxes):
            split = 1 if (args.holdout_every > 0 and i % args.holdout_every == args.holdout_every - 1) else 0
            f.write(f"{int(oz)} {int(oy)} {int(ox)} {split} {int(cnts[i])}\n")
    print(f"bench_to_shard plan: {args.control} -> {out_p} ({len(boxes)} slabs)", file=sys.stderr)


def cmd_build(args):
    cj = find_controls_json(args.controls_json)
    manifest = json.loads(cj.read_text())
    _, scan_info = verify_train_control(manifest, args.control)
    cdir = Path(args.controls_dir) if args.controls_dir else default_controls_dir(cj)
    pts, meshes, windings, knots, curl, voxel_um = load_train_truth(cdir, args.control, scan_info)
    uniq, starts, counts, idx_s = bucket_grid(pts)
    lookup = {(int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2])): (int(starts[i]), int(counts[i])) for i in range(len(uniq))}
    ctx = np.load(args.context_npy, mmap_mode="r")
    fetched_dir = Path(args.fetched_dir)
    out_path = Path(args.out_shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    source_id = 1 if args.dense_sheet else 3
    count = 0
    with open(args.plan_in) as pf, open(tmp_path, "wb") as f:
        head = bytearray(64)
        head[0:4] = b"LGSF"
        struct.pack_into("<IQ", head, 4, LAYOUT_LABELLED, 0)
        f.write(head)
        for row, line in enumerate(pf):
            parts = line.split()
            if len(parts) < 4:
                continue
            oz, oy, ox, split = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            npy_path = fetched_dir / f"{row:06d}.npy"
            if not npy_path.is_file():
                continue
            box_ct = np.load(npy_path)
            if box_ct.shape != (SD, SH, SW):
                continue
            hit = lookup.get((oz, oy, ox))
            if not hit:
                continue
            st, cn = hit
            inds = idx_s[st : st + cn]
            sheet, inst, pt_list = rasterize_slab_points(
                pts[inds], meshes[inds], windings[inds], oz, oy, ox, args.dense_sheet, args.half_thick
            )
            ctx_slab, box = build_context_slab(ctx, args.context_factor, oz, oy, ox)
            zs = np.arange(oz, oz + SD, dtype=np.float64)
            cy, cx = interp_axis(knots, zs)
            centre = np.stack([cy - oy, cx - ox], axis=1).astype(np.float32)
            meta = {
                "source": source_id,
                "split": split,
                "curl": curl,
                "ct_bits": 8,
                "voxel_um": voxel_um,
                "confidence": args.confidence,
                "oz": oz,
                "oy": oy,
                "ox": ox,
                "box": box,
            }
            write_record(f, meta, box_ct.astype(np.uint16), sheet, inst, ctx_slab, centre, pt_list)
            count += 1
        f.seek(8)
        f.write(struct.pack("<Q", count))
    os.replace(tmp_path, out_path)
    print(f"bench_to_shard build: {args.control} -> {out_path} ({count} slabs, source={source_id}, dense={args.dense_sheet})")


def cmd_volume(args):
    cj = find_controls_json(args.controls_json)
    manifest = json.loads(cj.read_text())
    spec, scan_info = verify_train_control(manifest, args.control)
    cdir = Path(args.controls_dir) if args.controls_dir else default_controls_dir(cj)
    pts, meshes, windings, knots, curl, voxel_um = load_train_truth(cdir, args.control, scan_info)
    ct = np.load(args.ct_npy, mmap_mode="r")
    ctx = np.load(args.context_npy, mmap_mode="r")
    if args.ct_origin:
        oz0, oy0, ox0 = (int(v) for v in args.ct_origin.split(","))
    else:
        oz0, oy0, ox0 = int(spec["z"][0]), 0, 0
    kz = np.floor((pts[:, 0] - oz0) / SD).astype(np.int32)
    ky = np.floor((pts[:, 1] - oy0) / SH).astype(np.int32)
    kx = np.floor((pts[:, 2] - ox0) / SW).astype(np.int32)
    valid = (
        (kz >= 0)
        & ((kz + 1) * SD <= ct.shape[0])
        & (ky >= 0)
        & ((ky + 1) * SH <= ct.shape[1])
        & (kx >= 0)
        & ((kx + 1) * SW <= ct.shape[2])
    )
    pts, meshes, windings = pts[valid], meshes[valid], windings[valid]
    kz, ky, kx = kz[valid], ky[valid], kx[valid]
    buckets = {}
    for i in range(len(pts)):
        buckets.setdefault((int(kz[i]), int(ky[i]), int(kx[i])), []).append(i)
    out_path = Path(args.out_shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    count = 0
    source_id = 1 if args.dense_sheet else 3
    with open(tmp_path, "wb") as f:
        head = bytearray(64)
        head[0:4] = b"LGSF"
        struct.pack_into("<IQ", head, 4, LAYOUT_LABELLED, 0)
        f.write(head)
        for idx, (bz, by, bx) in enumerate(sorted(buckets)):
            inds = buckets[(bz, by, bx)]
            if len(inds) < args.min_points:
                continue
            oz, oy, ox = oz0 + bz * SD, oy0 + by * SH, ox0 + bx * SW
            sub_ct = np.asarray(ct[bz * SD : (bz + 1) * SD, by * SH : (by + 1) * SH, bx * SW : (bx + 1) * SW], dtype=np.uint16)
            sheet, inst, pt_list = rasterize_slab_points(
                pts[inds], meshes[inds], windings[inds], oz, oy, ox, args.dense_sheet, args.half_thick
            )
            ctx_slab, box = build_context_slab(ctx, args.context_factor, oz, oy, ox)
            zs = np.arange(oz, oz + SD, dtype=np.float64)
            cy, cx = interp_axis(knots, zs)
            centre = np.stack([cy - oy, cx - ox], axis=1).astype(np.float32)
            meta = {
                "source": source_id,
                "split": 1 if (args.holdout_every > 0 and idx % args.holdout_every == args.holdout_every - 1) else 0,
                "curl": curl,
                "ct_bits": 8,
                "voxel_um": voxel_um,
                "confidence": args.confidence,
                "oz": oz,
                "oy": oy,
                "ox": ox,
                "box": box,
            }
            write_record(f, meta, sub_ct, sheet, inst, ctx_slab, centre, pt_list)
            count += 1
        f.seek(8)
        f.write(struct.pack("<Q", count))
    os.replace(tmp_path, out_path)
    print(f"bench_to_shard: {args.control} -> {out_path} ({count} slabs, source={source_id}, dense={args.dense_sheet})")


def main():
    ap = argparse.ArgumentParser(description="Convert HerculeX bench training controls into LOGOS shards.")
    sub = ap.add_subparsers(dest="cmd")

    p_plan = sub.add_parser("plan", help="Write a slab-fetch plan for one training control")
    p_plan.add_argument("control")
    p_plan.add_argument("plan_out")
    p_plan.add_argument("--controls-dir", default=None)
    p_plan.add_argument("--controls-json", default=None)
    p_plan.add_argument("--max-slabs", type=int, default=320)
    p_plan.add_argument("--min-points", type=int, default=256)
    p_plan.add_argument("--holdout-every", type=int, default=10)

    p_build = sub.add_parser("build", help="Pack fetched slab .npy files and truth.npz into a shard")
    p_build.add_argument("control")
    p_build.add_argument("plan_in")
    p_build.add_argument("fetched_dir")
    p_build.add_argument("context_npy")
    p_build.add_argument("out_shard")
    p_build.add_argument("--context-factor", type=float, required=True)
    p_build.add_argument("--controls-dir", default=None)
    p_build.add_argument("--controls-json", default=None)
    p_build.add_argument("--dense-sheet", action="store_true")
    p_build.add_argument("--half-thick", type=int, default=2)
    p_build.add_argument("--confidence", type=float, default=1.0)

    p_vol = sub.add_parser("volume", help="Convert a local 3D CT .npy box directly into a shard")
    p_vol.add_argument("control")
    p_vol.add_argument("ct_npy")
    p_vol.add_argument("context_npy")
    p_vol.add_argument("out_shard")
    p_vol.add_argument("--context-factor", type=float, required=True)
    p_vol.add_argument("--ct-origin", default=None)
    p_vol.add_argument("--controls-dir", default=None)
    p_vol.add_argument("--controls-json", default=None)
    p_vol.add_argument("--dense-sheet", action="store_true")
    p_vol.add_argument("--half-thick", type=int, default=2)
    p_vol.add_argument("--min-points", type=int, default=256)
    p_vol.add_argument("--confidence", type=float, default=1.0)
    p_vol.add_argument("--holdout-every", type=int, default=10)

    args = ap.parse_args()
    if args.cmd == "plan":
        cmd_plan(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "volume":
        cmd_volume(args)
    else:
        ap.print_help(sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
