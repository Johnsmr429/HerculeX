import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def read_tif(path):
    return np.array(Image.open(path), dtype=np.float32)


def reversed_axes(phi, z, known):
    kc = known[:, 1:] & known[:, :-1]
    kr = known[1:, :] & known[:-1, :]
    dc = (phi[:, 1:] - phi[:, :-1])[kc]
    dr = (z[1:, :] - z[:-1, :])[kr]
    col_rev = bool(len(dc) and np.median(dc) < 0)
    row_rev = bool(len(dr) and np.median(dr) < 0)
    return col_rev, row_rev


def load_pieces(folder, min_cm2):
    rec = json.loads(Path(os.path.join(folder, "record.json")).read_text())
    out = []
    for p in rec["pieces"]:
        if p["area_mm2"] / 100.0 < min_cm2:
            continue
        d = os.path.join(folder, p["uuid"])
        phi = read_tif(os.path.join(d, "phi.tif"))
        known = np.isfinite(phi)
        if known.sum() < 4:
            continue
        grids = {
            "phi": phi,
            "known": known,
            "x": read_tif(os.path.join(d, "x.tif")),
            "y": read_tif(os.path.join(d, "y.tif")),
            "z": read_tif(os.path.join(d, "z.tif")),
        }
        col_rev, row_rev = reversed_axes(grids["phi"], grids["z"], known)
        for name, grid in grids.items():
            g = grid[:, ::-1] if col_rev else grid
            g = g[::-1, :] if row_rev else g
            grids[name] = np.ascontiguousarray(g)
        out.append(dict(meta=p, col_rev=col_rev, row_rev=row_rev, **grids))
    return rec, out


def load_axis(path):
    ax = np.array(json.loads(Path(path).read_text()), dtype=float)
    return ax[np.argsort(ax[:, 0])]


def consensus(
    pieces, axis, min_res, min_frac, cap=400, theta_bins=90, z_bin=200.0, r_bin=6.0
):

    cols = []
    for i, p in enumerate(pieces):
        k = p["known"]
        n = int(k.sum())
        step = max(1, n // cap)
        f = p["phi"][k][::step]
        cnt = len(f)
        cols.append(
            np.stack(
                [
                    p["x"][k][::step],
                    p["y"][k][::step],
                    p["z"][k][::step],
                    f,
                    np.full(cnt, p["meta"]["area_mm2"] / cnt),
                    np.full(cnt, i),
                ],
                1,
            )
        )
    c = np.concatenate(cols).astype(np.float64)
    x, y, z, phi, w, pid = c.T
    pid = pid.astype(np.int64)
    cx = np.interp(z, axis[:, 0], axis[:, 1])
    cy = np.interp(z, axis[:, 0], axis[:, 2])
    r = np.hypot(x - cx, y - cy)
    th = np.arctan2(y - cy, x - cx)
    turn = phi / (2 * np.pi)
    tb = ((th + np.pi) / (2 * np.pi) * theta_bins).astype(np.int64) % theta_bins
    key = ((z // z_bin).astype(np.int64) * theta_bins + tb) * 100000 + (
        r // r_bin
    ).astype(np.int64)
    o = np.lexsort((turn, key))
    ks, ts, ws = key[o], turn[o], w[o]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    cw = np.cumsum(ws)
    base = np.r_[0.0, cw[:-1]][starts]
    mid = np.searchsorted(cw, base + (cw[ends - 1] - base) / 2)
    med = ts[np.minimum(mid, len(ts) - 1)]
    cons = np.empty_like(turn)
    cons[o] = np.repeat(med, ends - starts)
    res = turn - cons
    order = np.lexsort((res, pid))
    ps, rs = pid[order], res[order]
    st = np.r_[0, np.nonzero(np.diff(ps))[0] + 1]
    en = np.r_[st[1:], len(ps)]
    moved = 0
    for a, b in zip(st, en):
        rr = rs[a:b]
        m = float(np.median(rr))
        t = int(np.round(m))
        if abs(m) < min_res or float(np.mean(np.abs(rr - t) < 0.5)) < min_frac:
            continue
        p = pieces[ps[a]]
        p["phi"] = p["phi"] - 2 * np.pi * t
        p["turn_shift"] = p.get("turn_shift", 0) - t
        moved += 1
    return moved


def rate_table(pieces, stride, bin_rad, min_weight=200.0, min_rate=20.0, trim=0.0005):
    vals, wts = [], []
    for p in pieces:
        f = p["phi"][p["known"]][::16]
        vals.append(f)
        wts.append(np.full(len(f), p["meta"]["area_mm2"] / max(len(f), 1)))
    vals, wts = np.concatenate(vals), np.concatenate(wts)
    o = np.argsort(vals)
    cw = np.cumsum(wts[o]) / wts.sum()
    lo = float(vals[o][np.searchsorted(cw, trim)])
    hi = float(vals[o][min(len(o) - 1, np.searchsorted(cw, 1 - trim))])
    nb = int(np.ceil((hi - lo) / bin_rad)) + 2
    ds = np.zeros(nb)
    dp = np.zeros(nb)
    nc = np.zeros(nb)
    for p in pieces:
        phi, k = p["phi"], p["known"]
        kk = k[:, 1:] & k[:, :-1]
        dphi = (phi[:, 1:] - phi[:, :-1])[kk]
        d3 = np.sqrt(
            (p["x"][:, 1:] - p["x"][:, :-1]) ** 2
            + (p["y"][:, 1:] - p["y"][:, :-1]) ** 2
            + (p["z"][:, 1:] - p["z"][:, :-1]) ** 2
        )[kk]
        mid = (0.5 * (phi[:, 1:] + phi[:, :-1]))[kk]
        good = (np.abs(dphi) > 1e-5) & (d3 < 3 * stride) & (mid >= lo) & (mid < hi)
        idx = ((mid[good] - lo) / bin_rad).astype(np.int64)
        np.add.at(ds, idx, d3[good])
        np.add.at(dp, idx, np.abs(dphi[good]))
        np.add.at(nc, idx, 1.0)
    rate = np.where(dp > 1e-3, ds / np.maximum(dp, 1e-9), np.nan)
    centers = lo + (np.arange(nb) + 0.5) * bin_rad
    dense = np.isfinite(rate) & (nc >= min_weight)
    turn = np.floor(centers / (2 * np.pi)).astype(np.int64)
    tu = np.unique(turn)
    t_ds = np.array([ds[(turn == t) & dense].sum() for t in tu])
    t_dp = np.array([dp[(turn == t) & dense].sum() for t in tu])
    t_nc = np.array([nc[(turn == t) & dense].sum() for t in tu])
    full = t_nc >= min_weight * (2 * np.pi / bin_rad) * 0.5
    t_rate = t_ds[full] / np.maximum(t_dp[full], 1e-9)
    tc = (tu[full] + 0.5) * 2 * np.pi
    w = np.sqrt(t_nc[full])
    coef = np.linalg.lstsq(
        np.vstack([tc, np.ones_like(tc)]).T * w[:, None], t_rate * w, rcond=None
    )[0]
    model = np.maximum(min_rate, coef[0] * centers + coef[1])
    rate = np.where(dense, rate, model)
    cum = np.concatenate([[0.0], np.cumsum(rate * bin_rad)])
    edges = lo + np.arange(nb + 1) * bin_rad
    return edges, cum, centers, rate


def arc_of(phi, edges, cum):
    return np.interp(phi, edges, cum)


def place(pieces, edges, cum, stride):
    for p in pieces:
        k = p["known"]
        rr, cc = np.nonzero(k)
        s = arc_of(p["phi"][k], edges, cum)
        p["ox"] = float(np.median(s - cc * stride))
        p["oy"] = float(np.median(p["z"][k] - rr * stride))
        p["resid"] = float(np.sqrt(np.mean((s - cc * stride - p["ox"]) ** 2)))


def coarse_cells(p, stride, coarse):
    rr, cc = np.nonzero(p["known"])
    gy = np.floor((p["oy"] + rr * stride) / (coarse * stride)).astype(np.int64)
    gx = (cc // coarse).astype(np.int64) + 500000
    return np.unique(gy * 1000000 + gx)


def side_by_side(a, b, shift, max_overlap):
    if len(a) == 0 or len(b) == 0:
        return False
    common = np.intersect1d(a, b + shift, assume_unique=True)
    return len(common) <= max_overlap * min(len(a), len(b))


def joints(pieces, stride, radius, min_contacts, max_jump, max_overlap, coarse=4, slab=512.0):
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    pts, ids, cols, phis = [], [], [], []
    for i, p in enumerate(pieces):
        k = p["known"]
        edge = k & ~ndi.binary_erosion(k, iterations=2)
        rr, cc = np.nonzero(edge)
        if len(rr) == 0:
            continue
        pts.append(np.stack([p["x"][rr, cc], p["y"][rr, cc], p["z"][rr, cc]], 1))
        ids.append(np.full(len(rr), i))
        cols.append(cc)
        phis.append(p["phi"][rr, cc])
    pt = np.concatenate(pts)
    idx = np.concatenate(ids)
    col = np.concatenate(cols)
    ph = np.concatenate(phis)
    z = pt[:, 2]
    found_pairs = []
    lo = float(np.floor(z.min()))
    while lo <= z.max():
        hi = lo + slab
        sel = np.nonzero((z >= lo) & (z < hi + radius))[0]
        if len(sel) > 1:
            near = cKDTree(pt[sel]).query_pairs(r=radius, output_type="ndarray")
            i, j = sel[near[:, 0]], sel[near[:, 1]]
            keep = (np.minimum(z[i], z[j]) < hi) & (idx[i] != idx[j]) & (np.abs(ph[i] - ph[j]) < 0.3)
            found_pairs.append(np.stack([i[keep], j[keep]], 1))
        lo = hi
    pairs = np.concatenate(found_pairs) if found_pairs else np.zeros((0, 2), np.int64)
    first = np.where(idx[pairs[:, 0]] < idx[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
    second = np.where(idx[pairs[:, 0]] < idx[pairs[:, 1]], pairs[:, 1], pairs[:, 0])
    key = idx[first].astype(np.int64) * len(pieces) + idx[second]
    off = (col[first] - col[second]).astype(np.float64) * stride
    o = np.argsort(key, kind="stable")
    key, off = key[o], off[o]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    ends = np.r_[starts[1:], len(key)]
    cells = {}
    found = []
    for s, e in zip(starts, ends):
        if e - s < min_contacts:
            continue
        ia, ib = divmod(int(key[s]), len(pieces))
        d = float(np.median(off[s:e]))
        if abs(d - (pieces[ib]["ox"] - pieces[ia]["ox"])) > max_jump:
            continue
        for i in (ia, ib):
            if i not in cells:
                cells[i] = coarse_cells(pieces[i], stride, coarse)
        if not side_by_side(
            cells[ia], cells[ib], round(d / (coarse * stride)), max_overlap
        ):
            continue
        found.append((ia, ib, d, e - s))
    return found


def solve_joints(pieces, found, prior):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    n = len(pieces)
    rows, cols, vals, rhs = [], [], [], []
    for r, (ia, ib, d, cnt) in enumerate(found):
        w = min(np.sqrt(cnt), 10.0)
        rows += [r, r]
        cols += [ib, ia]
        vals += [w, -w]
        rhs.append(w * d)
    base = len(found)
    ox0 = np.array([p["ox"] for p in pieces])
    for i in range(n):
        rows.append(base + i)
        cols.append(i)
        vals.append(prior)
        rhs.append(prior * ox0[i])
    m = coo_matrix((vals, (rows, cols)), shape=(base + n, n)).tocsr()
    ox = lsqr(m, np.array(rhs), atol=1e-10, btol=1e-10, iter_lim=20000, x0=ox0)[0]
    for i, p in enumerate(pieces):
        p["joint_moved"] = float(ox[i] - ox0[i])
        p["ox"] = float(ox[i])
    res = np.array([ox[ib] - ox[ia] - d for ia, ib, d, _ in found])
    par = np.arange(n)

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for ia, ib, _, _ in found:
        ra, rb = root(ia), root(ib)
        if ra != rb:
            par[rb] = ra
    group = np.array([root(i) for i in range(n)])
    for i, p in enumerate(pieces):
        p["group"] = int(group[i])
    return res


def pack_groups(pieces, stride, total_len, coarse, max_shift, step, tolerance):
    zmax = max(float(np.nanmax(p["z"][p["known"]])) for p in pieces) + 2 * stride
    w = int((total_len + 2 * max_shift) / coarse) + 8
    h = int(zmax / coarse) + 8
    occ = np.zeros(h * w, dtype=bool)
    base = int(max_shift / coarse) + 4
    groups = {}
    for p in pieces:
        groups.setdefault(p["group"], []).append(p)
    order = sorted(
        groups.values(), key=lambda g: -sum(q["meta"]["area_mm2"] for q in g)
    )
    offsets = [0]
    for k in range(1, int(max_shift / step) + 1):
        offsets += [k * step, -k * step]
    moved = 0
    for g in order:
        gys, gxs = [], []
        for p in g:
            rr, cc = np.nonzero(p["known"])
            gys.append(((p["oy"] + rr * stride) / coarse).astype(np.int64))
            gxs.append((p["ox"] + cc * stride) / coarse)
        gy = np.concatenate(gys)
        gx0 = np.concatenate(gxs)
        best = None
        for off in offsets:
            gx = (gx0 + off / coarse).astype(np.int64) + base
            ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
            f = np.unique(gy[ok] * w + gx[ok])
            share = float(occ[f].mean()) if len(f) else 1.0
            if best is None or share < best[0] - 1e-9:
                best = (share, off, f)
            if share <= tolerance:
                break
        for p in g:
            p["ox"] += best[1]
            p["moved_mm"] = best[1]
        moved += len(g) if best[1] != 0 else 0
        occ[best[2]] = True
    return moved


def group_stats(pieces):
    if "group" not in pieces[0]:
        return {}
    area = {}
    for p in pieces:
        area[p["group"]] = area.get(p["group"], 0.0) + p["meta"]["area_mm2"] / 100.0
    a = np.array(list(area.values()))
    return {
        "groups": len(a),
        "largest_group_cm2": round(float(a.max()), 2),
        "groups_4cm2": int((a >= 4).sum()),
        "area_groups_4cm2": round(float(a[a >= 4].sum()), 1),
        "groups_10cm2": int((a >= 10).sum()),
        "area_groups_10cm2": round(float(a[a >= 10].sum()), 1),
    }


def coverage(pieces, stride, total_len):
    zmax = max(float(np.nanmax(p["z"][p["known"]])) for p in pieces) + 2 * stride
    w = int(total_len / stride) + 4
    h = int(zmax / stride) + 4
    cnt = np.zeros((h, w), dtype=np.uint16)
    for p in pieces:
        k = p["known"]
        rr, cc = np.nonzero(k)
        gx = ((p["ox"] + cc * stride) / stride).astype(np.int64)
        gy = ((p["oy"] + rr * stride) / stride).astype(np.int64)
        okb = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        lin = np.unique(gy[okb] * w + gx[okb])
        np.add.at(cnt.reshape(-1), lin, 1)
    once = int((cnt == 1).sum())
    multi = int((cnt >= 2).sum())
    return cnt, once, multi


def pack(pieces, stride, total_len, coarse, max_shift, step, tolerance):
    zmax = max(float(np.nanmax(p["z"][p["known"]])) for p in pieces) + 2 * stride
    w = int((total_len + 2 * max_shift) / coarse) + 8
    h = int(zmax / coarse) + 8
    occ = np.zeros(h * w, dtype=bool)
    base = int(max_shift / coarse) + 4
    moved = 0
    offsets = [0]
    for k in range(1, int(max_shift / step) + 1):
        offsets += [k * step, -k * step]
    for p in sorted(pieces, key=lambda q: -q["meta"]["area_mm2"]):
        k = p["known"]
        rr, cc = np.nonzero(k)
        gy = ((p["oy"] + rr * stride) / coarse).astype(np.int64)
        gx0 = (p["ox"] + cc * stride) / coarse
        best = None
        for off in offsets:
            gx = (gx0 + off / coarse).astype(np.int64) + base
            ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
            f = np.unique(gy[ok] * w + gx[ok])
            share = float(occ[f].mean()) if len(f) else 1.0
            if best is None or share < best[0] - 1e-9:
                best = (share, off, f)
            if share <= tolerance:
                break
        p["ox"] += best[1]
        p["moved_mm"] = best[1]
        moved += best[1] != 0
        occ[best[2]] = True
    return moved


def render(pieces, stride, total_len, scale, path, color_by="piece"):
    zmax = max(float(np.nanmax(p["z"][p["known"]])) for p in pieces) + 2 * stride
    w = int(total_len / scale) + 4
    h = int(zmax / scale) + 4
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for p in sorted(pieces, key=lambda q: q["meta"]["area_mm2"]):
        k = p["known"]
        rr, cc = np.nonzero(k)
        gx = ((p["ox"] + cc * stride) / scale).astype(np.int64)
        gy = ((p["oy"] + rr * stride) / scale).astype(np.int64)
        okb = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        col = rng.integers(60, 255, 3).astype(np.uint8)
        img[gy[okb], gx[okb]] = col
    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cm2", type=float, default=0.0)
    ap.add_argument("--stride", type=float, default=8.0)
    ap.add_argument("--bin-rad", type=float, default=0.05)
    ap.add_argument("--voxel-um", type=float, default=0.0)
    ap.add_argument("--render-scale", type=float, default=64.0)
    ap.add_argument("--pack-mm", type=float, default=0.0)
    ap.add_argument("--pack-tolerance", type=float, default=0.2)
    ap.add_argument("--axis", default="")
    ap.add_argument("--consensus-rounds", type=int, default=0)
    ap.add_argument("--consensus-min", type=float, default=1.5)
    ap.add_argument("--consensus-frac", type=float, default=0.6)
    ap.add_argument("--trim", type=float, default=0.0005)
    ap.add_argument("--joint-radius", type=float, default=0.0)
    ap.add_argument("--joint-contacts", type=int, default=20)
    ap.add_argument("--joint-jump-mm", type=float, default=12.0)
    ap.add_argument("--joint-overlap", type=float, default=0.15)
    ap.add_argument("--joint-prior", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec, pieces = load_pieces(args.pieces, args.min_cm2)
    args.voxel_um = args.voxel_um or float(rec["voxel_um"])
    relabelled = []
    if args.consensus_rounds > 0:
        axis = load_axis(args.axis)
        for _ in range(args.consensus_rounds):
            relabelled.append(
                consensus(pieces, axis, args.consensus_min, args.consensus_frac)
            )
            print(f"consensus moved {relabelled[-1]} pieces", flush=True)
    edges, cum, centers, rate = rate_table(
        pieces, args.stride, args.bin_rad, trim=args.trim
    )
    place(pieces, edges, cum, args.stride)
    total_len = float(cum[-1]) + 4 * args.stride
    shift = min(p["ox"] for p in pieces)
    for p in pieces:
        p["ox"] -= shift
    total_len -= shift
    found, res = [], np.zeros(0)
    if args.joint_radius > 0:
        found = joints(
            pieces,
            args.stride,
            args.joint_radius,
            args.joint_contacts,
            args.joint_jump_mm / (args.voxel_um * 1e-3),
            args.joint_overlap,
        )
        res = solve_joints(pieces, found, args.joint_prior)
        print(f"joints {len(found)}", flush=True)
    packed = 0
    if args.pack_mm > 0:
        vox = args.pack_mm / (args.voxel_um * 1e-3)
        if found:
            packed = pack_groups(
                pieces, args.stride, total_len, 16.0, vox, 64.0, args.pack_tolerance
            )
        else:
            packed = pack(
                pieces, args.stride, total_len, 16.0, vox, 64.0, args.pack_tolerance
            )
        lo = min(p["ox"] for p in pieces)
        for p in pieces:
            p["ox"] -= lo
        total_len = (
            max(p["ox"] + p["phi"].shape[1] * args.stride for p in pieces)
            + 4 * args.stride
        )
    cnt, once, multi = coverage(pieces, args.stride, total_len + 64)
    cell_cm2 = (args.stride * args.voxel_um * 1e-4) ** 2
    area = sum(p["meta"]["area_mm2"] for p in pieces) / 100.0
    resid = np.array([p["resid"] for p in pieces])
    wts = np.array([p["meta"]["area_mm2"] for p in pieces])
    stats = dict(
        pieces=len(pieces),
        area_cm2=round(area, 2),
        unrolled_once_cm2=round(once * cell_cm2, 2),
        unrolled_multi_cm2=round(multi * cell_cm2, 2),
        overlap_share=round(multi / max(1, once + multi), 4),
        strip_length_mm=round(total_len * args.voxel_um * 1e-3, 1),
        resid_vox_weighted=round(float(np.sum(resid * wts) / np.sum(wts)), 1),
        rate_median=round(float(np.median(rate)), 1),
        packed=packed,
        relabelled=relabelled,
        joints=len(found),
        joint_resid_mm=[
            round(float(np.percentile(np.abs(res), q)) * args.voxel_um * 1e-3, 3)
            for q in (50, 90)
        ]
        if len(res)
        else [],
        **group_stats(pieces),
        col_rev=sum(p["col_rev"] for p in pieces),
        row_rev=sum(p["row_rev"] for p in pieces),
    )
    print(json.dumps(stats), flush=True)
    Path(os.path.join(args.out, "layout_stats.json")).write_text(
        json.dumps(stats, indent=1)
    )
    np.savez(
        os.path.join(args.out, "arc_table.npz"),
        edges=edges,
        cum=cum,
        centers=centers,
        rate=rate,
    )
    mm = args.voxel_um * 1e-3
    rows = []
    for p in pieces:
        m = p["meta"]
        h, w = p["phi"].shape
        rows.append(
            {
                "piece": int(m["piece"]),
                "uuid": m["uuid"],
                "x_mm": round(p["ox"] * mm, 4),
                "y_mm": round(p["oy"] * mm, 4),
                "width_mm": round(w * args.stride * mm, 4),
                "height_mm": round(h * args.stride * mm, 4),
                "area": round(m["area_mm2"], 3),
                "um": mm,
                "stride": args.stride,
                "shape": [h, w],
                "flip_rows": False,
                "flip_cols": False,
                "group": 1,
                "moved_from_joint_mm": round(p.get("moved_mm", 0.0) * mm, 3),
                "phi_mean": m.get("phi_mean"),
                "resid_vox": round(p["resid"], 2),
                "turn_shift": p.get("turn_shift", 0),
                "col_rev": p["col_rev"],
                "joint_group": p.get("group", -1),
                "row_rev": p["row_rev"],
            }
        )
    Path(os.path.join(args.out, "scroll_layout.json")).write_text(
        json.dumps({"record": stats, "pieces": rows})
    )
    render(
        pieces,
        args.stride,
        total_len,
        args.render_scale,
        os.path.join(args.out, "overview.png"),
    )
    multi_img = (np.clip(cnt, 0, 3) * 85).astype(np.uint8)
    Image.fromarray(multi_img).resize(
        (max(1, multi_img.shape[1] // 8), max(1, multi_img.shape[0] // 8)),
        Image.NEAREST,
    ).save(os.path.join(args.out, "multiplicity.png"))


if __name__ == "__main__":
    main()
