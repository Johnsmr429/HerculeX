import argparse
import json
import os
import shutil
import time
from pathlib import Path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--axis-source", default="")
    ap.add_argument("--curl", type=int, required=True)
    ap.add_argument("--band", type=int, default=2560)
    ap.add_argument("--method", default="unroll")
    ap.add_argument("--shape", required=True)
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--min-cm2", type=float, default=0.0)
    args = ap.parse_args()
    shape = [int(v) for v in args.shape.split(",")]
    level = os.path.join(args.root, "unwrapping", "level", args.scroll, args.scan)
    if os.path.exists(os.path.join(level, "pieces")):
        shutil.rmtree(os.path.join(level, "pieces"))
    os.makedirs(os.path.join(level, "sheet"), exist_ok=True)
    src = json.loads(Path(os.path.join(args.pieces, "record.json")).read_text())
    meta_by_uuid = {p["uuid"]: p for p in src["pieces"]}
    lay = json.loads(Path(args.layout).read_text())
    rows = [r for r in lay["pieces"] if r["area"] / 100.0 >= args.min_cm2]
    units = {}
    for r in rows:
        m = meta_by_uuid[r["uuid"]]
        zmid = 0.5 * (m["bbox"][0][2] + m["bbox"][1][2])
        b = int(zmid // args.band)
        z0, z1 = b * args.band, (b + 1) * args.band
        name = f"band-{z0 // 256:04d}-{z1 // 256:04d}"
        units.setdefault(name, {"z0": z0, "z1": z1, "pieces": []})
        units[name]["pieces"].append((r, m))
        r["unit"] = name
    for name, u in units.items():
        udir = os.path.join(level, "pieces", name)
        os.makedirs(udir, exist_ok=True)
        os.makedirs(os.path.join(level, "nodes", name), exist_ok=True)
        kept = []
        for r, m in u["pieces"]:
            s = os.path.abspath(os.path.join(args.pieces, m["uuid"]))
            d = os.path.join(udir, m["uuid"])
            os.symlink(s, d)
            e = dict(m)
            e.setdefault("level_of_nodes", 1)
            e["shape"] = shape
            e.setdefault("turns_spanned", 0.25)
            e.setdefault("source", e["piece"])
            e.setdefault("block", [0.0, 0.0])
            e["overlap"] = [0, 0, 0, 0]
            e.setdefault("frame", 0)
            e.setdefault("median_radius_mm", 10.0)
            kept.append(e)
        rec = {
            "sample": args.scroll,
            "volume": args.scan,
            "voxel_um": src["voxel_um"],
            "level": 1,
            "chunks": [name],
            "cell_planes": [u["z0"] / 16.0, u["z1"] / 16.0],
            "shared": [
                [u["z0"] / 16.0, u["z0"] / 16.0 + 4.0],
                [u["z1"] / 16.0 - 4.0, u["z1"] / 16.0],
            ],
            "centres": {"rule": "umbilicus"},
            "method": args.method,
            "chirality": args.curl,
            "piece_count": len(kept),
            "area_cm2": sum(p["area_mm2"] for p in kept) / 100.0,
            "pieces": kept,
        }
        Path(os.path.join(udir, "record.json")).write_text(json.dumps(rec))
    Path(os.path.join(level, "curl.json")).write_text(
        json.dumps({"curl": args.curl, "rule": "set by hand"})
    )
    shutil.copyfile(args.axis, os.path.join(level, "sheet", "axis.json"))
    if args.axis_source:
        shutil.copyfile(
            args.axis_source, os.path.join(level, "sheet", "axis-source.json")
        )
    Path(os.path.join(level, "sheet", "joints.json")).write_text("[]")
    Path(os.path.join(args.root, "unwrapping", "level", "pieces.finished")).write_text(
        "finished\n"
    )
    xs = [r["x_mm"] for r in rows] + [r["x_mm"] + r["width_mm"] for r in rows]
    ys = [r["y_mm"] for r in rows] + [r["y_mm"] + r["height_mm"] for r in rows]
    x_lo, x_hi, y_lo, y_hi = min(xs), max(xs), min(ys), max(ys)
    out_rows = []
    for r in rows:
        o = dict(r)
        o.pop("uuid", None)
        o.pop("resid_vox", None)
        o.pop("phi_mean", None)
        if args.rotate180:
            o["x_mm"] = round(x_lo + x_hi - r["x_mm"] - r["width_mm"], 4)
            o["y_mm"] = round(r["y_mm"] - y_lo, 4)
            o["flip_rows"] = False
            o["flip_cols"] = True
        else:
            o["x_mm"] = round(r["x_mm"] - x_lo, 4)
            o["y_mm"] = round(y_hi - r["y_mm"] - r["height_mm"], 4)
            o["flip_rows"] = True
            o["flip_cols"] = False
        if o.pop("row_rev", False):
            o["flip_rows"] = not o["flip_rows"]
            o["flip_cols"] = not o["flip_cols"]
        o.pop("col_rev", None)
        o["turn"] = 1
        o["moved_from_joint_mm"] = 0
        o["group"] = 1
        o["overlap"] = [0, 0, 0, 0]
        o["level"] = 0
        o["source"] = o["piece"]
        out_rows.append(o)
    record = dict(lay["record"])
    record.update(
        sample=args.scroll,
        volume=args.scan,
        chirality=args.curl,
        pieces=len(out_rows),
        layout="scroll strip from spiral angle",
    )
    time.sleep(1.1)
    Path(os.path.join(level, "sheet", "layout.json")).write_text(
        json.dumps({"record": record, "pieces": out_rows})
    )
    total = sum(r["area"] for r in out_rows) / 100.0
    print(
        f"root {args.root}: {len(units)} units, {len(out_rows)} pieces, {total:.1f} cm2",
        flush=True,
    )


if __name__ == "__main__":
    main()
