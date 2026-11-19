"""v1 level-route tiles put back together into the source pieces they were cut from.

The level route writes each source piece (a connected set of firm edges, one exact level set, often spanning dozens
of turns) as tiles of at most 112 + 16 cells (about 21.6 mm square), each tile's meta.json naming its `source` and
its `block` (row, column offset in the source's flat cell grid). Scored tile by tile, a source looks no larger than
2.7 cm2; this puts every source's tiles back at their offsets as one tifxyz, so continuity is measured on the
surface the route actually made. Tiles without a source (coarse fallback pieces) are linked through unchanged.

  ./run.sh calibration.bench.v1_sources PIECES_DIR OUT_DIR
  ./run.sh calibration.bench.score PHerc0139-dev OUT_DIR --system v1 --label v1-level-sources-local --ignore-winding
"""

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import tifffile


def assemble(pieces, out):
    pieces, out = Path(pieces), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tiles, loose = collections.defaultdict(list), []
    for folder in sorted(pieces.glob("piece-*")):
        meta = json.loads((folder / "meta.json").read_text())
        (tiles[meta["source"]] if "source" in meta else loose).append((folder, meta))
    report = []
    for source, group in sorted(tiles.items()):
        extents = []
        for folder, meta in group:
            h, w = tifffile.imread(folder / "x.tif").shape
            extents.append((meta["block"][0] + h, meta["block"][1] + w))
        rows, cols = max(e[0] for e in extents), max(e[1] for e in extents)
        grid = np.full((rows, cols, 3), -1, np.float32)
        quality = np.zeros((rows, cols), np.uint8)
        for folder, meta in group:
            xyz = np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"], axis=-1)
            has_quality = (folder / "quality.npy").exists()
            q = np.load(folder / "quality.npy") if has_quality else np.ones(xyz.shape[:2], np.uint8)
            ok = np.all(xyz > 0, axis=-1) & (q > 0)
            r0, c0 = meta["block"]
            h, w = xyz.shape[:2]
            grid[r0 : r0 + h, c0 : c0 + w][ok] = xyz[ok]
            quality[r0 : r0 + h, c0 : c0 + w][ok] = q[ok]
        target = out / f"source-{source:04d}.tifxyz"
        target.mkdir(exist_ok=True)
        for i, a in enumerate("xyz"):
            tifffile.imwrite(target / f"{a}.tif", grid[..., i])
        np.save(target / "quality.npy", quality)
        first = group[0][1]
        meta = {
            "format": "tifxyz",
            "scale": first.get("scale"),
            "source": int(source),
            "tiles": len(group),
            "level": first.get("winding"),
            "turns_spanned": first.get("turns_spanned"),
            "note": "v1 level-route tiles reassembled at their block offsets; level is the source's base level",
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=1))
        report.append({"source": int(source), "tiles": len(group), "cells": int((grid[..., 2] > 0).sum())})
    for folder, _ in loose:
        target = out / f"{folder.name}.tifxyz"
        target.mkdir(exist_ok=True)
        for name in ("x.tif", "y.tif", "z.tif", "quality.npy"):
            if (folder / name).exists() and not (target / name).exists():
                os.symlink(folder / name, target / name)
        (target / "meta.json").write_text("{}")
    return {"sources": len(tiles), "loose_tiles": len(loose), "largest": sorted(report, key=lambda r: -r["cells"])[:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pieces", help="a v1 level-route pieces folder (piece-NNNN tiles)")
    parser.add_argument("out")
    args = parser.parse_args()
    print(json.dumps(assemble(args.pieces, args.out)))


if __name__ == "__main__":
    main()
