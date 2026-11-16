"""An ink model scored on an unlabelled scroll by reading each piece through the face its fibres name."""

import json
import sys
from pathlib import Path

import numpy as np

from ink.model import tile_reader
from ink.model.stock_model import SURE
from ink.reading.piece_reading import fullest, window
from sampling.fibre_faces import gap_state, recto_face
from storage.data_roots import INK_RECORDS, SURFACE_RECORDS, TARGET_STACKS

SAMPLE = "PHerc0211"


def faces_of(sample=SAMPLE):
    """The recto of every kept stack of a scroll, kept beside the results once found."""
    kept = SURFACE_RECORDS / f"fibre-faces-{sample}.json"
    if kept.exists():
        return json.loads(kept.read_text())
    found = {
        f"{path.parent.name}/{path.stem}": recto_face(np.load(path, mmap_mode="r"))
        for path in sorted((TARGET_STACKS / sample).glob("*/piece-*.npy"))
    }
    kept.write_text(json.dumps(found, indent=0))
    return found


def sure_on_faces(reader, cube):
    """Sure ink pixels read each way and the pixels counted, over a stack's fullest windows where the gaps are open."""
    spots, side = fullest(cube[cube.shape[0] // 2] > 0)
    blocks = [np.ascontiguousarray(cube[:, y : y + side, x : x + side]) for y, x in spots]
    where = [
        (block[block.shape[0] // 2] > 0) & (gap_state(block) == 0) & (gap_state(block[::-1]) == 0) for block in blocks
    ]
    sure = {"pixels": int(sum(w.sum() for w in where))}
    for name, stack in (("near", window(cube)), ("far", window(cube[::-1]))):
        reader.fix_to(stack)
        reads = [reader.read(np.ascontiguousarray(stack[:, y : y + side, x : x + side]), fixed=True) for y, x in spots]
        sure[name] = int(sum((read[w] >= SURE).sum() for read, w in zip(reads, where)))
    return sure


def scored(rows):
    pixels = sum(r["pixels"] for r in rows)
    if not pixels:
        return {"pieces": len(rows), "pixels": 0}
    return {
        "pieces": len(rows),
        "pixels": pixels,
        "forward": round(sum(r["forward"] for r in rows) / pixels, 4),
        "reversed": round(sum(r["reversed"] for r in rows) / pixels, 4),
    }


def run(weights, sample=SAMPLE):
    """Forward against reversed on pieces read as rolled, and on the pieces turned the other way."""
    reader = tile_reader.TileReader(weights)
    rows = []
    for name, face in faces_of(sample).items():
        if face["face"] == "unknown":
            continue
        sure = sure_on_faces(reader, np.load(TARGET_STACKS / sample / f"{name}.npy"))
        rows.append(
            {
                "piece": name,
                "face": face["face"],
                "polarity": face["polarity"],
                "pixels": sure["pixels"],
                "forward": sure[face["face"]],
                "reversed": sure["far" if face["face"] == "near" else "near"],
            }
        )
    result = {
        "weights": Path(weights).name,
        "as rolled": scored([r for r in rows if r["face"] == "near"]),
        "turned": scored([r for r in rows if r["face"] == "far"]),
        "rows": rows,
    }
    (INK_RECORDS / f"fibre-polarity-{sample}-{Path(weights).stem}.json").write_text(json.dumps(result, indent=0))
    print("as rolled", result["as rolled"], "turned", result["turned"], flush=True)
    return result


if __name__ == "__main__":
    run(*sys.argv[1:])
