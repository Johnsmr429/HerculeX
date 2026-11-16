"""The training recipe begun from Reader v3 weights, scored beside the stock-begun model."""

import hashlib
import json
import sys
import time

from ink.model import stock_model
from ink.training import model_training
from ink.training.left_out_training_runs import RECORD, scored
from ink.training.model_training import taught
from storage.data_roots import OWN_MESH_LABELS, PRETRAINED_WEIGHTS, TRAINED_WEIGHTS

READER_V3 = PRETRAINED_WEIGHTS / "reader_v3" / "reader-v3.pth"
READER_V3_SHA256 = "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a"
STARTING_WEIGHTS = READER_V3.with_name("reader-v3-under-stock-config.pth")
STOCK_SEED = "hybrid_3d2d-seed42/step-075000.pth"
NAME, STOCK_BEGUN, LEFT_OUT, SEED, STEPS = "reader-start-without-0841", "e10-stem-0841", "PHerc0841", 42, 5000
ALL_SCROLLS, TARGET, FURTHER_STEPS = "reader-start-all-scrolls", "PHerc0211", 3000
OWN_MESH, OWN_MESH_ALL_SCROLLS, OWN_MESH_SHARE, OWN_MESH_STEPS = (
    "reader-own-mesh-without-0841",
    "reader-own-mesh-all-scrolls",
    0.75,
    2000,
)
MORE_STEPS = "reader-more-steps-without-0841"
STROKE_CORE = "reader-stroke-core-without-0841"
OWN_READS, OWN_READ_SHARE, HELD_OUT_SEGMENT = "reader-own-reads-without-0841", 0.5, "20260221022814"
OWN_READS_STROKE_CORE = "reader-own-reads-stroke-core-without-0841"
OWN_READ_FOLDERS = [OWN_MESH_LABELS / route / LEFT_OUT for route in ("level_route", "grown_route")]


def in_stock_file(weights, path):
    """Weights in the stock seed's file, under the config the pinned inference code reads, which loads wrong names silently."""
    import torch

    stock = torch.load(stock_model.checkpoint(STOCK_SEED), map_location="cpu", weights_only=False)
    if {k: v.shape for k, v in weights.items()} != {k: v.shape for k, v in stock["model"].items()}:
        raise ValueError("These weights are not the stock network")
    torch.save(stock | {"model": weights}, path)
    return path


def starting_weights():
    """Reader v3's weights, checked against their published digest."""
    import torch

    with READER_V3.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != READER_V3_SHA256:
            raise ValueError(f"Checkpoint differs from its published digest: {READER_V3}")
    return in_stock_file(torch.load(READER_V3, map_location="cpu", weights_only=True)["model"], STARTING_WEIGHTS)


def taught_model_as_start(path):
    """A taught model's weights as a run's start."""
    import torch

    taught_weights = torch.load(path, map_location="cpu", weights_only=True)
    return in_stock_file(
        {k.removeprefix("model."): v for k, v in taught_weights.items()},
        STARTING_WEIGHTS.with_name(f"{path.stem}-under-stock-config.pth"),
    )


def taught_from(start, *arguments):
    """A model taught from these weights in place of the stock seed."""
    published = stock_model.checkpoint
    stock_model.checkpoint = lambda name: start
    try:
        return taught(*arguments)
    finally:
        stock_model.checkpoint = published


def taught_from_reader():
    """The model, taught once."""
    path = TRAINED_WEIGHTS / f"{NAME}.pth"
    if not path.exists():
        taught_from(starting_weights(), NAME, LEFT_OUT, LEFT_OUT, SEED, STEPS)
    return path


def all_scrolls():
    """The left-out model taught further with PHerc0841 among its scrolls and the target's papyrus as blank."""
    print(
        taught_from(
            taught_model_as_start(TRAINED_WEIGHTS / f"{NAME}.pth"), ALL_SCROLLS, TARGET, "none", SEED, FURTHER_STEPS
        ),
        flush=True,
    )


def run():
    began = time.time()
    path = taught_from_reader()
    minutes = round((time.time() - began) / 60, 1)
    for name, weights in ((NAME, path), (STOCK_BEGUN, TRAINED_WEIGHTS / f"{STOCK_BEGUN}.pth")):
        score = scored(weights, LEFT_OUT)
        with RECORD.open("a") as log:
            log.write(
                json.dumps(
                    {
                        "name": name,
                        "steps": STEPS,
                        "left_out": LEFT_OUT,
                        "begun_from": "reader-v3" if name == NAME else "stock",
                        "minutes": minutes if name == NAME else None,
                    }
                    | score
                )
                + "\n"
            )
        print(
            name,
            "AUC",
            round(score["auc2"], 4),
            "mean of windows",
            round(score["auc"], 4),
            "own mesh",
            score.get("own_mesh"),
            score.get("own_mesh_all"),
            flush=True,
        )


def own_mesh():
    """The left-out model taught further on hand labels carried onto our own pieces, PHerc0841 still left out, so that it has a fair score."""
    print(
        taught_from(
            taught_model_as_start(TRAINED_WEIGHTS / f"{NAME}.pth"),
            OWN_MESH,
            LEFT_OUT,
            LEFT_OUT,
            SEED,
            OWN_MESH_STEPS,
            OWN_MESH_SHARE,
        ),
        flush=True,
    )


def more_steps():
    """The control for own_mesh."""
    print(
        taught_from(
            taught_model_as_start(TRAINED_WEIGHTS / f"{NAME}.pth"), MORE_STEPS, LEFT_OUT, LEFT_OUT, SEED, OWN_MESH_STEPS
        ),
        flush=True,
    )


def as_own_mesh(name):
    """A model taught as own_mesh is, own_mesh being its control."""
    start = taught_model_as_start(TRAINED_WEIGHTS / f"{NAME}.pth")
    print(taught_from(start, name, LEFT_OUT, LEFT_OUT, SEED, OWN_MESH_STEPS, OWN_MESH_SHARE), flush=True)


def stroke_core():
    """own_mesh with every ink target thinned to its strokes' cores."""
    model_training.STROKE_CORE = 1
    as_own_mesh(STROKE_CORE)


def own_reads():
    """own_mesh with half its patches from stock reads of our own pieces of the left-out scroll, one segment held out, no label of the scroll seen."""
    model_training.OWN_READS = (OWN_READ_SHARE, OWN_READ_FOLDERS, HELD_OUT_SEGMENT)
    as_own_mesh(OWN_READS)


def own_reads_stroke_core():
    """own_reads with every ink target thinned to its strokes' cores."""
    model_training.OWN_READS, model_training.STROKE_CORE = (OWN_READ_SHARE, OWN_READ_FOLDERS, HELD_OUT_SEGMENT), 1
    as_own_mesh(OWN_READS_STROKE_CORE)


def own_mesh_all_scrolls():
    """The same teaching from the model of every labelled scroll, to read with."""
    print(
        taught_from(
            taught_model_as_start(TRAINED_WEIGHTS / f"{ALL_SCROLLS}.pth"),
            OWN_MESH_ALL_SCROLLS,
            TARGET,
            "none",
            SEED,
            OWN_MESH_STEPS,
            OWN_MESH_SHARE,
        ),
        flush=True,
    )


if __name__ == "__main__":
    {
        "all-scrolls": all_scrolls,
        "own-mesh": own_mesh,
        "more-steps": more_steps,
        "stroke-core": stroke_core,
        "own-reads": own_reads,
        "own-reads-stroke-core": own_reads_stroke_core,
        "own-mesh-all-scrolls": own_mesh_all_scrolls,
    }.get((sys.argv[1:] or [""])[0], run)()
