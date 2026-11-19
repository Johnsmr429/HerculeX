"""Where each kind of HerculeX data lives.

Two folders under one data root: LIBRARY holds what the pipeline reads and knows (catalogue, model weights, datasets,
hand marks, controls), OUTPUTS what it makes (sheet nodes, pieces, ink maps, the website). By default both sit in
`data/` beside the code; HERCULEX_DATA moves the root, HERCULEX_LIBRARY and HERCULEX_OUTPUTS put either folder anywhere
(another drive, say). Paths are resolved, so a symlinked `data/library` or `data/outputs` names its real place.
"""

import os
from pathlib import Path


def named_folder(parent, name, previous):
    """Use the short name, with a fallback for older datasets and run roots."""
    current = parent / name
    old = parent / previous
    return old if not current.exists() and not current.is_symlink() and old.exists() else current


HERE = Path(__file__).resolve().parent.parent
RUNTIME = HERE / "runtime"
SCRATCH = RUNTIME / "scratch"
KERNELS = RUNTIME / "kernels"
BROWSER_ASSETS = HERE / "website" / "assets"
RECORDS = HERE / "records"
INK_RECORDS = RECORDS / "ink"
SURFACE_RECORDS = RECORDS / "surface"


def root(name, *older, default):
    """A data folder from its environment variable (or an older name of it), else the default, resolved."""
    chosen = next((os.environ[n] for n in (name, *older) if os.environ.get(n)), None)
    return Path(chosen).expanduser().resolve() if chosen else default.resolve()


DATA = root("HERCULEX_DATA", default=HERE / "data")
LIBRARY = root("HERCULEX_LIBRARY", "HERCULEX_DISC_PCIE", default=DATA / "library")
OUTPUTS = root("HERCULEX_OUTPUTS", "HERCULEX_DISC_PCLE3", default=DATA / "outputs")
DISC_PCIE, DISC_PCLE3 = LIBRARY, OUTPUTS  # older names, kept for scripts that still use them

SCAN_CATALOG = LIBRARY / "catalog"
RAW_SCANS = LIBRARY / "raw"
OPEN_DATA_MIRROR = LIBRARY / "published" / "open_data"
DL_ASH2TXT_MIRROR = LIBRARY / "published" / "dl_ash2txt"

INK_TRAINING_DATA = LIBRARY / "datasets"
LABELLED_CASES = INK_TRAINING_DATA / "cases"
INFRARED_FRAGMENTS = INK_TRAINING_DATA / "fragments"
HAND_MARKS = LIBRARY / "marks"
OWN_SURFACE_HOLDOUT = INK_TRAINING_DATA / "holdout"

MODEL_WEIGHTS = LIBRARY / "weights"
INK_WEIGHTS = MODEL_WEIGHTS / "ink"
SURFACE_WEIGHTS = MODEL_WEIGHTS / "surface"
NEURAL_ENGINE_BUILDS = MODEL_WEIGHTS / "engine"
EXTERNAL_TOOLS = LIBRARY / "tools"

SHEET_CONTROLS = LIBRARY / "controls" / "sheets"
REFERENCE_SURFACES = LIBRARY / "controls" / "surfaces"
SURFACE_PREDICTIONS = OUTPUTS / "unwrapping" / "surface"
CALIBRATION_OUTPUTS = LIBRARY / "calibration"
STUDIES = LIBRARY / "studies"
SPOT_CANDIDATES = STUDIES / "shared" / "spot-candidates"
SUBMISSIONS = LIBRARY / "submissions"

UNWRAP_OUTPUTS = OUTPUTS / "unwrapping"
# the first unwrapping route, retired: its pieces and pages stay for the pages published from them,
# and every scan's sheet nodes, the level route's input, stay where it put them
NODE_ROUTE = UNWRAP_OUTPUTS / "nodes"
DETECTION_WORK = UNWRAP_OUTPUTS / "detection-work"
LEVEL_ROUTE = UNWRAP_OUTPUTS / "level"
ROUTE_OUTPUTS = LEVEL_ROUTE
ML_OUTPUTS = UNWRAP_OUTPUTS / "network"

NETWORK_TRAINING = UNWRAP_OUTPUTS / "network-training"
WINDING_LABELS = NETWORK_TRAINING / "labels"
WINDING_MODELS = NETWORK_TRAINING / "models"
WINDING_WEIGHTS = Path(os.environ.get("HERCULEX_NETWORK_WEIGHTS", WINDING_MODELS / "net" / "best.npy"))
SURFACE_BENCH = OUTPUTS / "bench"
CT_CHUNKS = root("HERCULEX_CHUNKS", default=OUTPUTS / "cache" / "chunks")
OPEN_DATA_EXTRA = OUTPUTS / "cache" / "open_data"
STOCK_INK_MODEL = "reader-v3"
INK_MODEL = os.environ.get("HERCULEX_INK_MODEL", STOCK_INK_MODEL)
PRODUCTION_FACES = "level-0" if INK_MODEL == STOCK_INK_MODEL else f"level-0-{INK_MODEL}"
FACES = os.environ.get("HERCULEX_FACES", PRODUCTION_FACES)
INK_LAYER_STACKS = OUTPUTS / "stacks"
WEBSITE = OUTPUTS / "website"
SHEETS = WEBSITE / "sheets"
CIRCLE_REVIEW = SHEETS / "circles-PHerc0211" / "review"  # the circles page's depth and paper images
SITE_ATLAS = WEBSITE / "atlas"


def route_id(route):
    """The level route's key in marks folders, its index and its kept stacks."""
    return f"{route.name}_route"


ROUTE_DATABASE = UNWRAP_OUTPUTS / f"{route_id(ROUTE_OUTPUTS)}.sqlite"


def scan_folder(sample, volume, *, route=ROUTE_OUTPUTS):
    """Where every stage writes its output for one scan of one scroll."""
    return route / sample / volume


def stage_folder(scan, stage):
    """One kind of artifact in a scan."""
    return scan / stage


def stage_entries(route, stage, pattern, *, sample="*"):
    """Find a scan's stage entries without counting compatibility links twice."""
    return (path for scan in route.glob(f"{sample}/*") for path in stage_folder(scan, stage).glob(pattern))


def ct_bricks(sample, store):
    """CT bricks of one scroll fetched for a stage, removed once read."""
    if override := os.environ.get("HERCULEX_BRICKS"):
        return Path(override) / sample / "ct_bricks" / store
    return LIBRARY / "cache" / "bricks" / sample / store


def calibration_output(name):
    """The folder of one calibration study's outputs."""
    return CALIBRATION_OUTPUTS / name
