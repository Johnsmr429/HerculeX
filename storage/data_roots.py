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
LABELLED_CASES = named_folder(INK_TRAINING_DATA, "cases", "labelled_cases")
LABELLED_CASE_STACKS = named_folder(INK_TRAINING_DATA, "stacks", "labelled_case_stacks")
INFRARED_FRAGMENTS = named_folder(INK_TRAINING_DATA, "fragments", "infrared_fragments")
PUBLISHED_PAIRS = named_folder(INK_TRAINING_DATA, "published", "published_pairs")
PUBLISHED_PAIRS_WRITING = named_folder(INK_TRAINING_DATA, "writing", "published_pairs_writing")
OWN_MESH_LABELS = named_folder(INK_TRAINING_DATA, "meshes", "own_mesh_labels")
OWN_MESH_FINE_LETTERS = named_folder(INK_TRAINING_DATA, "letters", "own_mesh_fine_letters")
TARGET_STACKS = named_folder(INK_TRAINING_DATA, "targets", "target_stacks")
HAND_MARKS = named_folder(LIBRARY, "marks", "datasets/hand_marks")
LETTER_SPOTS = named_folder(INK_TRAINING_DATA, "spots", "letter_spots")
AGREED_TARGETS = named_folder(INK_TRAINING_DATA, "consensus", "agreed_targets")
DEPTH_MEAN_TARGETS = named_folder(INK_TRAINING_DATA, "distilled", "depth_mean_targets")
OWN_MESH_SMOOTHING = named_folder(INK_TRAINING_DATA, "smoothing", "own_mesh_smoothing")
OWN_MESH_THIN_MARKS = named_folder(INK_TRAINING_DATA, "thin", "own_mesh_thin_marks")
OWN_SURFACE_HOLDOUT = named_folder(INK_TRAINING_DATA, "holdout", "own_surface_holdout")

MODEL_WEIGHTS = LIBRARY / "weights"
PRETRAINED_WEIGHTS = MODEL_WEIGHTS / "pretrained"
PRODUCTION_WEIGHTS = MODEL_WEIGHTS / "production"
READER_SEED42 = named_folder(PRODUCTION_WEIGHTS, "reader-seed42.pth", "seed42.pth")
READER_SEED43 = named_folder(PRODUCTION_WEIGHTS, "reader-seed43.pth", "seed43.pth")
TRAINED_WEIGHTS = MODEL_WEIGHTS / "trained"
NEURAL_ENGINE_BUILDS = named_folder(MODEL_WEIGHTS, "engine", "neural_engine_builds")
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
GROWN_ROUTE = named_folder(UNWRAP_OUTPUTS, "grown", "grown_route")
NODE_ROUTE = GROWN_ROUTE
LEVEL_ROUTE = named_folder(UNWRAP_OUTPUTS, "level", "level_route")
ROUTE_OUTPUTS = LEVEL_ROUTE
PRODUCTION_FACES = "level-0"
FACES = os.environ.get("HERCULEX_FACES", PRODUCTION_FACES)
INK_LAYER_STACKS = OUTPUTS / "stacks"
WEBSITE = OUTPUTS / "website"
SHEETS = WEBSITE / "sheets"
CIRCLE_REVIEW = SHEETS / "circles-PHerc0211" / "review"  # the circles page's depth and paper images
SITE_ATLAS = WEBSITE / "atlas"
BLIND_PAIRS = named_folder(WEBSITE, "blind", "blind_pairs")

STAGE_NAMES = {"nodes": "sheet-nodes", "faces": "papyrus-faces", "sheet": "papyrus-sheet"}
RETENTION_ROUTES = {"surface_ink": "grown", "surface_ink_fresh": "fresh", "surface_ink_level": "level"}


def route_id(route):
    """Keep the route keys stored in dataset records and annotation logs."""
    return f"{route.name}_route" if route.name in ("grown", "level") else route.name


ROUTE_DATABASE = UNWRAP_OUTPUTS / f"{route_id(ROUTE_OUTPUTS)}.sqlite"


def scan_folder(sample, volume, *, route=ROUTE_OUTPUTS):
    """Where every stage writes its output for one scan of one scroll."""
    return route / sample / volume


def stage_folder(scan, stage):
    """One kind of artifact in a scan, including the former on-disk names."""
    return named_folder(scan, stage, STAGE_NAMES.get(stage, stage))


def stage_entries(route, stage, pattern, *, sample="*"):
    """Find a scan's stage entries without counting compatibility links twice."""
    return (path for scan in route.glob(f"{sample}/*") for path in stage_folder(scan, stage).glob(pattern))


def ct_bricks(sample, store):
    """CT bricks of one scroll fetched for a stage, removed once read."""
    if override := os.environ.get("HERCULEX_BRICKS"):
        return Path(override) / sample / "ct_bricks" / store
    return named_folder(LIBRARY, f"cache/bricks/{sample}", f"raw/{sample}/ct_bricks") / store


def calibration_output(name):
    """The folder of one calibration study's outputs."""
    if name in RETENTION_ROUTES:
        return named_folder(CALIBRATION_OUTPUTS, f"retention/{RETENTION_ROUTES[name]}", name)
    return CALIBRATION_OUTPUTS / name
