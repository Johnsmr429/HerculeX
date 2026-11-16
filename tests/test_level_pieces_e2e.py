"""The level route end to end on crushed and flattened scrolls of known windings, and its solver against brute force."""

import itertools
import json

import numpy as np
import pytest
import tifffile

from calibration.unwrap.synthetic_scroll import build, true_turn
from runner.stage_queue import record_of
from unwrapping.detection.detect_nodes import detect_store
from unwrapping.level import kernel_bindings as kb
from unwrapping.level import level_pieces, piece_flattening, run_levels
from unwrapping.level.piece_flattening import MIN_PIECE_NODES
from unwrapping.level.piece_kernel_bindings import EDGE, PAIR
from unwrapping.level.run_levels import extract, piece_name

VOXEL_UM = 18.7


@pytest.fixture(scope="module")
def crushed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("level")
    record = build(folder / "level", gap=7.0, half_flat=150.0, turns=16, height=128, noise=8.0, wander=0.1, fading=0.2)
    detect_store(
        folder / "level", folder / "nodes", {"voxel_um": VOXEL_UM, "scan_voxel_um": VOXEL_UM, "level": 0}, device="cpu"
    )
    return folder, record


@pytest.fixture(scope="module")
def flattened(tmp_path_factory):
    folder = tmp_path_factory.mktemp("flattened")
    record = build(folder / "level", gap=7.0, half_flat=400.0, turns=10, height=96, noise=8.0, wander=0.1, fading=0.2)
    detect_store(
        folder / "level", folder / "nodes", {"voxel_um": VOXEL_UM, "scan_voxel_um": VOXEL_UM, "level": 0}, device="cpu"
    )
    return folder, record


def exact_share(record, nodes, theta, root, potential):
    """Share of the largest piece's nodes whose level matches the true winding, in the better of the two senses."""
    sizes = np.bincount(root, minlength=len(nodes))
    keep = sizes[root] >= MIN_PIECE_NODES
    k, f = true_turn(record, nodes["x"], nodes["y"])
    largest = root == np.bincount(root[keep]).argmax()
    shares = []
    for sense in (1, -1):
        offset = np.rint((potential + theta / (2 * np.pi))[largest] * sense - (k + f)[largest]).astype(np.int64)
        shares.append(np.unique(offset, return_counts=True)[1].max() / largest.sum())
    return keep.mean(), largest.mean(), max(shares)


def test_solver_equals_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        count, terms = 6, 12
        p = rng.integers(0, count, terms).astype(np.int32)
        q = (p + rng.integers(1, count, terms)).astype(np.int32) % count
        k = rng.integers(-2, 3, terms).astype(np.int32)
        up, dn = rng.integers(1, 9, (2, terms)).astype(np.int32)
        x, log = kb.minimize(count, p, q, k, up, dn, np.zeros(count, dtype=np.int32))

        def energy(v):
            r = v[q] - v[p] - k
            return int(np.where(r > 0, r * up, -r * dn).sum())

        best = min(energy(np.array((0, *v))) for v in itertools.product(range(-6, 7), repeat=count - 1))
        assert energy(x) == best == log[2]


def test_a_join_of_many_stays_and_a_join_of_one_moves():
    patch = np.arange(3, dtype=np.int32).repeat(100)
    inner = np.zeros(300, dtype=np.int32)
    theta = np.zeros(300, dtype=np.float32)
    edges = np.zeros(101, dtype=EDGE)
    edges["a"], edges["b"], edges["weight"] = np.r_[np.arange(100), 0], np.r_[np.arange(100, 200), 200], 1.0
    rays = np.zeros(1, dtype=PAIR)
    rays["a"], rays["b"] = 1, 201
    ray_weight = np.ones(1, dtype=np.float32)
    p, q, k, term = kb.level_terms(patch, inner, 3, edges, rays, ray_weight, theta, 1)
    up, dn = kb.term_weights(len(p), term, edges, ray_weight, 0.5, 64)
    assert len(p) == 3 and up.max() == 6400
    x, _ = kb.minimize(3, p, q, k, up, dn, np.zeros(3, dtype=np.int32))
    votes = kb.votes(3, p, q, k, term, edges, ray_weight, 0.5, x, 16, 2, 64, 1)
    strong = np.flatnonzero(up == 6400)[0]
    assert votes[strong] == 16 and votes.min() < 16


def test_level_pieces_stay_on_their_sheet(crushed):
    folder, record = crushed
    field_record = record_of(folder / "nodes")
    nodes, theta, _, _, pairs, field, _ = level_pieces.prepare([folder / "nodes"])
    assert field["orientation"]["turned_share"] < 0.05
    root, potential, _a, _b, _between, _ = level_pieces.grow_pieces(nodes, theta, pairs, field)
    sizes = np.bincount(root, minlength=len(nodes))
    keep = sizes[root] >= MIN_PIECE_NODES
    assert keep.mean() > 0.8
    k, f = true_turn(record, nodes["x"], nodes["y"])
    largest = root == np.bincount(root[keep]).argmax()
    shares = []
    for sense in (1, -1):
        offset = np.rint((potential + theta / (2 * np.pi))[largest] * sense - (k + f)[largest]).astype(np.int64)
        shares.append(np.unique(offset, return_counts=True)[1].max() / largest.sum())
    assert max(shares) > 0.95, shares
    assert field_record["level"] == 0


def test_a_round_enough_scroll_keeps_its_axis(crushed):
    folder, _ = crushed
    *_, field, _ = level_pieces.prepare([folder / "nodes"])
    assert field["centres"]["rule"] == "axis"


def test_a_flattened_scroll_is_read_about_its_core(flattened):
    folder, record = flattened
    nodes, theta, _, _, pairs, field, _ = level_pieces.prepare([folder / "nodes"])
    centres = field["centres"]
    assert centres["rule"] == "segment", centres
    assert abs(centres["core_mm"][0] - 2 * 400.0 * VOXEL_UM / 1000) < 1.5, centres
    root, potential, _a, _b, _between, _ = level_pieces.grow_pieces(nodes, theta, pairs, field)
    kept, largest, exact = exact_share(record, nodes, theta, root, potential)
    assert kept > 0.8 and largest > 0.6, (kept, largest)
    assert exact > 0.95, exact


def test_level_grid_keeps_lengths(crushed, tmp_path):
    folder, _ = crushed
    summary, notes = extract([folder / "nodes"], tmp_path, "sample", "volume", VOXEL_UM, 0)
    assert summary["method"] == run_levels.METHOD and any(tmp_path.glob("piece-*"))
    assert notes["piece_terms"]["chirality"] == summary["chirality"]
    assert summary["isometry"]["isometric_share"] >= piece_flattening.UNIT_ISOMETRIC_SHARE
    top = max(summary["pieces"], key=lambda p: p["area_mm2"])
    surface = tmp_path / piece_name(top["piece"])
    grid = np.stack([tifffile.imread(surface / f"{a}.tif") for a in "xyz"])
    locked = np.load(surface / "quality.npy") == 1
    stride = json.loads((surface / "meta.json").read_text())["stride"]
    steps = []
    for axis in (1, 2):
        length = np.linalg.norm(np.diff(grid, axis=axis), axis=0) / stride
        steps.append(length[locked[1:, :] & locked[:-1, :]] if axis == 1 else length[locked[:, 1:] & locked[:, :-1]])
    steps = np.concatenate(steps)
    assert abs(np.median(steps) - 1) < 0.02
    assert np.mean(np.abs(steps - 1) <= 0.1) > 0.97, "neighbouring nodes a grid step apart in 3D: an isometric grid"


def test_a_filled_cell_just_past_the_scan_is_left_out_and_a_grid_far_past_it_stops(tmp_path):
    meta = {"shape": [50, 60, 70]}
    grid = np.stack(np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")).astype(np.float64)
    grid = np.stack([grid[1] * 5 + 1, grid[0] * 5 + 1, np.full((3, 4), 10.0)])
    known, locked = np.ones((3, 4), dtype=bool), np.zeros((3, 4), dtype=bool)
    edge = grid.copy()
    edge[0, 0, 0] = -1.0  # x = -2 at scale 2: a filled cell a voxel past the edge
    piece_flattening.write_piece(tmp_path / "edge", edge, known, locked, 2, meta)
    x = tifffile.imread(tmp_path / "edge" / "x.tif")
    assert x[0, 0] == -1 and (x[known & (np.arange(12).reshape(3, 4) > 0)] >= 0).all()
    far = grid.copy()
    far[0, 0, 0] = -20.0
    with pytest.raises(ValueError, match="leaves the scan"):
        piece_flattening.write_piece(tmp_path / "far", far, known, locked, 2, meta)


def test_a_unit_from_before_the_isometric_grids_is_out_of_date_whatever_digest_is_listed(tmp_path):
    """A digest proven to give the same output can outlive a later change of output; the isometry record cannot."""
    from runner import stage_queue
    from unwrapping.level.piece_flattening import UNIT_NEEDS

    unit = tmp_path / "rows-0000-0012"
    unit.mkdir()
    (unit / stage_queue.RECORD).write_text(json.dumps({"method": "57e06326f8bd"}))
    assert stage_queue.done(unit, "new", ("57e06326f8bd",))
    assert not stage_queue.done(unit, "new", ("57e06326f8bd",), UNIT_NEEDS)
    (unit / stage_queue.RECORD).write_text(json.dumps({"method": run_levels.METHOD, "isometry": {"all": {}}}))
    assert stage_queue.done(unit, run_levels.METHOD, run_levels.SAME_OUTPUT, UNIT_NEEDS)
    old = {
        "57e06326f8bd",
        "f36042e9f4d2",
        "81df5578dc29",
        "f50a3a345fc7",
        "60ff925f69b6",
        "c7e5a6feda3d",
        "99a2ef30cdbe",
        "8ed2b035cf4a",
    }
    assert not old & set(run_levels.SAME_OUTPUT)
