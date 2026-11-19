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
    count, terms = 6, 12
    every = np.array(list(itertools.product(range(-6, 7), repeat=count - 1)), dtype=np.int32)
    every = np.hstack([np.zeros((len(every), 1), dtype=np.int32), every])
    for _ in range(30):
        p = rng.integers(0, count, terms).astype(np.int32)
        q = (p + rng.integers(1, count, terms)).astype(np.int32) % count
        k = rng.integers(-2, 3, terms).astype(np.int32)
        up, dn = rng.integers(1, 9, (2, terms)).astype(np.int32)
        x, log = kb.minimize(count, p, q, k, up, dn, np.zeros(count, dtype=np.int32))

        def energy(v):
            r = v[..., q] - v[..., p] - k
            return np.where(r > 0, r * up, -r * dn).sum(axis=-1)

        assert int(energy(x)) == int(energy(every).min()) == log[2]


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


def test_a_ray_s_own_skip_cost_decides_a_skip():
    patch = np.arange(2, dtype=np.int32).repeat(100)
    inner = np.zeros(200, dtype=np.int32)
    theta = np.zeros(200, dtype=np.float32)
    edges = np.zeros(1, dtype=EDGE)
    edges["a"], edges["b"], edges["weight"], edges["step"] = 0, 100, 0.6, 2
    rays = np.zeros(1, dtype=PAIR)
    rays["a"], rays["b"] = 1, 101
    ray_weight = np.ones(1, dtype=np.float32)
    p, q, k, term = kb.level_terms(patch, inner, 2, edges, rays, ray_weight, theta, 1)

    def gap(ray_up):
        up, dn = kb.term_weights(len(p), term, edges, ray_weight, 0.5, 64, ray_up)
        x, _ = kb.minimize(2, p, q, k, up, dn, np.zeros(2, dtype=np.int32))
        votes = kb.votes(2, p, q, k, term, edges, ray_weight, 0.5, x, 4, 1, 64, 1, False, 0, ray_up)
        assert votes.max() == 4
        return int(x[1] - x[0])

    assert gap(None) == 2
    assert gap(np.ones(1, dtype=np.float32)) == 1


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


def test_a_refused_piece_is_cut_at_its_doubled_cells_not_dropped(crushed, monkeypatch):
    folder, record = crushed
    nodes, theta, _, _, pairs, field, _ = level_pieces.prepare([folder / "nodes"])
    real = level_pieces.refused_pieces
    marked = {}

    def largest_refused_with_a_doubled_band(nodes_, root, potential, attract, area_node, cells=False):
        if marked:
            return real(nodes_, root, potential, attract, area_node, cells)
        largest = root == np.bincount(root).argmax()
        band = largest & (nodes_["z"] >= nodes_["z"][largest].max() - 4)
        marked.update(rest=largest & ~band, band=band)
        return largest, band, []

    monkeypatch.setattr(level_pieces, "refused_pieces", largest_refused_with_a_doubled_band)
    monkeypatch.setattr(level_pieces, "GUARD_CUT", 1)
    root, potential, *_ = level_pieces.grow_pieces(nodes, theta, pairs, field)
    rest, band = marked["rest"], marked["band"]
    kept_root = np.bincount(root[rest]).argmax()
    assert band.any() and (root[rest] == kept_root).mean() > 0.95
    assert (root[band] == kept_root).mean() < 0.05
    k, f = true_turn(record, nodes["x"], nodes["y"])
    piece = root == kept_root
    shares = [
        np.unique(np.rint((potential + theta / (2 * np.pi))[piece] * sense - (k + f)[piece]), return_counts=True)[
            1
        ].max()
        / piece.sum()
        for sense in (1, -1)
    ]
    assert max(shares) > 0.95, shares


def test_a_doubled_cell_keeps_the_layer_its_neighbours_continue(monkeypatch):
    from unwrapping.detection.kernel_bindings import CELL, NODE

    side = 12
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    flat = np.zeros(side * side + 1, dtype=NODE)
    flat["x"][:-1] = (xx.ravel() + 0.5) * CELL
    flat["y"][:-1] = (yy.ravel() + 0.5) * CELL
    flat["z"][:-1] = 40.0
    flat["x"][-1], flat["y"][-1], flat["z"][-1] = (6.5 * CELL, 6.5 * CELL, 46.0)
    flat["nz"] = 127
    normal = np.tile([0.0, 0.0, 1.0], (len(flat), 1))
    u, v = flat["x"].astype(np.float64), flat["y"].astype(np.float64)
    grid, locked, _ = piece_flattening.rasterise(flat, normal, u, v)
    assert not locked[6, 6] and locked.sum() == side * side - 1
    monkeypatch.setattr(level_pieces, "GUARD_CUT", 1)
    dropped_all = level_pieces.second_layer(flat, normal, u, v, grid, locked)
    monkeypatch.setattr(level_pieces, "GUARD_CUT", 2)
    dropped = level_pieces.second_layer(flat, normal, u, v, grid, locked)
    on_sheet = side * 6 + 6
    assert dropped_all[on_sheet] and dropped_all[-1]
    assert not dropped[on_sheet] and dropped[-1] and dropped.sum() == 1


def test_a_piece_is_turned_so_the_scroll_axis_runs_up_its_rows():
    from unwrapping.detection.kernel_bindings import NODE

    rng = np.random.default_rng(0)
    across, up = rng.uniform(0, 100, 500), rng.uniform(0, 60, 500)
    nodes = np.zeros(500, dtype=NODE)
    nodes["x"], nodes["y"], nodes["z"] = 200 + across, 300.0, 50 + up
    turn = np.radians(37)
    u, v = across * np.cos(turn) - up * np.sin(turn), across * np.sin(turn) + up * np.cos(turn)
    gu, gv = piece_flattening.upright(nodes, u, v, np.zeros(500), np.full(500, 100.0))
    slope = np.linalg.lstsq(np.c_[gu, gv, np.ones(500)], nodes["z"].astype(np.float64), rcond=None)[0]
    assert abs(slope[0]) < 1e-6 and abs(slope[1] - 1) < 1e-6


def test_every_written_piece_stands_upright(crushed, tmp_path):
    folder, _ = crushed
    summary, _notes = extract([folder / "nodes"], tmp_path, "sample", "volume", VOXEL_UM, 0)
    leaning = []
    for piece in tmp_path.glob("piece-*"):
        grid = np.stack([tifffile.imread(piece / f"{a}.tif") for a in "xyz"])
        rows, cols = np.nonzero(np.all(grid > 0, axis=0))
        slope = np.linalg.lstsq(np.c_[rows, cols, np.ones(len(rows))], grid[2][rows, cols], rcond=None)[0]
        if np.degrees(np.arctan2(abs(slope[1]), abs(slope[0]))) > 10:
            leaning.append(piece.name)
    assert summary["pieces"] and not leaning, leaning


def test_every_piece_is_written_however_small(crushed, tmp_path):
    folder, _ = crushed
    summary, _notes = extract([folder / "nodes"], tmp_path, "sample", "volume", VOXEL_UM, 0)
    small = [p for p in summary["pieces"] if p["area_mm2"] < 25 and p["cells"] < MIN_PIECE_NODES]
    assert small, [(p["area_mm2"], p["cells"]) for p in summary["pieces"]]
    assert all(p["written"] and (tmp_path / piece_name(p["piece"]) / "x.tif").is_file() for p in summary["pieces"])


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
    assert summary["isometry"]["isometric_share"] >= 0.8
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


def test_no_unit_from_before_the_isometric_grids_counts_as_current(tmp_path):
    """Units made before the isometric grids have other pieces, so none of their digests is listed as the same output."""
    from runner import stage_queue

    unit = tmp_path / "rows-0000-0012"
    unit.mkdir()
    (unit / stage_queue.RECORD).write_text(json.dumps({"method": "57e06326f8bd"}))
    assert not stage_queue.done(unit, run_levels.METHOD, run_levels.SAME_OUTPUT)
    (unit / stage_queue.RECORD).write_text(json.dumps({"method": run_levels.METHOD}))
    assert stage_queue.done(unit, run_levels.METHOD, run_levels.SAME_OUTPUT)
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
