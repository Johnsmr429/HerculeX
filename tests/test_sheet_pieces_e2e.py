"""Pieces flattened along the sheet, gridded, bridged and joined across a seam: on a crushed scroll of exactly known
windings and on a tilted cylinder."""

import shutil
from unittest.mock import patch

import numpy as np
import pytest

from calibration.unwrap.synthetic_scroll import build, true_winding
from runner.stage_queue import record_of
from storage.node_store import read_cells, read_nodes
from unwrapping.detection.detect_nodes import detect_store
from unwrapping.detection.kernel_bindings import CELL, NODE
from unwrapping.level import level_pieces, piece_flattening
from unwrapping.level.piece_flattening import bridge_holes, rasterise, tears
from unwrapping.level.piece_joiner import SEAM_PLANES, seam_field, seam_joins
from unwrapping.level.run_levels import extract

VOXEL_UM = 18.7


def grid_steps(grid, stride, keep=None):
    """3D distance over the grid step of every pair of neighbouring nodes along rows and columns."""
    good = (grid[0] > 0) if keep is None else keep
    out = []
    for axis in (1, 2):
        step = np.linalg.norm(np.diff(grid, axis=axis), axis=0) / stride
        both = good[1:, :] & good[:-1, :] if axis == 1 else good[:, 1:] & good[:, :-1]
        out.append(step[both])
    return np.concatenate(out)


def tilted_cylinder(radius=600.0, span=1.0, height=480.0, tilt=np.pi / 6, depth_noise=0.7, seed=3):
    """Nodes as the detector places them on a cylinder whose axis leans by tilt: one per cell the sheet crosses, at
    the sheet point nearest the cell's centre; their exact flattening (radius x angle, height); edges between
    neighbouring cells; and the sheet point of any flat place."""
    rng = np.random.default_rng(seed)
    reach = radius + 2 * CELL
    lo = np.floor(np.array([-reach, -reach, -height]) / CELL).astype(int)
    hi = np.ceil(np.array([reach, reach, 2 * height]) / CELL).astype(int)
    cells = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    turn = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    local = ((cells + 0.5) * CELL) @ turn
    theta, r = np.arctan2(local[:, 1], local[:, 0]), np.hypot(local[:, 0], local[:, 1])
    normal = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], 1)
    keep = (np.abs(theta) < span / 2) & (local[:, 2] > 0) & (local[:, 2] < height)
    keep &= np.abs(r - radius) < 0.5 * CELL * np.abs(normal @ turn.T).sum(1)
    theta, z, normal, cells = theta[keep], local[keep, 2], normal[keep], cells[keep]
    # the detector finds the sheet anywhere in its cell: each node moved along the sheet by up to half a cell, kept
    # where it stays in its cell
    moved_theta = theta + rng.uniform(-0.5, 0.5, len(z)) * CELL / radius
    moved_z = z + rng.uniform(-0.5, 0.5, len(z)) * CELL
    moved = np.stack([radius * np.cos(moved_theta), radius * np.sin(moved_theta), moved_z], 1) @ turn.T
    stays = (np.floor(moved / CELL).astype(int) == cells).all(axis=1)
    theta, z = np.where(stays, moved_theta, theta), np.where(stays, moved_z, z)
    normal = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], 1)
    depth = rng.normal(scale=depth_noise, size=len(z))[:, None]
    world = (np.stack([radius * np.cos(theta), radius * np.sin(theta), z], 1) + depth * normal) @ turn.T + 2 * reach
    nodes = np.zeros(len(z), NODE)
    nodes["x"], nodes["y"], nodes["z"] = world.T
    nodes["nx"], nodes["ny"], nodes["nz"] = np.rint((normal @ turn.T).T * 127).astype(np.int8)
    index = {tuple(c): i for i, c in enumerate(cells)}
    pairs = [(i, index[j]) for i, c in enumerate(map(tuple, cells))
             for j in ((c[0] + d[0], c[1] + d[1], c[2] + d[2]) for d in np.argwhere(np.ones((3, 3, 3))) - 1)
             if index.get(j, -1) > i]  # fmt: skip
    a, b = (np.ascontiguousarray(e, dtype=np.int32) for e in zip(*pairs))

    def sheet_at(u, v):
        """The sheet point and its normal at flat places."""
        t = u / radius
        point = np.stack([radius * np.cos(t), radius * np.sin(t), v], -1) @ turn.T + 2 * reach
        return point, np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], -1) @ turn.T

    return nodes, normal @ turn.T, radius * theta, z, a, b, sheet_at


@pytest.fixture(scope="module")
def crushed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("crushed")
    record = build(folder / "level", gap=7.0, half_flat=150.0, turns=16, height=128, noise=8.0, wander=0.1, fading=0.2)
    detect_store(
        folder / "level", folder / "nodes", {"voxel_um": VOXEL_UM, "scan_voxel_um": VOXEL_UM, "level": 0}, device="cpu"
    )
    return folder, record


def test_grids_of_cell_means_fail_the_guard():
    """The earlier rasterisation and hole fill (cell means, normalised convolution) fail a unit; the
    centred grids of the same nodes pass."""
    nodes, normal, _, _, a, b, _ = tilted_cylinder()
    _, _, sign, u, v = level_pieces.flatten_pieces(nodes, np.zeros(len(nodes), dtype=np.int32), a, b)

    def rows():
        grid, locked, _ = level_pieces.rasterise(nodes, normal * sign[:, None], u, v)
        filled, known = bridge_holes(grid, locked)
        return [{"isometry": piece_flattening.isometry(filled, known, locked, CELL)}]

    assert piece_flattening.unit_isometry(rows())["isometric_share"] == 1.0
    with (
        patch.object(piece_flattening, "centred", lambda xyz, u, v, shape, mean, take, source: mean),
        patch.object(piece_flattening, "relaxed", lambda grid, known, free, rounds=0: grid),
    ):
        assert piece_flattening.unit_isometry(rows())["isometric_share"] < 1.0


def halves(store, seam, tmp_path):
    """The pieces of each node in the shared planes of a store's two units, lower and upper, split at a seam."""
    lo, hi = record_of(store)["cell_planes"]
    labels = []
    for name, planes, shared in (
        ("lower", (lo, seam), (seam - SEAM_PLANES, seam)),
        ("upper", (seam, hi), (seam, seam + SEAM_PLANES)),
    ):
        out = tmp_path / name
        out.mkdir()
        extract([store], out, "synthetic", "0", VOXEL_UM, 0, planes=planes, shared=[shared])
        boundary = np.load(out / "boundary.npz")
        order = np.argsort(boundary["ids"])
        labels.append((boundary["ids"][order], boundary["piece"][order]))
    return labels


def test_chunks_join_across_their_seam(crushed, tmp_path):
    folder, record = crushed
    lo, hi = record_of(folder / "nodes")["cell_planes"]
    seam = (lo + hi) // 2
    labels = halves(folder / "nodes", seam, tmp_path)
    joins = seam_joins([folder / "nodes"], seam, labels)
    assert joins
    piece, side, edges, _, _, ids, _ = seam_field([folder / "nodes"], seam, labels)
    nodes = read_nodes(folder / "nodes")
    winding = true_winding(record, nodes["x"], nodes["y"], nodes["z"])[ids & 0xFFFFFFFF]
    a, b = edges["a"], edges["b"]
    lower, upper = np.where(side[a] == 0, a, b), np.where(side[a] == 0, b, a)
    for x, y, _ in joins:
        crossing = (side[a] != side[b]) & (piece[lower] == x) & (piece[upper] == y)
        assert np.median(np.abs(winding[lower[crossing]] - winding[upper[crossing]])) < 0.5, (x, y)


def test_a_seam_whose_nodes_are_all_too_weak_joins_nothing(crushed, tmp_path):
    folder, _ = crushed
    record = record_of(folder / "nodes")
    lo, hi = record["cell_planes"]
    seam = (lo + hi) // 2
    store = tmp_path / "nodes"
    store.mkdir()
    nodes, cells = read_nodes(folder / "nodes"), read_cells(folder / "nodes")
    plane = lo + cells // (record["grid"][1] * record["grid"][2])
    nodes["prominence"][(plane >= seam - SEAM_PLANES) & (plane < seam + SEAM_PLANES)] = 0
    nodes.tofile(store / "nodes.bin")
    cells.tofile(store / "cells.bin")
    shutil.copy(folder / "nodes" / "record.json", store / "record.json")
    labels = halves(store, seam, tmp_path)
    assert seam_joins([store], seam, labels) == []
    field = seam_field([store], seam, labels)
    assert all(len(part) == 0 for part in field)
    assert {"a", "b", "weight"} <= set(field[2].dtype.names)
    assert {"a", "b"} <= set(field[3].dtype.names)


def test_holes_are_bridged_but_rims_and_folds_are_not():
    side = 40
    y, x = np.mgrid[:side, :side].astype(np.float64) * CELL
    grid = np.stack([x, y, np.where(np.arange(side) < 20, 0.0, 50.0 * CELL)[None, :].repeat(side, axis=0)])
    locked = np.zeros((side, side), dtype=bool)
    locked[5:35, 5:35] = True
    locked[15:18, 10:13] = False
    filled, known = bridge_holes(grid, locked)
    assert known[15:18, 10:13].all(), "a small hole is filled"
    assert np.abs(filled[0][16, 11] - grid[0][16, 11]) < 1e-6, "on the sheet"
    assert not known[4, 5:35].any() and not known[35, 5:35].any(), "a rim does not grow"
    assert not tears(filled, known).any(), "no mesh edge spans the fold"
    assert known[:, 21:].sum() > 0 and not known[5:35, 19:21].any(), "only the cells at the fold go"


def test_holes_are_bridged_evenly_on_a_curved_sheet():
    side, radius = 60, 300.0
    row, column = np.mgrid[:side, :side].astype(np.float64)
    angle = column * CELL / radius
    grid = np.stack([radius * np.cos(angle), radius * np.sin(angle), row * CELL + 0.3 * column * CELL]) + 1000
    locked = np.ones((side, side), dtype=bool)
    for y, x, h, w in ((5, 5, 6, 6), (20, 20, 4, 9), (40, 10, 2, 2), (10, 40, 6, 3)):
        locked[y : y + h, x : x + w] = False
    filled, known = bridge_holes(grid * locked, locked)
    assert known.all(), "every hole is filled and none is torn"
    assert np.linalg.norm(filled - grid, axis=0).max() < 1.0, "the fill lies on the sheet"
    steps = grid_steps(filled, 1.0) / grid_steps(grid, 1.0)
    assert np.abs(steps - 1).max() < 0.02, "filled cells lie a grid step apart, not bunched at the rim"


def test_grid_nodes_sit_at_their_cells_centres():
    nodes, normal, u, v, _, _, sheet_at = tilted_cylinder()
    grid, locked, crowding = rasterise(nodes, normal, u, v)
    assert crowding > 1.2, "a leaning sheet crosses more than one cell per flat cell"
    rows, columns = np.nonzero(locked)
    true, across = sheet_at(u.min() + (columns + 0.5) * CELL, v.min() + (rows + 0.5) * CELL)
    off = grid[:, rows, columns].T - true
    along = np.linalg.norm(off - (off * across).sum(1, keepdims=True) * across, axis=1)  # depth noise aside
    assert np.quantile(along, 0.95) < 0.5, (
        f"95% of nodes within half a voxel of their cell's centre, not {np.quantile(along, 0.95):.2f}"
    )
    filled, known = bridge_holes(grid, locked)
    steps = grid_steps(np.where(known, filled, -1), CELL)
    assert np.mean(np.abs(steps - 1) <= 0.05) > 0.95, np.percentile(steps, [5, 50, 95])


def test_flattened_grid_is_isometric():
    nodes, normal, _, _, a, b, _ = tilted_cylinder()
    _, _, sign, u, v = level_pieces.flatten_pieces(nodes, np.zeros(len(nodes), dtype=np.int32), a, b)
    flat = np.hypot(u[b] - u[a], v[b] - v[a])
    true = np.sqrt(sum((nodes[c][b].astype(np.float64) - nodes[c][a]) ** 2 for c in "xyz"))
    assert np.percentile(flat / true, 5) > 0.95, "the flattening keeps the sheet's lengths"
    grid, locked, _ = rasterise(nodes, normal * sign[:, None], u, v)
    filled, known = bridge_holes(grid, locked)
    steps = grid_steps(np.where(known, filled, -1), CELL)
    assert np.mean(np.abs(steps - 1) <= 0.1) > 0.97, np.percentile(steps, [5, 50, 95])
