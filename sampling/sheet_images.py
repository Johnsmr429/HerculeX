"""Mosaics, margins and grids shared by every image of a sheet."""

import math

import numpy as np

SHEET_HALF_UM = 28.0
LAYER_STEP_UM = 9.4
BAND_ROWS = 1024
MARGIN_CELLS = 2


def mosaics_of(pieces):
    """The pieces of a layout grouped into its continuous areas."""
    mosaics = {}
    for p in pieces:
        mosaics.setdefault((p.get("group"), round(p.get("moved_from_joint_mm", 0.0))), []).append(p)
    return list(mosaics.values())


def largest_mosaic(pieces):
    """The pieces of a layout's largest continuous area."""
    return max(mosaics_of(pieces), key=lambda members: sum(p["area"] for p in members))


def extended(xyz):
    """A mesh carried straight on past its edges and into its small holes, padded by the margin."""
    grid = np.pad(xyz, ((0, 0), (MARGIN_CELLS, MARGIN_CELLS), (MARGIN_CELLS, MARGIN_CELLS)), constant_values=-1.0)
    for _ in range(MARGIN_CELLS):
        before = grid.copy()
        source = before[0] > 0
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
            fresh = (grid[0] <= 0) & np.roll(source, step, axis) & np.roll(source, 2 * step, axis)
            where = np.nonzero(fresh)
            near, far = list(where), list(where)
            near[axis] = (where[axis] - step) % source.shape[axis]
            far[axis] = (where[axis] - 2 * step) % source.shape[axis]
            grid[:, where[0], where[1]] = 2 * before[:, near[0], near[1]] - before[:, far[0], far[1]]
    return grid


def sheet_layers(voxel_um):
    """The number of centred CT layers used to make the sheet image."""
    return 2 * math.ceil(SHEET_HALF_UM / voxel_um) + 1


def across_sheet(stack, voxel_um, box=None, *, all_layers=True):
    """The brightest voxel across the sheet and the requested scan layers by depth."""
    middle, reach = stack.shape[0] // 2, sheet_layers(voxel_um) // 2
    step = max(1, round(LAYER_STEP_UM / voxel_um))
    y0, y1, x0, x1 = box or (0, stack.shape[1], 0, stack.shape[2])
    sheet = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    depths = range(-(reach // step) * step, reach + 1, step) if all_layers else (0,)
    layers = {depth: np.zeros_like(sheet) for depth in depths}
    for a in range(y0, y1, BAND_ROWS):
        b = min(a + BAND_ROWS, y1)
        band = stack.read((middle - reach, middle + reach + 1), (a, b), (x0, x1))
        sheet[a - y0 : b - y0] = band.max(axis=0)
        for depth, layer in layers.items():
            layer[a - y0 : b - y0] = band[depth + reach]
    return sheet, layers
