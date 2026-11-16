"""spots.reflatten: an uneven grid on a curved sheet comes out isometric, and holes and lone nodes do not break it."""

import numpy as np

from spots.reflatten import isometric, stretch

STEP = 16.0


def uneven_cylinder(h=36, w=44, amplitude=(0.25, 0.15), radius=400.0):
    """A cylinder patch whose grid steps vary along both axes, as our unwraps' grids do."""
    s = np.cumsum(STEP * (1 + amplitude[0] * np.sin(np.arange(w) / 6)))[None].repeat(h, 0)
    t = np.cumsum(STEP * (1 + amplitude[1] * np.cos(np.arange(h) / 5)))[:, None].repeat(w, 1)
    return np.stack([radius * np.cos(s / radius) + 1000, radius * np.sin(s / radius) + 1000, t + 1000]).astype(
        np.float32
    )


def test_uneven_grid_comes_out_isometric():
    xyz = uneven_cylinder()
    assert np.mean(np.abs(stretch(xyz, STEP) - 1) <= 0.1) < 0.5
    new, _, report = isometric(xyz, STEP, 1500)
    after = stretch(new, STEP)
    assert np.mean(np.abs(after - 1) <= 0.1) >= 0.98
    assert report["within_10pct_after"] >= 0.98


def test_holes_and_isolated_nodes():
    xyz = uneven_cylinder()
    xyz[:, 10:14, 10:20] = -1
    xyz[:, 20:23, 30:33] = -1
    xyz[:, 21, 31] = [1200, 1200, 1300]  # alone inside a hole
    new, position, report = isometric(xyz, STEP, 1500)
    assert report["isolated_nodes_left_out"] == 1
    assert not np.isnan(new).any()
    assert np.isnan(position[:, 21, 31]).all()  # the lone node has no place in the layout
    good = new[0] > 0
    assert good.mean() > 0.8
