"""End to end on a synthetic roll."""

import numpy as np

from unwrapping.layout import roll_placement as sheet

VOXEL, STRIDE, RADIUS, PITCH = 10.0, 16, 2000.0, 40.0


def spiral_piece(number, turn, span, height=120):
    """A patch of an Archimedean spiral, with a band of cells filled across a fold in the last piece."""
    columns = int(2 * np.pi * RADIUS * span / STRIDE)
    theta = 2 * np.pi * (turn + np.arange(columns) / columns * span)
    radius = RADIUS + PITCH * theta / (2 * np.pi)
    z = 500.0 + STRIDE * np.arange(height)[:, None] + 0 * theta
    x, y = 5000 + radius * np.cos(theta) + 0 * z, 5000 + radius * np.sin(theta) + 0 * z
    xyz = np.stack([x, y, z]).astype(np.float64)
    row = {
        "piece": number,
        "winding": int(round(turn + span / 2)),
        "area_mm2": float(columns * height) * (STRIDE * VOXEL / 1000) ** 2,
    }
    meta = {"sample": "synthetic", "scan": "roll", "voxel_um": VOXEL, "stride": STRIDE}
    return {"row": row, "xyz": xyz, "good": np.ones(xyz.shape[1:], dtype=bool), "meta": meta}


def test_pieces_of_a_spiral_fall_in_order_along_the_roll():
    pieces = [spiral_piece(k, 0.1 + 0.3 * k, 0.28) for k in range(6)]
    pieces[-1]["xyz"][:, 60, 40] += 3000.0
    centre = sheet.scroll_centre(pieces)
    assert np.hypot(*(centre - 5000.0)) < 150
    sheet.placed(pieces, centre, 1)
    arcs = [p["arc"] for p in pieces]
    assert arcs == sorted(arcs)
