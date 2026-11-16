"""The raw sheet ruler reads one level a raw sheet on a stack of known levels, sees through one fold, and reads the loss when levels are merged."""

import numpy as np

from calibration.unwrap import raw_sheet_ruler as ruler
from storage.brick_store import BrickStore

BRICK, PITCH, SHEETS, UM = 64, 10, 43, 0.018724
SHAPE = (BRICK, 7 * BRICK, BRICK)


def stack(tmp_path):
    """Flat sheets ten voxels apart in a raw scan, with nodes on each."""
    rows = 10 + PITCH * np.arange(SHEETS)
    y = np.arange(SHAPE[1])[:, None]
    profile = 40 + 180 * np.exp(-0.5 * ((y - rows[None, :]) / 1.5) ** 2).sum(axis=1)
    volume = np.broadcast_to(profile.astype(np.uint8)[None, :, None], SHAPE)
    store = BrickStore(tmp_path / "bricks", SHAPE, (BRICK,) * 3)
    for index in np.ndindex(*store.grid):
        z, y0, x = (i * BRICK for i in index)
        store.put(index, np.ascontiguousarray(volume[z : z + BRICK, y0 : y0 + BRICK, x : x + BRICK]).tobytes())
    store.save()
    sheet, z, x = np.meshgrid(np.arange(SHEETS), np.arange(8, 57, 4), np.arange(8, 57, 4), indexing="ij")
    nodes = np.zeros(sheet.size, dtype=[(c, np.float32) for c in ("x", "y", "z", "nx", "ny", "nz")])
    nodes["x"], nodes["y"], nodes["z"], nodes["ny"] = x.ravel(), rows[sheet.ravel()], z.ravel(), 1
    return store, nodes, sheet.ravel().astype(np.float64)


def read(store, nodes, winding):
    return ruler.summary(ruler.scored(nodes, winding, store, UM), UM)


def test_ruler_on_known_levels(tmp_path):
    store, nodes, sheet = stack(tmp_path)
    exact = read(store, nodes, sheet)
    assert exact["lines"] >= 50 and exact["right"] == 1.0 and exact["lost"] == 0.0 and exact["folded"] == 0.0
    assert exact["levels_per_raw_sheet"][2] == 1.0 and exact["raw_sheets_with_a_node"] == 1.0
    folded = read(store, nodes, np.abs(sheet - SHEETS // 2))
    assert folded["right"] == 1.0 and folded["folded"] > 0.2
    merged = read(store, nodes, sheet // 2)
    assert merged["right"] == 0.0 and 0.4 <= merged["levels_per_raw_sheet"][2] <= 0.6
    paired = read(store, nodes, sheet + (sheet % 2))
    assert paired["noisy"] == 0.0 and paired["right"] == 1.0
    zigzag = read(store, nodes, sheet % 2)
    assert zigzag["right"] == 0.0 and zigzag["noisy"] == 1.0
