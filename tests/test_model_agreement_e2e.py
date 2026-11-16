"""The ways of combining reads, on a stack with a known bright layer and readers that need no GPU."""

import numpy as np
import pytest

from calibration.ink import model_agreement as agreement
from ink.training.model_training import DEPTH


class Reader:
    def __init__(self, gain):
        self.gain = gain

    def read(self, block):
        assert block.shape[0] == DEPTH
        return self.gain * block.astype(np.float32).mean(axis=0) / 255


def test_every_way_keeps_the_map_and_its_range():
    rng = np.random.default_rng(1)
    stack = rng.integers(40, 200, (31, 64, 96), dtype=np.uint8)
    maps = agreement.maps_of([Reader(1.0), Reader(0.5)], stack)
    assert maps.shape == (2, len(agreement.SHIFTS), 64, 96)
    assert np.allclose(maps[1], maps[0] * 0.5)
    ways = agreement.ways(maps)
    middle = agreement.SHIFTS.index(0)
    assert set(ways) == {
        "first model",
        "models averaged",
        "first two multiplied",
        "depths averaged",
        "models and depths averaged",
    }
    assert np.allclose(ways["models averaged"], maps[0, middle] * 0.75)
    assert np.allclose(ways["first two multiplied"], maps[0, middle] ** 2 * 0.5)
    assert all(read.shape == (64, 96) and read.min() >= 0 and read.max() <= 1 for read in ways.values())


@pytest.mark.skipif(
    not (agreement.calibration_output("scroll_triage") / "census-pieces.json").is_file(),
    reason="needs the PHerc0211 piece census of calibration/scroll_triage",
)
def test_the_top_pieces_are_the_ones_that_read_like_text():
    pieces = agreement.top_pieces()
    assert pieces and all(row["ratio"] >= agreement.TOP_RATIO for row in pieces)
    assert [row["ratio"] for row in pieces] == sorted((row["ratio"] for row in pieces), reverse=True)
