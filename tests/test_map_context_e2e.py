"""The second reader learns from reach on unseen windows; destroying rows keeps each strip."""

import numpy as np
from scipy import ndimage as ni

from calibration.ink.ink_yield import rank_auc
from calibration.ink.map_context_invention import rows_destroyed
from ink.training import map_context_reader as map_context

SIDE = 256


def written_window(rng):
    """Long bars under noise of their own width."""
    truth = np.zeros((SIDE, SIDE), np.float32)
    for y in range(20, SIDE - 20, 48):
        for x in range(rng.integers(0, 24), SIDE - 40, 40):
            truth[y : y + 5, x : x + 30] = 1
    noise = ni.gaussian_filter(rng.normal(size=truth.shape), 2.0)
    read = np.clip(0.4 + 0.25 * ni.gaussian_filter(truth, 1.0) + 0.12 * noise / noise.std(), 0, 1).astype(np.float32)
    return read, truth, np.ones_like(truth)


def test_the_second_reader_learns_from_reach_on_unseen_windows():
    rng = np.random.default_rng(1)
    net = map_context.taught([written_window(rng) for _ in range(6)], levels=3, steps=120)
    read, truth, sheet = written_window(rng)
    second = map_context.read(net, read, sheet)
    assert second.shape == read.shape
    assert rank_auc(second[truth > 0.5], second[truth < 0.5]) > rank_auc(read[truth > 0.5], read[truth < 0.5])


def test_destroying_rows_keeps_what_each_strip_holds():
    rng = np.random.default_rng(2)
    read, _, sheet = written_window(rng)
    mixed, cover = rows_destroyed(read, sheet, rng)
    assert mixed.shape == read.shape and (cover == 1).all()
    assert np.allclose(np.sort(mixed[:, :40], axis=None), np.sort(read[:, :40], axis=None))
    assert np.abs(mixed - read).mean() > 0.01
