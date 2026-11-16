"""Whether a map holds lines of writing."""

import numpy as np
from scipy import ndimage as ni

UM = 9.362e-3
CELL = 8
WINDOW_MM = (24.0, 12.0)
STEP_MM = 6.0
TILTS = np.linspace(-0.08, 0.08, 17)
PERIODS = np.arange(3.0, 9.01, 0.25)
MIN_COVER = 0.7


def cells(image, seen):
    """A map on cells of mean value over what is seen, and each cell's cover."""
    h, w = image.shape[0] // CELL * CELL, image.shape[1] // CELL * CELL
    shape = (h // CELL, CELL, w // CELL, CELL)
    cover = seen[:h, :w].reshape(shape).mean(axis=(1, 3))
    total = np.where(seen, image, 0)[:h, :w].reshape(shape).sum(axis=(1, 3)) / (CELL * CELL)
    return total / np.maximum(cover, 1e-3), cover


def whitened(values, cover):
    """Values with their slow trend and their level taken off, zero where nothing is seen."""
    good = cover > 0.5
    weight = good.astype(np.float32)
    sigma = 4.0 / (UM * CELL)
    trend = ni.gaussian_filter(np.where(good, values, 0), sigma) / np.maximum(ni.gaussian_filter(weight, sigma), 1e-3)
    rest = np.where(good, values - trend, 0)
    return rest / max(float(rest[good].std()), 1e-6), good


def window_power(rest, good):
    """Power by period, the best tilt for each, of one window."""
    height, width = rest.shape
    across = (np.arange(height) * UM * CELL)[:, None]
    along = (np.arange(width) - width / 2) * UM * CELL
    waves = np.exp(-2j * np.pi * (across / PERIODS[None, :]))
    best = np.zeros(len(PERIODS))
    for tilt in TILTS:
        shifts = np.rint(tilt * along / (UM * CELL)).astype(int)
        total = np.zeros((height, len(PERIODS)), dtype=np.complex128)
        profile = np.zeros(height)
        count = np.zeros(height)
        for x, shift in enumerate(shifts):
            profile += np.roll(rest[:, x], -shift)
            count += np.roll(good[:, x], -shift)
        profile = profile / np.sqrt(np.maximum(count, 1))
        profile = profile * np.hanning(height)
        total = np.abs(profile @ waves) ** 2 / height
        best = np.maximum(best, total)
    return best


def spectrum(values, cover):
    """Mean of window powers across the lines and, as a control, along them."""
    rest, good = whitened(values, cover)
    height, width = (int(mm / (UM * CELL)) for mm in WINDOW_MM)
    step = int(STEP_MM / (UM * CELL))
    found = {"across": [], "along": []}
    for name, a, b in (("across", rest, good), ("along", rest.T, good.T)):
        for y in range(0, a.shape[0] - height + 1, step):
            for x in range(0, a.shape[1] - width + 1, step):
                part = b[y : y + height, x : x + width]
                if part.mean() >= MIN_COVER:
                    found[name].append(window_power(a[y : y + height, x : x + width], part))
    return found
