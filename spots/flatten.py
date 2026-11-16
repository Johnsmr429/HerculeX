"""A rendered stack flattened onto its own papyrus layers: the mesh made to follow the sheet pixel by pixel.

A mesh at 16-voxel nodes, smoothed, cannot follow papyrus corrugated at a millimetre: in a section along the normal the
sheet's layers then wave up and down across the mesh by several layers, and a reader's fixed window mixes them. Here,
as seismic horizons are flattened, each pixel's column is shifted along the normal so its layering lines up with its
neighbourhood's. Measured at half resolution in plane: the reference is the stack already flattened, blurred in plane
by REFERENCE_PX; each pixel's shift is the one (within REACH layers, in steps of STEP) whose column best correlates
with the reference over the central 2 x WINDOW + 1 layers; the shifts are median-filtered, smoothed in plane by
SMOOTH_PX and added, and the round repeats. The update per round is held within REACH, under half the spacing of
neighbouring sheets, so a column cannot jump to the next wrap. The shift field is then brought to full resolution and
the stack resampled along each column by it (linear between layers): a move of the mesh along its normal, per pixel.
"""

import numpy as np
from scipy import ndimage

REACH = 3.0
STEP = 0.25
WINDOW = 12
REFERENCE_PX = 24.0
SMOOTH_PX = 6.0
ROUNDS = 5
HALF = 2


def shifted(stack, offset, layers=None):
    """stack[z + offset(y, x), y, x] for the layers z asked (all by default), linear between layers, edges held."""
    count = stack.shape[0]
    z = np.arange(count) if layers is None else np.asarray(layers)
    at = np.clip(z.astype(np.float32)[:, None, None] + offset[None], 0, count - 1)
    low = np.floor(at).astype(np.int32)
    t = at - low
    high = np.minimum(low + 1, count - 1)
    return np.take_along_axis(stack, low, 0) * (1 - t) + np.take_along_axis(stack, high, 0) * t


def normalised(columns):
    return (columns - columns.mean(0)) / (columns.std(0) + 1e-6)


def offsets(stack, valid, rounds=ROUNDS, reach=REACH, window=WINDOW, reference_px=REFERENCE_PX, smooth_px=SMOOTH_PX):
    """The shift per pixel (layers) that flattens a (layers, h, w) stack, at the stack's resolution, and each round's
    median |update|."""
    layers, h, w = stack.shape
    hh, ww = h // HALF, w // HALF
    small = stack[:, : hh * HALF, : ww * HALF].astype(np.float32).reshape(layers, hh, HALF, ww, HALF).mean((2, 4))
    small = ndimage.gaussian_filter(small, (0, 1.0, 1.0))
    seen = valid[: hh * HALF, : ww * HALF].reshape(hh, HALF, ww, HALF).all((1, 3))
    band = np.arange(layers // 2 - window, layers // 2 + window + 1)
    weight = np.maximum(ndimage.gaussian_filter(seen.astype(np.float32), reference_px / HALF), 1e-6)
    offset = np.zeros((hh, ww), np.float32)
    steps = np.arange(-reach, reach + STEP / 2, STEP, dtype=np.float32)
    history = []
    for _ in range(rounds):
        flat = shifted(small, offset, band)
        reference = ndimage.gaussian_filter(np.where(seen, flat, 0), (0, reference_px / HALF, reference_px / HALF))
        reference = normalised(reference / weight)
        best, score = np.zeros((hh, ww), np.float32), np.full((hh, ww), -np.inf, np.float32)
        for s in steps:
            c = (normalised(shifted(small, offset + s, band)) * reference).mean(0)
            better = c > score
            best[better], score[better] = s, c[better]
        best = np.where(seen, best, 0)
        update = np.clip(ndimage.gaussian_filter(ndimage.median_filter(best, 5), smooth_px / HALF), -reach, reach)
        offset += update
        history.append(round(float(np.median(np.abs(update[seen]))), 3))
    full = ndimage.zoom(offset, (h / hh, w / ww), order=1)[:h, :w]
    full = np.pad(full, ((0, h - full.shape[0]), (0, w - full.shape[1])), mode="edge")
    return np.where(valid, full, 0).astype(np.float32), history


def flattened(stack, valid=None, **settings):
    """(the stack flattened, as its own dtype; the shift per pixel; each round's median |update|)."""
    valid = np.ones(stack.shape[1:], bool) if valid is None else valid
    offset, history = offsets(stack, valid, **settings)
    flat = shifted(stack.astype(np.float32), offset)
    if np.issubdtype(stack.dtype, np.integer):
        flat = np.clip(np.rint(flat), np.iinfo(stack.dtype).min, np.iinfo(stack.dtype).max)
    return flat.astype(stack.dtype), offset, history
