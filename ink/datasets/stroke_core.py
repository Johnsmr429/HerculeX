"""Ink drawn fat by hand labels and published maps, thinned to the width strokes have on the infrared fragments."""

import numpy as np
from scipy import ndimage as ni
from skimage.morphology import skeletonize

FULL_MM, EDGE_MM = 0.10, 0.25
MARGIN = 64


def stroke_core(ink, pixel_mm):
    """A ridge along the centre lines of an ink mask: 1 within FULL_MM of a line, falling to 0 at EDGE_MM, never outside the mask."""
    ink = np.asarray(ink, dtype=bool)
    if not ink.any():
        return np.zeros(ink.shape, dtype=np.float32)
    away = ni.distance_transform_edt(~skeletonize(ink)) * pixel_mm
    return (np.clip((EDGE_MM - away) / (EDGE_MM - FULL_MM), 0, 1) * ink).astype(np.float32)


def core_of_patch(ink, known_shape, there, pixel_mm):
    """The ridge of one patch, drawn with a margin so that its centre lines are those of the whole mask."""
    (y0, y1), (x0, x1) = ((s.start, s.stop) for s in there)
    top, left = max(y0 - MARGIN, 0), max(x0 - MARGIN, 0)
    wide = stroke_core(ink[top : min(y1 + MARGIN, known_shape[0]), left : min(x1 + MARGIN, known_shape[1])], pixel_mm)
    return wide[y0 - top : y1 - top, x0 - left : x1 - left]
