"""The written face found from either side, no face without plies, a gap told from a contact."""

import numpy as np
from scipy import ndimage as ni

from sampling.fibre_faces import gap_state, recto_face

SIDE = 400


def ply(rng, along_rows):
    grain = ni.gaussian_filter(rng.normal(size=(5, SIDE, SIDE)), (0.5, 1, 10) if along_rows else (0.5, 10, 1))
    return grain / grain.std()


def two_ply_sheet():
    """Towards higher layers."""
    rng = np.random.default_rng(3)
    stack = np.full((31, SIDE, SIDE), 50.0)
    stack[9:14] = 110 + 12 * ply(rng, along_rows=False)
    stack[14:19] = 110 + 12 * ply(rng, along_rows=True)
    return np.clip(stack, 1, 255).astype(np.uint8)


def test_the_written_face_is_found_from_either_side():
    stack = two_ply_sheet()
    found = recto_face(stack)
    assert found["face"] == "near" and found["polarity"] > 0.5
    assert recto_face(stack[::-1])["face"] == "far"


def test_a_sheet_without_plies_names_no_face():
    rng = np.random.default_rng(4)
    stack = np.full((31, SIDE, SIDE), 50.0)
    stack[9:19] = 110 + 12 * ni.gaussian_filter(rng.normal(size=(10, SIDE, SIDE)), (0.5, 2, 2)) / 0.05
    assert recto_face(np.clip(stack, 1, 255).astype(np.uint8))["face"] == "unknown"


def test_a_gap_is_told_from_a_contact():
    stack = two_ply_sheet()
    assert (gap_state(stack) == 0).mean() > 0.9
    stack[19:] = 110
    assert (gap_state(stack) == 2).mean() > 0.5
