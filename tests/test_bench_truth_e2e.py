"""The bench's named truth: whole-turn corrections of misnamed meshes land only on the meshes and controls they name."""

import re

import numpy as np

from calibration.bench import truth


def test_name_corrections_move_only_the_listed_wraps_of_the_listed_controls():
    names = np.array([f"20260101000000-w{w:03d}_x" for w in (49, 50, 51, 55, 59, 60)] + ["unnamed"])
    control = "PHerc0139-20250728140407-rows-0019-0031"
    assert truth.name_corrections(control, names).tolist() == [0, 0, -1, -1, -1, 0, 0]
    assert not truth.name_corrections("PHerc1667-20231117161658-rows-0008-0020", names).any()
    for listed, spans in truth.NAME_TURNS.items():
        assert re.fullmatch(r"PHerc\w+-\d{14}-rows-\d{4}-\d{4}", listed)
        assert all(first <= last and turns for (first, last), turns in spans.items())
