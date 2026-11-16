"""A suggested square lands where the viewer draws that part of the piece, turned or not."""

from calibration.text.blind_pairs import FIELD
from website.suggested_areas import square_on_sheet


def piece(**turned):
    return {"x_mm": 100.0, "y_mm": 50.0, "stride": 16, "um": 0.01, "flip_rows": False, "flip_cols": False} | turned


def test_the_margin_comes_off_and_flips_measure_from_the_other_edge():
    shape = (FIELD + 300, FIELD + 500)
    assert square_on_sheet(piece(), (100, 200), shape) == {"x": 101.68, "y": 50.68, "w": 14.4, "h": 14.4}
    assert square_on_sheet(piece(flip_cols=True), (100, 200), shape)["x"] == 102.68
    assert square_on_sheet(piece(flip_rows=True), (100, 200), shape)["y"] == 51.68
