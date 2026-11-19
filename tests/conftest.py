"""Tests marked apple or cuda run only on that lane's machines."""

import pytest

from runner.lane import LANE, LANES


def pytest_collection_modifyitems(items):
    for item in items:
        for lane in LANES:
            if lane != LANE and item.get_closest_marker(lane):
                item.add_marker(pytest.mark.skip(reason=f"runs on the {lane} lane"))
