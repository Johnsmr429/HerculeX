"""The network simplex must reach the same optimum as a general LP solver, on graphs with conflicting votes and bounds."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

from unwrapping.level.l1_synchronization import (
    LENIENCY,
    SCALE,
    cost_of,
    solve_offsets,
    synchronize,
)


def reference_cost(count, a, b, k, w, bound):
    above = np.where(bound < 0, LENIENCY, 1.0)
    below = np.where(bound > 0, LENIENCY, 1.0)
    size, rows = len(w), np.arange(len(w))

    def diff(scale):
        return sparse.coo_matrix(
            (np.concatenate([scale, -scale]), (np.concatenate([rows, rows]), np.concatenate([b, a]))),
            shape=(size, count),
        )

    eye = sparse.identity(size, format="csr")
    system = sparse.vstack([sparse.hstack([diff(above), -eye]), sparse.hstack([diff(-below), -eye])]).tocsr()
    result = linprog(
        np.concatenate([np.zeros(count), w]),
        A_ub=system,
        b_ub=np.concatenate([above * k, -below * k]),
        bounds=[(None, None)] * count + [(0, None)] * size,
        method="highs",
    )
    return result.fun


@pytest.mark.parametrize("seed", range(6))
def test_matches_general_solver(seed):
    rng = np.random.default_rng(seed)
    count, votes = 60, 400
    a = rng.integers(0, count, votes)
    b = (a + rng.integers(1, count - 1, votes)) % count
    truth = rng.integers(-6, 7, count)
    k = truth[b] - truth[a] + rng.choice([0, 0, 0, 1, -1, 2], votes)
    w = np.round(rng.uniform(0.1, 1.0, votes) * SCALE) / SCALE
    bound = rng.choice([0, 0, 1, -1], votes).astype(float)
    f = solve_offsets(count, a, b, k.astype(float), w, bound)
    assert np.allclose(f, np.round(f))
    assert cost_of(f, a, b, k, w, bound) == pytest.approx(
        reference_cost(count, a, b, k.astype(float), w, bound), rel=1e-6
    )


@pytest.mark.parametrize("seed", range(4))
def test_windowed_solve_stays_near_the_optimum(seed):
    rng = np.random.default_rng(100 + seed)
    zones, per_zone = 16, 30
    count = zones * per_zone
    zone = np.repeat(np.arange(zones), per_zone)
    truth = rng.integers(-8, 9, count)
    a = rng.integers(0, count, 6000)
    b = np.clip(a + rng.integers(-2 * per_zone, 2 * per_zone, 6000), 0, count - 1)
    keep = a != b
    a, b = a[keep], b[keep]
    k = truth[b] - truth[a] + rng.choice([0, 0, 0, 0, 1, -1], len(a))
    w = np.round(rng.uniform(0.2, 1.0, len(a)) * SCALE) / SCALE
    bound = np.zeros(len(a))
    exact = solve_offsets(count, a, b, k.astype(float), w, bound)
    windowed, _, _ = synchronize(a, b, k, w, count, bound, zone)
    assert np.allclose(windowed, np.round(windowed))
    assert cost_of(windowed, a, b, k, w, bound) <= 1.02 * cost_of(exact, a, b, k, w, bound)
