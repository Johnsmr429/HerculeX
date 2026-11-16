"""The scroll's axis from sheet normals."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

BAND_CELLS = 32
ZOOM = 0.75
ZOOM_STEPS = 12
WORKERS = 6


def line_fit(x, y, ux, uy, weight):
    """Weighted least-squares point nearest the lines through (x, y) along unit (ux, uy)."""
    tx, ty = -uy, ux
    a = np.array(
        [[np.sum(weight * tx * tx), np.sum(weight * tx * ty)], [np.sum(weight * tx * ty), np.sum(weight * ty * ty)]]
    )
    b = np.array([np.sum(weight * tx * (tx * x + ty * y)), np.sum(weight * ty * (tx * x + ty * y))])
    return np.linalg.solve(a, b)


def fit_centre(x, y, ux, uy, weight):
    """Robust focus of the normals, zooming inward."""
    c = np.array([np.average(x, weights=weight), np.average(y, weights=weight)])
    radius = float(np.sqrt(np.average((x - c[0]) ** 2 + (y - c[1]) ** 2, weights=weight)))
    miss_scale = 0.5 * radius
    for reach in radius * ZOOM ** np.arange(ZOOM_STEPS):
        for _ in range(3):
            miss = (c[0] - x) * (-uy) + (c[1] - y) * ux
            near = 1.0 / (1.0 + (np.hypot(x - c[0], y - c[1]) / reach) ** 4)
            c = line_fit(x, y, ux, uy, weight * near / (1 + (miss / miss_scale) ** 2))
            miss_scale = max(miss_scale * 0.8, 8.0)
    return c


def axis(nodes, z_low, z_high, cell=8):
    """Knots (z, x, y) of the axis over the nodes' planes."""
    n = np.stack([nodes["nx"], nodes["ny"]], axis=1).astype(np.float32) / 127
    length = np.hypot(n[:, 0], n[:, 1])
    usable = (length > 0.7) & (nodes["prominence"] >= 16)
    z = nodes["z"]
    step = BAND_CELLS * cell

    def knot(a):
        s = usable & (z >= a) & (z < a + step)
        if s.sum() < 500:
            return None
        c = fit_centre(
            nodes["x"][s].astype(np.float64),
            nodes["y"][s].astype(np.float64),
            n[s, 0] / length[s],
            n[s, 1] / length[s],
            nodes["prominence"][s].astype(np.float64),
        )
        return a + step / 2, c[0], c[1]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return np.array([k for k in pool.map(knot, np.arange(z_low, z_high, step)) if k is not None])


def centre_at(knots, z):
    return np.interp(z, knots[:, 0], knots[:, 1]), np.interp(z, knots[:, 0], knots[:, 2])
