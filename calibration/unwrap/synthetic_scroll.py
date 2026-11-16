"""A crushed scroll of exactly known windings, built voxel by voxel."""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ni

from storage.brick_store import BrickStore

BRICK = 128
AIR, PAPYRUS = 0, 120
BACKGROUND = 50


class Scroll:
    """Sheets are the integer levels of phase = d / gap - f."""

    def __init__(self, gap, half_flat, turns, core, seed=7):
        self.gap, self.half_flat, self.turns, self.core = gap, half_flat, turns, core
        self.rng = np.random.default_rng(seed)

    def phase(self, x, y):
        h = self.half_flat
        cx = np.clip(x, -h, h)
        dx = x - cx
        d = np.hypot(dx, y)
        angle = np.arctan2(y, dx)
        right = d * (angle + np.pi / 2)
        top = np.pi * d + (h - x)
        left = np.pi * d + 2 * h + d * (np.mod(angle, 2 * np.pi) - np.pi / 2)
        bottom = 2 * np.pi * d + 2 * h + (x + h)
        walk = np.where(dx > 0, right, np.where(dx < 0, left, np.where(y > 0, top, bottom)))
        f = walk / np.maximum(4 * h + 2 * np.pi * d, 1e-6)
        return d / self.gap - f, f, d


def smooth_field(rng, shape, sigma):
    """Unit-variance noise smoothed to the given correlation lengths."""
    field = ni.gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma, mode="wrap")
    return field / field.std()


def build(
    folder,
    gap=7.0,
    half_flat=0.0,
    turns=60,
    core=6,
    height=256,
    thickness=1.4,
    noise=12.0,
    wander=0.0,
    fading=0.0,
    seed=7,
):
    """wander moves each sheet by this share of a gap, so neighbours touch."""
    scroll = Scroll(gap, half_flat, turns, core, seed)
    reach = (turns + core + 2) * gap
    Y = int(2 * reach + 32)
    X = int(2 * (reach + half_flat) + 32)
    Y, X = -(-Y // BRICK) * BRICK, -(-X // BRICK) * BRICK
    Z = -(-height // BRICK) * BRICK
    cy, cx = (Y - 1) / 2, (X - 1) / 2
    y, x = np.mgrid[:Y, :X].astype(np.float32)
    x, y = x - cx, y - cy
    phase, _, d = scroll.phase(x, y)
    rng = scroll.rng
    table_shape = (turns + core + 4, 64, Z // 8 + 2)
    wandering = smooth_field(rng, table_shape, (0.7, 3.0, 6.0))
    contrast = smooth_field(rng, table_shape, (1.5, 2.0, 4.0))
    store = BrickStore(folder, (Z, Y, X), (BRICK, BRICK, BRICK), source="synthetic")
    k = np.clip(np.round(phase), 0, table_shape[0] - 1)
    _, f, _ = scroll.phase(x, y)
    s = f * (table_shape[1] - 1)
    slab = np.zeros((BRICK, Y, X), dtype=np.uint8)
    for z0 in range(0, Z, BRICK):
        for dz in range(BRICK):
            zc = np.full_like(k, (z0 + dz) / 8)
            coordinates = np.stack([k, s, zc])
            shift = ni.map_coordinates(wandering, coordinates, order=1, mode="nearest") * wander if wander else 0.0
            fade = ni.map_coordinates(contrast, coordinates, order=1, mode="nearest") if fading else None
            moved = phase + shift
            distance = np.abs(moved - np.round(moved)) * gap
            amplitude = (PAPYRUS - BACKGROUND) * (1.0 if fade is None else np.clip(1 + fading * fade, 0.1, 1.5))
            plane = BACKGROUND + amplitude * np.exp(-0.5 * (distance / thickness) ** 2)
            plane = plane + rng.normal(0, noise, plane.shape).astype(np.float32)
            plane[(d < core * gap) | (d > (turns + core) * gap)] = AIR
            slab[dz] = np.clip(plane, 0, 255).astype(np.uint8)
        for by in range(Y // BRICK):
            for bx in range(X // BRICK):
                block = slab[:, by * BRICK : (by + 1) * BRICK, bx * BRICK : (bx + 1) * BRICK]
                store.put((z0 // BRICK, by, bx), np.ascontiguousarray(block).tobytes())
    store.save()
    record = {
        "gap": gap,
        "half_flat": half_flat,
        "turns": turns,
        "core": core,
        "height": height,
        "thickness": thickness,
        "noise": noise,
        "wander": wander,
        "fading": fading,
        "seed": seed,
        "centre": [cx, cy],
        "shape": [Z, Y, X],
    }
    (Path(folder) / "truth.json").write_text(json.dumps(record, indent=1))
    return record


def true_turn(record, x, y):
    """Sheet number and walk fraction of node positions, before wander."""
    scroll = Scroll(record["gap"], record["half_flat"], record["turns"], record["core"])
    phase, f, _ = scroll.phase(
        np.asarray(x, dtype=np.float64) - record["centre"][0], np.asarray(y, dtype=np.float64) - record["centre"][1]
    )
    return np.round(phase).astype(np.int64), f


def true_winding(record, x, y, z):
    """Continuous winding of each position, wander included."""
    scroll = Scroll(record["gap"], record["half_flat"], record["turns"], record["core"], record["seed"])
    phase, f, _ = scroll.phase(
        np.asarray(x, dtype=np.float64) - record["centre"][0], np.asarray(y, dtype=np.float64) - record["centre"][1]
    )
    if not record["wander"]:
        return phase + f
    table_shape = (record["turns"] + record["core"] + 4, 64, record["shape"][0] // 8 + 2)
    wandering = smooth_field(scroll.rng, table_shape, (0.7, 3.0, 6.0))
    k = np.clip(np.round(phase), 0, table_shape[0] - 1)
    shift = ni.map_coordinates(
        wandering, np.stack([k, f * (table_shape[1] - 1), np.asarray(z, dtype=np.float64) / 8]), order=1, mode="nearest"
    )
    return phase + f + shift * record["wander"]
