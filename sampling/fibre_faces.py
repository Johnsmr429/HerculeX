"""Which face of a sheet is the recto, from its fibres alone."""

import numpy as np

from sampling import fibre_kernels as kernels

TILE, SMALLEST_TILE = 320, 160
MOST_TILES, FEWEST_TILES = 8, 3
TRIES_PER_TILE = 8
FACE_SEARCH = 8
OUTER_PLY = (-3, -2, -1)
FIBRE_BANDS = (1.5, 3.0)
SHEET_INSET, TILE_INSET = 12, 20
SMALLEST_COVER, FEWEST_PIXELS = 0.5, 8000
CLEAR_ERRORS, SMALLEST_ERROR = 2.5, 0.03
OPEN_GAP, TIGHT_GAP = 65, 85
GAP_LAYERS = (1, 2, 3)
GAP_SMOOTHING = 32


def searched(block):
    """The layers around the stack's middle, smoothed, where a face is looked for."""
    start = block.shape[0] // 2 - FACE_SEARCH
    return kernels.gaussian(block[start : start + 2 * FACE_SEARCH + 1].astype(np.float32), (0.7, 2.5, 2.5))


def face_depth(block, smoothed=None):
    """Per column, the sub-layer depth of the steepest fall of the sheet along the layers."""
    return kernels.face_depth(searched(block) if smoothed is None else smoothed, block.shape[0] // 2 - FACE_SEARCH)


def gap_grey(block):
    """Per column, the darkest grey just beyond the near face, smoothed along the sheet."""
    from scipy import ndimage as ni

    depth = face_depth(block)
    return ni.uniform_filter(
        np.min([kernels.layer_at(block, depth, layer) for layer in GAP_LAYERS], axis=0), GAP_SMOOTHING
    )


def gap_state(block):
    """Per column, what lies beyond the near face."""
    grey = gap_grey(block)
    return np.where(grey < OPEN_GAP, 0, np.where(grey > TIGHT_GAP, 2, 1)).astype(np.uint8)


def grain(image, where):
    """Log ratio of gradient energy across the rows to along them, over the fibre bands."""
    ratios = []
    smooth = {}
    for scale in FIBRE_BANDS:
        for width in (scale, 2 * scale):
            if width not in smooth:
                smooth[width] = kernels.gaussian(image, width)
        across, along = np.gradient(smooth[scale] - smooth[2 * scale])
        ratios.append(np.log(max((across[where] ** 2).mean(), 1e-9) / max((along[where] ** 2).mean(), 1e-9)))
    return float(np.mean(ratios))


def outer_ply_grain(block, where, smoothed=None):
    depth = face_depth(block, smoothed)
    return float(np.mean([grain(kernels.layer_at(block, depth, layer), where) for layer in OUTER_PLY]))


def polarity(block, where):
    """Fibre grain of the near face's outer ply minus the far face's."""
    smoothed = searched(block)
    return outer_ply_grain(block, where, smoothed) - outer_ply_grain(
        block[::-1], where, smoothed[::-1] if block.shape[0] % 2 else None
    )


def judged_columns(sheet):
    from scipy import ndimage as ni

    where = ni.binary_erosion(sheet, iterations=SHEET_INSET)
    where[:TILE_INSET], where[-TILE_INSET:], where[:, :TILE_INSET], where[:, -TILE_INSET:] = False, False, False, False
    return where


def tile_polarities(stack, tiles=MOST_TILES):
    """(row, column, polarity) of random tiles well covered by sheet."""
    depth, height, width = stack.shape
    side = min(TILE, height, width)
    if side < SMALLEST_TILE:
        return []
    rng = np.random.default_rng(height * 31 + width)
    found = []
    for _ in range(tiles * TRIES_PER_TILE):
        y, x = int(rng.integers(0, height - side + 1)), int(rng.integers(0, width - side + 1))
        block = np.asarray(stack[:, y : y + side, x : x + side])
        sheet = block[depth // 2] > 0
        if np.count_nonzero(sheet) < SMALLEST_COVER * sheet.size:
            continue
        where = judged_columns(sheet)
        if np.count_nonzero(where) >= FEWEST_PIXELS:
            found.append((y, x, polarity(block, where)))
            if len(found) == tiles:
                break
    return found


def recto_face(stack, tiles=MOST_TILES):
    """Which face of a stack is the recto."""
    values = np.array([value for _, _, value in tile_polarities(stack, tiles)])
    if len(values) < FEWEST_TILES:
        return {"face": "unknown", "polarity": None, "error": None, "tiles": len(values)}
    mean, error = float(values.mean()), float(values.std() / np.sqrt(len(values)))
    face = ("near" if mean > 0 else "far") if abs(mean) >= CLEAR_ERRORS * max(error, SMALLEST_ERROR) else "unknown"
    return {"face": face, "polarity": round(mean, 4), "error": round(error, 4), "tiles": len(values)}
