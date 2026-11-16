"""The NumPy input recipe shared by the ink reader, training, and controls."""

import numpy as np

from sampling.piece_images import FACE_LAYER

DEPTH = 17
PAST_FACE = 2


def profile_of(stack):
    """Mean raw value of each layer over columns the sheet covers in every layer."""
    sample = stack[:, ::8, ::8].reshape(stack.shape[0], -1).astype(np.float32)
    return sample[:, sample.min(axis=0) > 0].mean(axis=1) if (sample.min(axis=0) > 0).any() else sample.mean(axis=1)


def face_layer_of(stack):
    """Where the production window starts on a stack."""
    profile = profile_of(stack)
    middle = stack.shape[0] // 2
    fall = middle - 2 + int(np.argmin(np.diff(profile)[middle - 2 : middle + 8]))
    return min(max(fall - FACE_LAYER, 2), stack.shape[0] - DEPTH - 2)


def scrambled(cube, rng):
    """Every layer with the same random phases added."""
    good = cube[cube.shape[0] // 2] > 0
    phases = np.exp(2j * np.pi * rng.random((cube.shape[1], cube.shape[2] // 2 + 1)))
    out = np.zeros(cube.shape, dtype=np.uint8)
    for layer in range(cube.shape[0]):
        image = cube[layer].astype(np.float32)
        mean = image[good].mean() if good.any() else 0.0
        image = np.where(good, image - mean, 0)
        out[layer] = np.clip(np.fft.irfft2(np.fft.rfft2(image) * phases, s=image.shape) + mean, 1, 255)
    return out
