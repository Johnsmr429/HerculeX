"""The PyTorch reader, on CUDA where there is one, reads a stream of stacks as TileReader's callers expect."""

import numpy as np
import pytest
from scipy import ndimage as ni

from ink.model import tile_reader
from ink.model.weights import STOCK_READER as WEIGHTS


def papyrus(side):
    grain = ni.gaussian_filter(np.random.default_rng(7).normal(size=(17, side, side)), (1, 2, 2))
    return np.clip(110 + 300 * grain, 1, 255).astype(np.uint8)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs Reader v3: ./run.sh ink.model.weights fetch")
def test_a_gpu_stream_of_stacks_reads_as_each_stack_alone_and_as_the_cpu_reads_it():
    from ink.model.portable_reader import PortableReader

    reader = PortableReader(WEIGHTS)
    stacks = [papyrus(700), papyrus(700)[:, :300, :100].copy(), papyrus(600)]
    alone = [reader.read(stack) for stack in stacks]
    assert all(
        a.shape == s.shape[1:] and np.isfinite(a).all() and a.min() > 0 and a.max() < 1 for a, s in zip(alone, stacks)
    )
    together = list(reader.maps((stack, tile_reader.measured(stack)) for stack in stacks))
    assert all(np.array_equal(a, b) for a, b in zip(alone, together))
    prepared = tile_reader.prepare_stack(stacks[0])
    (read,) = reader.maps([(prepared, tile_reader.measured(stacks[0]))])
    assert np.array_equal(read, alone[0])
    reader.fix_to(stacks[0])
    assert np.array_equal(reader.read(stacks[0], fixed=True), alone[0])
    if reader.device.type != "cpu":
        on_cpu = PortableReader(WEIGHTS, "cpu").read(stacks[1])
        assert np.abs(on_cpu - alone[1]).max() < 0.01
