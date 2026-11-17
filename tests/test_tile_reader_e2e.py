"""The Metal stem and the Neural Engine body read as the network does."""

import sys

import numpy as np
import pytest
from scipy import ndimage as ni

from ink.model.tile_reader import TileReader, measured, scaled
from ink.model.weights import STOCK_READER as WEIGHTS

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal and the Neural Engine; elsewhere see test_portable_reader_e2e.py"
)


def papyrus(side):
    grain = ni.gaussian_filter(np.random.default_rng(7).normal(size=(17, side, side)), (1, 2, 2))
    return np.clip(110 + 300 * grain, 1, 255).astype(np.uint8)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs Reader v3: ./run.sh ink.model.weights fetch")
def test_metal_stem_matches_the_network_and_a_read_covers_the_stack():
    import torch

    reader = TileReader(WEIGHTS)
    stack = papyrus(700)
    read = reader.read(stack)
    assert read.shape == stack.shape[1:] and np.isfinite(read).all() and read.min() > 0 and read.max() < 1
    small = stack[:, :100, :70].copy()
    assert reader.read(small).shape == (100, 70)
    reader.fix_to(stack)
    image = scaled(torch.from_numpy(stack[:, :128, :128].copy()).to("mps"), reader.fixed[0])[None]
    with torch.inference_mode():
        kernel, network = reader.stem(image, reader.fixed).float(), reader.stem.module(image[:, None]).float()
    assert (kernel - network).abs().mean() < 0.01 * network.abs().mean()


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs Reader v3: ./run.sh ink.model.weights fetch")
def test_a_stream_of_stacks_reads_as_each_stack_alone():
    reader = TileReader(WEIGHTS)
    stacks = [papyrus(700), papyrus(700)[:, :300, :100].copy(), papyrus(600)]
    alone = [reader.read(stack) for stack in stacks]
    together = list(reader.maps((stack, measured(stack)) for stack in stacks))
    assert all(np.array_equal(a, b) for a, b in zip(alone, together))


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs Reader v3: ./run.sh ink.model.weights fetch")
def test_sharing_the_depth_stem_preserves_both_tiles_including_their_edges():
    import torch

    reader = TileReader(WEIGHTS)
    stack = papyrus(797)
    reader.fix_to(stack)
    with torch.inference_mode():
        for gap in (1, 137, 280):
            block = stack[:, :512, : 512 + gap]
            for offset, shared in zip((0, gap), reader.stem.tiles(block, reader.fixed, gap)):
                tile = torch.from_numpy(block[:, :, offset : offset + 512].copy()).to("mps")
                separate = reader.stem(scaled(tile, reader.fixed[0])[None], reader.fixed)
                assert torch.equal(shared.cpu(), separate.cpu())


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs Reader v3: ./run.sh ink.model.weights fetch")
def test_the_neural_engine_reads_as_the_network_does(tmp_path, monkeypatch):
    import torch

    from ink.model import neural_engine, stock_model

    monkeypatch.setattr(neural_engine, "COMPILED", tmp_path)
    body = (
        stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[WEIGHTS], device="cpu")
        .models[0]
        .model.model.network.float()
        .eval()
    )
    engine = neural_engine.EngineBody(WEIGHTS, load_body=lambda: body)
    warm = neural_engine.EngineBody(WEIGHTS, load_body=lambda: pytest.fail("compiled body should be reused"))
    features = np.random.default_rng(3).normal(size=(1, 32, neural_engine.SIDE, neural_engine.SIDE)).astype(np.float16)
    with torch.inference_mode():
        truth = body(torch.from_numpy(features).float())["ink"][0, 0].numpy()
    read = engine(features)
    assert np.array_equal(read, warm(features))
    ink, true_ink = 1 / (1 + np.exp(-read)), 1 / (1 + np.exp(-truth))
    assert np.corrcoef(ink.ravel(), true_ink.ravel())[0, 1] > 0.998
    assert np.abs(ink - true_ink).max() < 0.04 and np.percentile(np.abs(ink - true_ink), 99) < 0.02
