# ink/model

```text
weights.py           stock Reader v3 fetched at its pinned revision, checked and converted for the tile reader
stock_model.py       the network built from the open villa code at a pinned commit; the open ink_9um models
tile_reader.py       the network on 512-pixel tiles every 384: stem norms fixed per stack, body norms per tile
depth_stem.metal     the 3-D stem as one Metal kernel, run ahead of the engine on its own thread
neural_engine.py     the 2-D body compiled once per weights for the Neural Engine, half precision
engine_calls.m       engine called from C, since coremltools holds Python's lock for a whole prediction
input_stack.py       the input recipe: the 17-layer window from the written face, raw values scaled
tile_filling.c       a tile's empty columns filled from shifted, flipped copies of its own stack
portable_reader.py   the same arithmetic in PyTorch, float32, on CPU or CUDA: the reference for a port
```

A tile's empty columns are filled from shifted, flipped copies of its own stack, so per-tile norms do not read a hole's
edge as ink. On the Neural Engine, half precision keeps every ink probability within 0.03 of the float network (99th
percentile 0.015).
