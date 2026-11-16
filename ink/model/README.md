# ink/model

```text
weights.py         stock Reader v3 fetched at its pinned revision, checked and converted for the tile reader
stock_model.py     Vesuvius Challenge open models from tools/villa: mean of two seeds
tile_reader.py     the taught network on 512-pixel tiles every 384: stem norms fixed per stack, body norms per tile
turned_reader.py   a tile reader's mean over the eight turns and flips of what it reads
depth_stem.metal   the 3-D stem as one Metal kernel, run ahead of the engine on its own thread
neural_engine.py   the 2-D body compiled once per weights for the Neural Engine, half precision
engine_calls.m     engine called from C, since coremltools holds Python's lock for a whole prediction
patch_reader.py    InkModel.read for the organisers' hybrid network (TRACE v3B and others), the same maps, faster
hecate_mlx.py      Hecate 9.6 um on MLX in float32: the exact port, GPU
hecate_engine.py   Hecate 9.6 um as 2-D convolutions on the Neural Engine, half precision: 87 ms a patch, hecate.py 400
```

Hecate on the engine: depth is folded into the batch, so each 3x3x3 convolution is one 2-D convolution whose three
depth taps are shifted one image within the patch. Group norms reduce one axis at a time (Core ML's CPU and engine get a
two-axis reduction wrong) and are scaled near their RMS, at most 16 of it in the weights (more drives the weights into
half precision's subnormals). A change to the rewrite takes a new `VERSION`, which builds a new program.

A tile's empty columns are filled from shifted, flipped copies of its own stack, so per-tile norms do not read a hole's edge as ink.
