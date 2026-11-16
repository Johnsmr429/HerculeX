# Ink model

This module runs stock Reader v3 over 17-layer CT stacks. On Apple silicon, it uses Metal for the 3-D stem and the Neural Engine for the 2-D body.

## Files

| File | Purpose |
| --- | --- |
| `weights.py` | Downloads Reader v3 at a pinned revision, verifies its checksum, and converts its state for the tile reader. |
| `stock_model.py` | Builds the open `ink_9um` network that defines the Reader v3 architecture. |
| `input_stack.py` | Prepares and scales the 17-layer input window around the written face. |
| `tile_reader.py` | Reads overlapping 512-pixel tiles and blends their predictions. |
| `depth_stem.metal` | Runs the network's 3-D stem as a Metal kernel. |
| `neural_engine.py` and `engine_calls.m` | Compile and call the 2-D network body on the Neural Engine. |
| `tile_filling.c` | Fills uncovered tile columns from shifted and reflected stack content. |
| `portable_reader.py` | Runs equivalent Reader v3 arithmetic with PyTorch on a CPU or CUDA device. |

## Commands

```sh
./run.sh ink.model.weights fetch
./run.sh ink.model.weights status
```

The fetch command is resumable: it reuses verified weights and only creates missing converted state.
