# ink

Stage 6 and the ink models: layer stacks to ink maps, and the data and runs that teach the model.

```text
model/      the networks: stock open models, the taught tile reader, its Neural Engine build
reading/    the ink stage over every piece, and re-reads of circled spots
datasets/   every labelled dataset, its builder and dataset_check.py
training/   the training recipe, left-out runs, the second reader on the map
```

The network: 34.5M parameters, a 3-D stem over 17 layers into a 2-D residual U-Net, begun from `scrollprize/ink_9um`.
Weights in the library's `weights/`: `pretrained/` (ink_9um, Reader v3), `production/`, `trained/` (every run), `engine/` (Neural Engine builds, made on first use).
Reads use stock Reader v3 (`./run.sh ink.model.weights fetch` downloads it); set `HERCULEX_WEIGHTS` to read with another checkpoint.
