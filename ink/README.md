# ink

The `ink` package samples laid-out pieces and predicts ink maps. Reader v3 is the default; Reader-PU, Studio, Jeff Chen's Studio post-train, and local checkpoints run through the same stage without changing geometry.

## Commands

```sh
./run.sh weights fetch [reader-v3|reader-pu-bhand|studio|studio-jeff]
./run.sh weights import MODEL FILE      # install local file if SHA-256 matches
./run.sh weights status                 # check installed checkpoints and Apple engine cache
./run.sh models                         # list registered ink models
./run.sh models add NAME TITLE FILE [reader|studio]
./run.sh ink SCROLL [SCAN | --scan ID]  # read an existing layout
```

Select a model with `HERCULEX_INK_MODEL=NAME` (level) or `UNROLL_INK=NAME` (trace).

## Supported models

- **Reader v3** (`reader-v3`, default, MIT): 34.5M-param 3D-stem + 2D residual U-Net over 17 CT layers by Domenico Russo (<https://huggingface.co/domenicor046/reader-v3>). Runs natively on Metal/Core ML (`apple`) and CUDA/cuDNN (`cuda`).
- **Reader-PU** (`reader-pu-bhand`, MIT): ScribblePU retune of Reader v3 (<https://huggingface.co/domenicor046/scribble-pu>). Uses the same native reader.
- **Studio** (`studio`, CC BY-NC 4.0): 223M-param residual 3D U-Net over 16 CT layers in `128×128` windows by Priyank Aranke (<https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt>). Runs in a persistent PyTorch subprocess (`model/studio/worker.py`), with a native FP16 Winograd CUDA implementation in [`model/studio_cuda/`](model/studio_cuda/README.md).
- **Studio post-train** (`studio-jeff`, weight license unstated; downloaded directly, not redistributed): Jeff Chen's post-trained Studio checkpoint pinned to SHA-256 `474d0f4be9db7bd18a4a8c8a1a12a0252a72b31dea5bb2edce5189366ec8f2b2`.

## Execution flow

`ink.h` groups pieces by unit (ordered by area), overlaps the first unit's CT fetch with model startup, prefetches one unit ahead when disk allows, and releases each unit's temporary CT store after sampling.

## Subfolders

| Folder | Role |
| --- | --- |
| [`model/`](model/README.md) | Weight fetching, model registry, native Apple/CUDA readers, and Studio worker. |
| [`reading/`](reading/README.md) | Ink stage outputs, settings, CT prefetch (`unit_fetch.c`), and `same_output_models.txt`. |
