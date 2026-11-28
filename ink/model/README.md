# ink/model

Implements checkpoint management and inference backends for the Reader (`reader-v3`, `reader-pu-bhand`) and Studio (`studio`, `studio-jeff`) model families.

## Commands

```sh
./run.sh weights fetch [MODEL]          # download and SHA-256 verify a published checkpoint
./run.sh weights import MODEL FILE      # verify and install a local download
./run.sh weights status                 # inspect checkpoints and optional Apple Core ML body
./run.sh models                         # list registered models
./run.sh models add NAME TITLE FILE [reader|studio]
```

*(Also reachable as `ink.model.weights` and `ink.model.models`.)*

Checkpoints install under `data/library/weights/ink/` and register in `data/library/weights/ink/models.json`. Run `./run.sh check all` to execute native reader golden tests.

## Backends

- **Apple (`reader_apple/`):** Runs the 3D stem on Metal and the 2D body on Metal GPU, or dispatches full tiles to a matching compiled Core ML body in `data/library/weights/engine/` when installed.
- **CUDA (`reader_cuda/`):** Runs the full Reader network natively with CUDA, cuBLAS, and cuDNN while CPU sampling workers prepare the next piece's normalization windows.
- **Studio (`studio/`):** Persistent PyTorch worker (`worker.py`) speaking a binary `HXOK` stdin/stdout pipe (`reader.h`).
- **Studio CUDA (`studio_cuda/`):** Standalone native C/CUDA FP16 Winograd Studio engine (`studio_cuda worker`, drop-in `HXOK` protocol replacement for `studio/worker.py`; see [`studio_cuda/README.md`](studio_cuda/README.md)).

## Files

| File | Role |
| --- | --- |
| `../ink.h` | Weight fetch/verification, model registry, and ink-stage dispatcher. |
| `reader_apple/reader.m`, `reader.metal` | Native Apple Reader backend and Metal kernels. |
| `reader_cuda/reader.c`, `reader.cu` | Native NVIDIA Reader backend and CUDA kernels. |
| `studio/model.py`, `worker.py`, `reader.h` | Studio `DeepResidualUNet`, persistent PyTorch worker, and C pipe client. |
| `studio_cuda/studio_cuda.c`, `studio_kernels.cu`, `studio_safetensors.c` | Native C/CUDA Studio FP16 Winograd engine, safetensors loader, and verification harness. |
