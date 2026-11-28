# Surface

Native inference for the two published papyrus-surface networks: m7 (nnU-Net ResEnc-L, 192³ patches) and Recto (ResEnc with scSE, 256³). The forward runs on MLX with custom Metal kernels. Sliding-window blending reproduces nnUNetPredictor for m7 and vesuvius for Recto, including their dtypes and summation order.

## Build

```sh
HERCULEX_MLX_ROOT=/path/to/mlx-sdk ./run.sh port surface surface_predict
```

Use an installed MLX C++ SDK prefix containing its CMake package and libraries. To configure
with CMake directly, pass `-DMLX_ROOT=/path/to/mlx-sdk`. Without an override the build searches
CMake's normal prefixes, then the project's legacy MLX installation. The executable also
links the system libcurl.

## Usage

```sh
runtime/ports/surface/surface_predict --spec recto.json --weights recto.safetensors \
    --store STORE [--source PATH//LEVEL] --box z0:z1,y0:y1,x0:x1 --out DIR \
    [--step 0.5] [--no-mirror] [--crop N] [--fiesta-grid GRID] [--threshold 128]
```

Bricks of the box that the store lacks are fetched from the open-data bucket first; `--no-fetch` reads them as air and `--fetch-only` fetches and stops. A missing store is created from `--source`, a raw uint8 zarr level such as `PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr//0`. A store written by the pipeline's passage reads holds only some z-planes of each brick and is refused.

Writes `DIR/surface.npy` (uint8, 255 is certain surface), `DIR/surface.json` (settings, weights and timing) and `DIR/crops.json`, from which an interrupted run resumes; `--crop N` predicts a wide box in N x N columns. `--fiesta-grid GRID` also writes ScrollFiesta's input grid of 128³ TIFF cubes, for a box on multiples of 128. Both networks mirror every patch eight ways as their authors' predictors do (`--no-mirror` turns it off). `--npy CT.npy` reads a uint8 volume instead of a store, `--logits FILE.npy` writes the blended logits, `--forward` runs one forward pass over the whole input and `--forward-mirrored` one averaged over the mirrored passes. Place weights and network descriptions under `data/library/weights/surface/` (or pass explicit `--spec` and `--weights` paths).

## Memory

A forward whose first-level activation is larger than 320 MB, such as m7 at 192³ or Recto at 256³, runs in z-slabs. Each level of 192 MB or more is held as slabs of about 128 MB, and each level of 512 MB or more is kept in an unlinked scratch file in `$TMPDIR`, written and read ahead on a second thread. Convolutions read one halo plane on each side with the same padding as the whole-volume call, and channel statistics carry each Welford lane's state from slab to slab, so the logits are bit-identical to the whole-volume forward. Recto's eight mirrored passes at 256³ are averaged from a scratch file in the same way. MLX's cache is held to 256 MB, or to `--cache-gb` if that is smaller, during a slab forward.

| Patch | Peak footprint | Forward | Scratch written per forward |
| --- | --- | --- | --- |
| m7 192³ | 1.8 GB | 3.1–3.5 s | 2.7 GB |
| Recto 256³ | 2.2–2.4 GB | 8.5–12.6 s | 12.3 GB |
| m7 192³, `HX_SURFACE_DISK_MB=1024` | 2.65 GB | 2.9 s | none |
| Recto 256³, `HX_SURFACE_DISK_MB=1024` | 2.7 GB | 8.1 s | 6.4 GB |

Measured on an M1 Pro with 16 GB, one forward with the default cache, while other jobs shared the machine. The scratch
volume reaches the disk: for long Recto runs, point `TMPDIR` at a scratch drive.

`HX_SURFACE_WHOLE_MB`, `HX_SURFACE_CHUNK_MB`, `HX_SURFACE_DISK_MB`, `HX_SURFACE_SLAB_MB`, `HX_SURFACE_CACHE_MB` and `HX_SURFACE_MIRROR_MB` change these thresholds; none of them changes the output. `HX_SURFACE_TRACE=1` prints the time and memory of each stage.

## Verification

`tests/test_surface_port_golden.py` compares the logits with the authors' PyTorch networks and their mirrored passes (nnU-Net's for m7, vesuvius's for Recto) on real CT, a 256³ volume with nnUNetPredictor's blended output, and a ScrollFiesta grid and a fetched store with those of the Python tools they replace, byte for byte.
