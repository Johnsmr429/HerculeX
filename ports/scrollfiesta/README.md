# ScrollFiesta

Apple silicon port of ScrollFiesta ([upstream](https://github.com/Hob3rMallow/scrollfiesta_public), commit `3c46077`).

## Build

```sh
./run.sh port scrollfiesta grid_pipeline
./run.sh port scrollfiesta sheet_assemble
./run.sh port scrollfiesta vmesh_tifxyz
./run.sh port scrollfiesta vcr_gpu
```

Requires Homebrew `libomp` and `libtiff`. LAPACK comes from Accelerate.

## Changes from upstream

- Only the files the tools compile are kept. Comments are removed from ScrollFiesta's own sources; vendored libraries in `deps/` are unchanged.
- `grid_pipeline` is compiled with OpenMP, so cubes are meshed in parallel.
- The AVX2 MLS kernel runs on NEON (`src/common/avx2_neon.h`) with results bit-identical to the x86-64 build.
- `vcr_gpu` runs on Metal (`src/common/cuda_drv_metal.m`, `src/flatten/vcr_render_metal_kernels.inc`).
- C sources compile with `_DARWIN_C_SOURCE` on macOS.

## Verification

End to end on a PHerc0846A box, `grid_pipeline`, `sheet_assemble`, `vmesh_tifxyz` and `vcr_gpu` write the same bytes as the upstream sources built with these changes; only timings and paths in their logs differ. The NEON MLS kernel's sums equal the x86-64 AVX2 build's.

## Licenses

ScrollFiesta is MIT (`LICENSE`). Vendored code is listed in `THIRD_PARTY_LICENSES.md`; GCoptimization is for research use only and Triangle may not be used commercially.
