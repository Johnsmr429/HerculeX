# Spiral

Native port of the open spiral fitter by Henderson (WACV 2026, [villa](https://github.com/ScrollPrize/villa)
`spiral-fitting` at `7e6b82164`, MIT) for Apple silicon. It fits a canonical Archimedean spiral,
deformed by a two-stage diffeomorphic flow, a z-varying linear map and a per-winding gap
expander, to surface tracks and Lasagna normals.

## Build

```sh
fit=$(./run.sh port spiral spiral_fit)
```

Requires Homebrew `lz4`. Runs on the CPU; `--metal` runs the flow integration, lattice upsampling, AdamW and the per-point transforms on Metal (`--points cpu` keeps the per-point transforms on the CPU). The CPU is faster on the PHerc0211 band: 0.17–0.21 s per step against 0.29–0.33 s with `--metal`, timed alternately on the same loaded machine.

## Usage

```sh
"$fit" fit --dataset DATASET --out DIR [--z 6900:7100] [--steps 1500] [--schedule N] [--seed 1] [--no-patches] [--metal [--points cpu|gpu]]
"$fit" crossings --dataset DATASET --out DIR [--z 6900:7100]
```

`--schedule N` sets the learning-rate decay horizon when it differs from `--steps`.

`DATASET` follows the fitter's layout: `spiral-scroll.json`, `umbilicus.json`, the tracks DBM with its packed `.vctracks` store and `.crossings.npz` sidecar, and the Lasagna `nx`, `ny` and `grad_mag` stores. The fit writes its parameters to `DIR/params/` as `.npy` and one tifxyz surface per winding to `DIR/meshes/wNNN/` in the original's export layout; `spiral_fit export --dataset DATASET --params DIR --out DIR` writes the surfaces of saved parameters. Supported: the track radius, dense normal, symmetric Dirichlet, minimum spacing and umbilicus losses, and verified patches. Patches are read from the dataset's `verified_patches/` (or `paths.verified_patches` in `spiral-scroll.json`) as tifxyz directories, eroded and restricted to the band as the original does; `--no-patches` ignores them. With patches, track points within 16 voxels of a patch vertex are dropped and the crossings remapped onto the clipped tracks, each patch's seam is unwrapped by integer potentials along a spanning tree refreshed every 100 steps (patches whose potentials are path-dependent are excluded), and the patch radius loss (weight 8) holds each sampled patch to one winding. `crossings` writes the crossings found from the tracks themselves beside those of the dataset's crossing file; the fit uses that file when present and finds them itself otherwise. Not ported: the patch and track distance-transform losses (from step 25,000; a longer fit with patches is refused), point collections, fibres and the outer shell.

## Verification

The historical goldens compared transforms, loss values and gradients on CPU and Metal.
The clipped tracks, crossings and seam potentials matched the original on the
`50899f48` patch at z 5000–5200 (losses within `1e-4`, gradient cosine `>= 0.9999995`, 1,500 steps in 5.1 min vs 30.7 min on M1 Pro).
