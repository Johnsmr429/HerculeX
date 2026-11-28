# unroll

`unroll` is HerculeX's default trace 1.5 pipeline. It turns a scroll's surface prediction (such as the published `m7` OME-Zarr) into one laid-out papyrus strip, runs the shared ink stage, and publishes a viewer page.

## Prerequisites

Run commands from the repository root after `./run.sh setup`.

| Requirement | Command or setup |
| --- | --- |
| Scan catalog entry | `./run.sh catalog SCROLL` |
| Hand-verified curl (`1` or `-1`) | `./run.sh curl SCROLL SCAN 1` (or `-1`) |
| Umbilicus centerline | Automatic if the open-data catalog includes one (e.g., `PHerc0125`, `PHerc0826`); otherwise place centerlines in `records/umbilici/vesuvius-center-lines/` or provide `sheet/axis.json` (see [level/level](../level/level/README.md#curl-and-umbilicus)) |
| Surface prediction | Published `m7` prediction, or pass `plan --m7 URL` with a surface Zarr level near 9 µm |
| Voxel size | 8.6–9.4 µm (parameters are in voxels, tuned at 9.362 µm) |

Software: Python 3 with `numpy`, `scipy`, and `pillow` (`blosc` is optional when `libblosc` is installed via `./run.sh setup`); `json-c` and `libtiff` via `pkg-config`. On Linux CUDA, `nvcc` 12+ and an NVIDIA GPU of compute capability 7.5+. For remote Colab tracing, the signed-in `colab` CLI.

Typical PHerc1545 (9.362 µm) footprint: ~21 GB unit graphs, ~3 GB stitched pieces, ~19 GB ink faces, ~20 GB viewer tiles; peak RAM ~18 GB (`stitch`, `read`), ~21 GB (`layout`, or ~11 GB with tighter joints), ~3.5 GB (`preview`). On 16 GB Macs, run `stitch`, `layout`, and `read` one at a time via `--queue`.

## Quick start

```sh
./run.sh catalog PHerc0826
./run.sh curl PHerc0826 SCAN SIGN
./run.sh unwrap --queue PHerc0826:SCAN
./run.sh board status
```

- Omit `--queue` to run all stages (`plan → trace → stitch → layout → root → read → page`) in the foreground.
- Use `--from STAGE` (`plan|trace|stitch|layout|root|read|page`) to resume without overwriting earlier stages (for example, `./run.sh unwrap --queue --from read SCROLL:SCAN`).
- Add `--rotate180` to rotate the strip 180° when a scroll reads upside down.
- Run `./run.sh unroll SCROLL SCAN preview [--rotate180]` after `layout` to render the strip from the CT atlas before committing to a full ink read. Fetch atlases first with `./run.sh scans.transfer.fetch_atlas OUT`.
- After `page` finishes, run `./run.sh serve` and open `http://localhost:8765/sheets/index.html`.

## Pipeline stages

```text
m7 surface prediction
  └─ plan    keep umbilicus, build L4 papyrus mask, plan 768-plane slabs (step 576, 192 overlap)
  └─ trace   extract 8-voxel surfels, cut forks, bridge gaps, solve turns (8 reweighted min-cuts)
  └─ stitch  match 192-slice seams, solve global windings + radial refine, cut & flatten tifxyz pieces
  └─ layout  map spiral angle to arc length, consensus-shift outliers, solve joints, pack strip
  └─ root    link pieces into 2,560-plane band units in a HerculeX outputs root
  └─ read    run ./run.sh ink with UNROLL_INK (fetches CT up to 60 GiB ahead)
  └─ page    keep per-model face links, compose tiles, publish viewer page and start-page card
```

Run an individual stage with `./run.sh unroll SCROLL SCAN STAGE [ARGS]`:

| Stage | Action | Main outputs | Typical time (PHerc1545) |
| --- | --- | --- | --- |
| `plan [--m7 URL]` | Keep umbilicus, build papyrus mask, write unit plan. | `WORK/{scan,axis,units_plan}.json`, `papyrus_mask_L4.*` | Minutes |
| `trace [UNIT...]` | Prefetch masked m7 slabs one unit ahead and run `unroll-trace` (`--skip-emit`). Skips finished units. | `WORK/graphs/UNIT/surfels.bin`, `units/UNIT/record.json` | ~15 min/unit (M1 Pro); 64 s trace overlapped with fetch (RTX 5090) |
| `colab SESSION [UNIT...]` | Pack sources and trace units on a Colab runtime. | Remote graphs | 15–90 min/unit (Colab CPU High-RAM) |
| `colab-fetch SESSION` | Pull completed graphs from Colab. | `WORK/graphs/`, `units/`, `logs/` | Minutes |
| `stitch [--partial] [OPT...]` | Run `unroll-stitch` across planned units (requires complete set unless `--partial`). | `RUNS/pieces/piece-NNNN/`, `record.json` | 24 min (M1 Pro) |
| `layout [OPT...]` | Run `layout.py` strip placement. | `RUNS/layout/{scroll_layout,layout_stats}.json` | 6 min (M1 Pro) |
| `preview [--rotate180]` | Render strip from the CT atlas without running ink. | `RUNS/layout/preview.png`, `preview_overview.jpg` | 2 min |
| `root [--rotate180]` | Materialize HerculeX outputs root. | `RUNS/root/` | 2 min |
| `read [UNITS]` | Run `./run.sh ink` on all or named band units. | `RUNS/root/.../faces/`, `RUNS/bricks/` | 4.1 h (Reader v3) / 7.2 h (Reader-PU) |
| `page` | Compose tiles, publish viewer page, add start-page card. | `RUNS/root/website/`, `RUNS/faces/` | 5 min |
| `all` / `queue` | Run or queue all stages (`--from STAGE`, `--rotate180`). | `WORK` and `RUNS` | — |

`unroll-trace` and `unroll-stitch` rebuild automatically via `build.sh` when sources change, print all CLI options when invoked without arguments, and produce byte-identical graphs and pieces across CPU and CUDA runs (`STITCH_FLATTEN=gpu` is faster but not byte-identical). `unroll-trace --field DIR` accepts external model predictions (`surface.npy`, plus optional `barrier.npy`, `normals.npy`, `phase.npy`, `aff.npy`, `spacing.npy`).

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `UNROLL_WORK` | `data/library/cache/unroll/SCROLL-SCAN` | Work directory (plan, masks, unit graphs, logs). |
| `UNROLL_RUNS` | `data/outputs/runs/unroll/SCROLL-SCAN` | Run directory (pieces, layout, HerculeX root, bricks, faces). |
| `UNROLL_INK` | `reader-v3` | Ink model for `read` and `page` (see `./run.sh models`). |
| `UNROLL_PAGE` | Level page name with `-level` → `-trace` | Viewer page name. |
| `UNROLL_SURFACE` | `curl ±1` | Surface label shown on the start-page card (fit is always `trace 1.5`). |
| `UNROLL_SET` | unset | Stores unit graphs and logs under `WORK/sets/NAME` to compare trace sets. |
| `UNROLL_ANCHOR` | `0` | Set to `1` to seed each unit's turns from previously traced neighbors (off by default: increases seam agreement but splits side-by-side pieces). |
| `UNROLL_TRACE_ARGS` | unset | Extra arguments passed to `unroll-trace` (use `WORK/trace_args.txt` for Colab). |
| `OMP_WAIT_POLICY` | `PASSIVE` | Prevents OpenMP spin-waiting under shared load on Linux. |
| `STITCH_CUDA` | `1` (CUDA builds) | Set to `0` to run `unroll-stitch` on CPU. |
| `STITCH_FLATTEN` | CPU | Set to `gpu` to flatten on CUDA (faster, not byte-identical). |
| `STITCH_TIMES` / `STITCH_CHECK` / `STITCH_DUMP` | off | Per-stage timing, CPU/CUDA parity checksums, or piece/seam diagnostics. |

Common tunable flags: `stitch --bridge N` (default `3` grid cells = 24 voxels), `stitch --neighbours W` (default `4` with `--neighbours-trust`), and `layout --joint-contacts N --joint-radius R` (default `10` contacts within `48` voxels; `20`/`20` lowers p90 joint residual from 1.05 mm to 0.44 mm on PHerc1545 with fewer joined groups).

## Hardware lanes

- **CUDA (Linux):** `build.sh` compiles for `sm_80`, `sm_89`, `sm_120`, and `compute_75` PTX with `--fmad=false` for CPU/GPU parity. `unroll-trace` runs crests, 8 concurrent draws (`--draw-jobs 8`), and free-unit min-cuts (≥ 65,536 patches) on GPU; anchored units use CPU min-cut. Disable selectively with `--no-cuda`, `--no-cuda-crest`, or `--no-cuda-flow`. `unroll-stitch` accelerates seams, rays, edge gating, components, and bridges on GPU and flattens on CPU.
- **Apple (macOS):** Runs native CPU paths via Grand Central Dispatch (no Metal kernels yet).
- **Colab:** `colab/drive.sh up SESSION L4` provisions a GPU runtime instead of the default CPU High-RAM box; run `colab/keepalive.sh SESSION` to prevent idle GPU disconnects.

## Outputs and formats

- `surfels.bin`: two `int64` headers (`surfel_count`, `piece_count`), followed by 60-byte surfel records (`x, y, z, nx, ny, nz, score, cos_w, sin_w, spacing, theta` as `float32`; `cell, count, component, winding` as `int32`), then one `int32` piece index per surfel (`-1` if dropped).
- `pieces/piece-NNNN/`: `x.tif`, `y.tif`, `z.tif` (`float32` scan voxels on an 8-voxel grid, `-1` unknown), `w.tif`, `phi.tif` (`NaN` unknown), `quality.npy`, and `meta.json`.
- `layout/layout_stats.json`: `area_cm2`, `unrolled_once_cm2`, `unrolled_multi_cm2`, `overlap_share`, `strip_length_mm`, `joints`, `joint_resid_mm` (p50/p90), `groups_4cm2`, `area_groups_4cm2`, `largest_group_cm2`.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `is missing its hand-set curl or kept umbilicus` | Run `./run.sh curl SCROLL SCAN 1` (or `-1`) and ensure an umbilicus exists (`./run.sh level.level.umbilicus SCROLL SCAN`). |
| `is not in the scan catalog` | Run `./run.sh catalog SCROLL`. |
| `has no published m7 prediction` | Pass `plan --m7 URL` pointing to a surface Zarr level near 9 µm. |
| `at most 64 units` | `unroll-stitch` uses a 64-bit unit mask (~37,000 planes); stitch the scroll in subsets. |
| `FAIL fetch` in unit log | S3 download failed after retries; rerun `trace` (finished units are skipped). |
| 16-byte unit graph | Papyrus mask excluded the slab; `stitch` skips empty graphs automatically. |
| `pieces still need ink` | Finish `read` before running `page`. |
| Upside-down or mirrored page | Rebuild `root` with or without `--rotate180` and verify orientation on a known passage. |

## Files

| File | Role |
| --- | --- |
| `unwrap.sh` | Top-level `./run.sh unwrap` CLI wrapper (`--queue`, `--from`, `--rotate180`). |
| `pipeline.sh` | Orchestrates all `unroll` stages. |
| `trace.c`, `trace_cuda.cu`, `trace_cuda.h` | `unroll-trace` native CPU/CUDA slab tracer. |
| `stitch.c`, `stitch_cuda.cu`, `stitch_cuda.h` | `unroll-stitch` native CPU/CUDA multi-slab stitcher. |
| `build.sh` | Compiles `runtime/bin/unroll-trace` and `runtime/bin/unroll-stitch`. |
| `config.py`, `plan.py`, `fetch.py`, `units.sh` | Scan config, papyrus mask/unit planning, S3 slab fetch, and local trace loop. |
| `layout.py`, `preview.py`, `root.py` | Strip layout, CT-atlas preview, and HerculeX outputs-root adapter. |
| `colab/{drive,setup,worker,graphs,keepalive}.sh` | Remote Colab tracing helpers. |
