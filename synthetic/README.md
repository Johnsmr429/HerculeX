# HerculeX synthetics

Run these research tools on demand from the HerculeX directory. On Apple, use the job board for GPU work,
downloads, and long runs.

| Module                | What to use it for                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `synthetic.two_d`     | Generate independent Synthetic-Slices CT tiles with sheet centrelines and per-tile IDs for 2D detection and tracing.             |
| `synthetic.three_d`   | Generate procedural CT scroll bands with continuous geometry and winding truth for sheet detection, level fitting, and assembly. |
| `synthetic.injection` | Place known 3D fragments into real CT and measure their fidelity for injection-and-recovery studies.                             |

**2D tiles.** This queues a small batch on Apple. On CUDA, run the generator command directly.

```sh
./run.sh board add synth-tiles --gpu --link --memory 4 -- \
    ./run.sh synthetic.two_d.generate --n 4 --style PHerc0211 --out data/outputs/synthetic/slices/example
./run.sh synthetic.two_d.discriminator REAL.npz GENERATED/shard_*.npz
```

The output directory must be new. Each shard contains the image, binary centrelines, sheet IDs, distances, style,
and seed. The discriminator expects a `crops` array of matching uint8 real CT tiles. The pinned weights are cached
under the library's `weights/synthetic-slices/`. Use `--weights FILE` for a local checkpoint. The code is MIT and the
released weights and tiles are CC BY-NC 4.0. [Code licence](two_d/LICENSE) · [Weights and released tiles](https://huggingface.co/vincentl065/Synthetic-Slices)

Use `synthetic.two_d.layout` with the same layout options to generate the condition shards without inference.
Use `synthetic.two_d.generate` with the pinned weights, a local float32 safetensors checkpoint,
or a `torch.save` ZIP checkpoint containing a state dict. A ZIP checkpoint can wrap the state dict
under `model`. The generator selects Metal on Apple or CUDA on Linux when available. Use
`--device cpu` for CPU inference. The discriminator trains the crop CNN on the CPU.

**3D bands.** Build one seeded band, run the sheet detector and level fit, and score the result against its known
windings. The band goes to `OUTPUTS/synthetic/bands/round-normal-s7/`, alongside existing bands.

```sh
./run.sh board add synth-band --memory 4 --cores 2 -- \
    ./run.sh synthetic.three_d.batch round/normal:7 --slices 128 --device cpu
./run.sh synthetic.three_d.bench round-normal-s7
```

Run the scorer after the build finishes. Use `synthetic.three_d.families list` to see the available
geometries and conditions, or `synthetic.three_d.band render SPEC.json OUT` to write just the CT bricks.
Rendering uses Metal on Apple and the native CPU renderer on Linux. `--device cpu` selects the CPU on Apple.

`./run.sh synthetic.three_d.solver_view fit SCROLL` fits 32-cell-plane windows of a training control.
`./run.sh synthetic.three_d.solver_view compare SCROLL` compares detector and solver patch features with
the synthetic bands for that scroll. The logistic and neural classifiers hold out whole windows and bands
and report AUC before and after standardizing each band. The report is saved under
`OUTPUTS/synthetic/realism/solver/`.

**Calibration and replay.** The 9.362 µm scans at 113 keV and the 8.64 µm scans at 116 keV
share a 1.2 m propagation distance. The 7.91 µm scans use 53–54 keV. Tuning crops stay inside
the tabulated mid-radius window. The training reference uses the per-scroll 5th, 25th, 50th,
75th and 95th percentiles of radial pitch and the fraction of truth with another winding
within 80 µm along its normal. These measurements come from the labelled training windows. Split plies can resemble closely spaced thin
sheets in a crop. Unlabelled targets supply CT appearance at detected sheet nodes. Holdouts and
PHerc0343, which may be the same roll as holdout PHerc0343P, are excluded.
`HERCULEX_SYNTHETIC_LOOKS` selects an alternative looks file.

`./run.sh synthetic.three_d.tune PROFILE --round NAME` fits synthetic settings to cached real crop
features and training geometry. Use `tune finish --profiles SCROLL --round NAME` to rescore the
best candidate from an interrupted search and `tune publish --profiles SCROLL --round NAME` to
write that round's `looks.json`. The trail and looks are saved under `OUTPUTS/synthetic/realism/`.
The optional `--cnn` score for a training scroll needs cached labelled CT crops.

Older specs retain the geometry and texture of their version. Version 4 changed RK4 from
eight steps to four. Version 5 capped axial knot spacing at one fifth of the finest roll
correlation and one sixth of a flow's axial scale. Version 6 normalized outline harmonics
by their mean around a turn and made folds slide their layers to retain length. Version 7
turned the verso fibres along the axis. Earlier versions transposed the verso texture back
along the roll.

The native evaluator accepts the previous voxel's turn when its radii bracket the next voxel.
It skips the saturated error function beyond four (error below 1.6 × 10⁻⁸) and Gaussian terms
past seven standard deviations (below 3 × 10⁻¹¹). Voxel coordinates use z, y, x in µm.
Point queries return CT, then radius, angle, stretch, three normal components, lower turn and
fraction. Each neighbouring sheet contributes its index, presence, distance and thickness.
The twelve entries of a crack are centre y and x, normal y and x, half-length, opening, slip,
wedge, wiggle amplitude and wavelength, followed by two reserved slots.

**3D gallery.** On the Apple lane, `./run.sh synthetic.three_d.gallery` renders injected scrolls
beside their source CT and links the gallery from the local website. Use
`synthetic.three_d.inject render FAMILY SEED OUT.npz` to keep a single labelled CT crop.

**Real-CT injection.** Prepare a slab library and scan profile, then use `synthetic.injection.fragment demo` to
inspect a labelled fragment or `synthetic.injection.fidelity check` to measure it. The injection guide covers
preparation and the status of recovery scoring. [Real-CT injection guide](injection/README.md)
