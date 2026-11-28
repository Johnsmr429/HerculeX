# synthetic/injection: state and next steps

Written 18 Nov 2026 at the end of the first injection day, for the next session with time and compute. All numbers
are measured on PHerc0139 (9.362 µm, package PHerc0139-train), training zone only.

## Goal

KeplerX-style injection and recovery on real scrolls. Known structures go into real CT with exact truth, so that
winding assignment and the global fit can be scored where the bench has no truth (contacts, crushed regions). The
target Adam set on 18 Nov is whole slices of dense, traced synthetic sheets that look real to the eye. The injected
structure must sit in the real CT so that Adam can't pick it out, and the CNN discriminator must score near its
real-against-real control.

## What exists

| File | State |
| --- | --- |
| `profile.c` | Measures a scan: the air noise as its spectrum plus 100,001 quantiles (Gaussian copula), the grain response, papyrus levels and the sheets' relief (the ridge's height over each 128-voxel tile: quadric bends plus a residual spectrum). `check` passes: autocovariance ratios 0.955–1.048 and KS 0.003 against real air. |
| `../library.c` | Slabs of real papyrus along truth meshes, now sampled by cubic spline (it was trilinear), with half-gaps and the ridge shift per sample (`ridge`). The library is rebuilt: 40 slabs, seed 1. |
| `fragment.c`, `render_apple/fragment.metal` | Fragments with a torn elliptic rim. The plane is fitted to the neighbours' pull, so the fragment lies parallel to them, with half the real relief on top. Material comes from slab places that are 90% single isolated sheets (`sheet_rule`, `single`). The kernel reads the slab by Catmull-Rom at the voxel's foot point and its perpendicular distance, blurs both cuts with the grains' width, blends while keeping the variance, and writes labels and truth records with hashes. |
| `fidelity.c` | Applies the same code to real and injected single isolated sheets in the same blocks: profile, peak, width, flank, outer noise, a CNN discriminator split by block, and a real-against-real control. |
| `papyrus.c`, `render_apple/papyrus.metal` | An analytic papyrus with two plies of fibres and closed-form blur. It came out too regular and too faint (peak 92 against 120), so it is parked; the slabs give the material instead. |

## Fidelity runs (24 blocks, seed 0)

| Run | Change | AUC | Control | Findings |
| --- | --- | --- | --- | --- |
| fid1 | Valley cut, tapered from one response width inside | 1.00 | 0.51 | Injected sheets too thin (width 4.0 against 5.5 voxels), flank at air level. |
| fid2 | Taper centred on the valley, clean slab places | 0.99 | 0.57 | Width matched. Half the real "isolated" points were two sheets touching (a second peak 5–6 voxels out), hence `sheet_rule`. |
| fid3 | `single` rule on both kinds, cubic library | 0.96 | 0.48 | Injected sheets thinner (4.5 against 6.0) with a lower peak. The kernel lost the flank: depth was taken along the base normal, and the fitted surfaces were too steep (median slope 0.24–0.45; real sheets are about 0.2 at most). |

The plane re-fit, 9-voxel smoothing and the perpendicular distance and foot point in the kernel are in the code but
were not re-measured: the slope check was stopped part way.

## What the measurements say

1. **A real sheet facing void air** (`flank.py` in the session scratchpad; the method is below) has a steep edge, 120–126 two
   voxels inside the 70 crossing. Past it lies a deep phase fringe, 37–40 at 1–2 voxels beyond the crossing against void air at 52,
   back to 52 by 5–6 voxels. The fine noise in that air is 2.1–3.4 (the spread about a profile smoothed over 2 voxels, sampled every
   0.5 voxel).
2. **The stored slabs keep that edge** (127 inside, fringe 36–38 at 1 voxel). Their air further out sits at about 48,
   not 52, because they face 10–20-voxel gaps rather than voids. Pasted into a void, that is the darker band Adam saw
   ("its black space doesn't fit the dark grey of the empty space").
3. **Injected fragments fill in the fringe** (46–47), soften the edge (107–109) and smooth the grain (1.5–2.3). Each
   slab passes through three interpolations before it lands: along the mesh normals, when centring, and in the kernel.
4. **A kernel test with a tilted flat fragment in clean air** (single cells, cap 8) matches the slab in the core. On the
   flanks it falls 10–15 below the slab from 4 voxels out, because the extents are interpolated bilinearly between a
   single cell (8) and non-single neighbours (3–5).
5. Real sheets in PHerc0139: about 38% of truth vertices have 10 voxels of room on both sides, and about half of those
   are two sheets touching. Single isolated sheets are about 19% of slab cells and of truth vertices.

The method for item 1: real truth vertices with inward normals; profiles every 0.5 voxel; the edge is where the profile
smoothed over 1 voxel first drops below 70 past the peak; a side faces a void when nothing rises above 70 from 6 to
18 voxels past the edge.

## Next steps, in order

1. **Read the scan once, not three times.** Keep each slab's geometry (mesh points and normals every 4 samples,
   the ridge shift per sample). In the kernel, map each voxel to its source point P(row, col) + (shift + depth) N and
   sample the original CT box once, with a prefiltered cubic B-spline or Lanczos-3. Read the box from the volume per
   fragment. Expect the edge, fringe and grain to survive. Check against item 1 above.
2. **Inject the sheet over its own air, not its air.** Either add the delta value = site + alpha (source − A_source),
   with A_source the source's local air level (robust, smoothed), or draw source places from the same environment as
   the site (a void beside a void). Then the fringe stays relative and the 48-against-52 band goes.
3. **Extents per cell from the source's own edge**: the 70 crossing plus the fringe (about 5 voxels). No bilinear mixing
   with neighbouring cells; take the nearest cell or the minimum.
4. **Grain.** Measure the injected air's fine noise against real air beside sheets. If it is still short, add white
   jitter (juliet's sigma_w) at a calibrated level.
5. **Re-run** the slope check, `fidelity.c` (24 blocks) and the edge-to-void measurement. Show Adam zoomed z-slices
   at about 2.5x, real and injected in the same block. The bar: he can't find the fragment, and the AUC sits near the
   control.
6. **Then the package**:
   - a frozen seeded plan (KeplerX style);
   - materialised blocks with truth: label volume, surfaces, records with hashes and a claim boundary;
   - the viewer section showing injections (before and after swipe, truth overlay, 3D).
7. **Then the hard kinds, one at a time, each verified**:
   - contact: a fragment pressed against a real sheet, with the gap as the knob, from open through touching to fused;
   - kollesis;
   - break: a strip of a real sheet replaced by air with its fringe, the ends displaced;
   - fraying;
   - delamination.
8. **Recovery**: run the pipeline's node detector (`level.detection.detect_nodes store`) on injected blocks. A point counts as recovered when a node lies within 1.5 voxels with its normal within
   20°. Report recovery against the knobs: contrast, size, room and gap.

## Commands

```sh
./run.sh synthetic.library build PHerc0139-train --count 40 --seed 1
./run.sh synthetic.injection.profile measure PHerc0139
./run.sh synthetic.injection.profile check PHerc0139
./run.sh synthetic.injection.fragment demo OUT.png --seed 0
./run.sh synthetic.injection.fidelity check PHerc0139 --blocks 24 --out OUT.json --rows ROWS.npz
```

Library, profile and air cubes live in `OUTPUTS/synthetic/library/PHerc0139/`.
