# synthetic/injection

Known structures injected into real scrolls, so that what the pipeline makes of them can be checked against exact
truth. It works the way transit injections work for Kepler light curves: the real data keeps its real noise,
artefacts and neighbouring material, and only the injected signal is synthetic. What exists, what the measurements
show and what comes next are in `TODO.md`.

| Kepler (batman, juliet, KeplerX) | Scrolls |
| --- | --- |
| A real light curve: flux against time, real noise and systematics. | A real CT volume: grey level against position, real noise, artefacts and sheets. |
| batman: a transit from a few physical parameters, exact geometry, averaged over the exposure. | `papyrus.c`: papyrus structures from the anatomy's parameters (plies, fibres, sheets, kolleseis, delamination, fraying, breaks, contacts), exact geometry, averaged over each voxel. |
| The instrument response: Kepler's pixel response function, scaled by the star's flux. | The scan's response: blur, phase fringes at edges and the grey levels of papyrus and air, measured on the scan. |
| juliet's noise model: white jitter plus a correlated Gaussian process, per instrument. | `profile.c`: white noise plus anisotropic correlated noise (across the axis and along it), per material and per scan. |
| KeplerX's injection: real flux plus a delta, from a frozen seeded plan. Physics and coverage decide eligibility; truth and provenance are kept per injection. | Planned workflow: real CT plus a delta, from a frozen seeded plan. Sites qualify by the real CT's physics (a gap wide enough, a sheet to split), never by pipeline output. Truth, hashes and a claim boundary are kept per plan. |
| Recovery: does detection find the injected transit at its period? Completeness against depth and S/N. | Planned recovery test: does the sheet detector and the level fit find the injected structure and assign it the right winding? Completeness against contrast, thickness and gap. Fidelity: do injected structures match their real kind? |

## Files

| File | Description |
| --- | --- |
| `profile.c` | Measure and check air noise, the grain response, papyrus levels, and slab relief. |
| `papyrus.c`, `render_apple/papyrus.metal` | Render a papyrus preview through the scan's blur on Apple. |
| `fragment.c`, `render_apple/fragment.metal` | Place papyrus fragments in real CT gaps and record their truth on Apple. |
| `fidelity.c` | The same measurements on injected sheets and on real ones in the same real blocks. |
| `render_apple/` | Fragment and papyrus shaders and their shared Metal host. |

The library, profile, and fragment commands fetch missing level-0 CT chunks into the shared cache. Use the job
board on Apple for fetches and Metal rendering.

## Injection kinds

Kinds come in order of what they need and what they test.

1. **Grain.** A mineral grain, a small dense blob in the air. It tests the scan response: injected grains must match the real
   grains seen in the slabs, as point sources calibrate a PSF.
2. **Fibre.** A detached fibre bundle lying in a gap. It tests fibre diameter, density and visibility against real detached fibres.
3. **Fragment.** A patch of papyrus floating in a real gap: two plies, recto fibres along the roll and verso fibres along the axis. It tests
   sheet fidelity; recovery is whether the detector finds it.
4. **Kollesis.** A second sheet laid on a real sheet over a strip, the same winding, its interface sometimes open.
5. **Delamination.** A real sheet split inside by moving one ply outward and filling the opening with real air. The
   same winding.
6. **Contact.** Two real neighbouring sheets pressed together by a known displacement of the real CT, closing their
   gap over a patch. They stay different windings. This is the case the level fit gets wrong today.
7. **Break.** The real CT cut along a plane across windings and shifted by a known slip. Truth is the correspondence
   across the cut.
8. **Fraying.** Fibres peeling from a real sheet's edge or face.

Kinds 5 to 7 move real material by a known displacement field: the field is the injected signal, as a transit-timing
variation moves real transit times. The opening is filled with real air, and the variance that resampling removes is
added back from the noise profile.

## Order of work

1. `profile.c` measures each scan. Air gives the mean, the white noise and the correlation lengths across the axis and
   along it. Real grains and sheet edges give the blur and the phase fringe. The real slabs give papyrus grey levels and
   ply and fibre contrast. It is verified by drawing noise from the profile and comparing its spectrum and histogram
   with real air.
2. `papyrus.c` renders grains, fibres and fragments, averaged over each voxel and passed through the scan's response,
   on the GPU. It is verified by comparing injected grains with real grains, and injected fragments with real sheets
   (profiles across them, face texture).
3. `fragment.c` places a fragment in real CT and returns its truth.
4. `fidelity.c` compares crops of real and injected sheets from the same blocks.

The frozen injection plan, additional injection kinds, and detector recovery scoring are future work.
See [the injection TODO](TODO.md) for the measurements and outstanding work.

```sh
./run.sh synthetic.injection.profile --help
./run.sh synthetic.injection.fragment demo OUT.png [--seed N]
./run.sh synthetic.injection.fidelity check SCROLL [--blocks N] [--seed N] [--out OUT.json] [--rows OUT.npz]
```

## Claim boundary

Injected structures test whether the pipeline finds and assigns structures of known kinds in real scans. They don't
test the kinds the plan doesn't contain, the global shape of a scroll, or ink.
