# calibration/ink

```text
ink_yield.py                 AUC and recall at a fixed false-alarm rate
stock_ink_scoring.py         the open two-seed model on labelled crops it was not trained on
taught_model_controls.py     taught weights against the stock seed on true labels
own_mesh_windows.py          left-out labels read on our own pieces instead of published surfaces
surface_ink_retention.py     ink signal a surface keeps: published mesh against our pieces
invention_guard.py           checks that nothing is read on structureless papyrus, far faces and blank papyrus
ink_injection.py             real letters injected into the target's stacks and recovered
tile_grid_scoring.py         AUC over shifted tile grids; tile_reader_study.py, tile reader against patch reader
model_agreement.py           model ensembles
fibre_polarity.py            which face is which, from fibre direction
map_context_invention.py     what the second reader invents
stroke_widths.py             stroke widths, sure share and AUC of a read against finer truths
read_targets.py              a reader's sure ink and blank on our own pieces as targets, against hand labels
agreed_truth.py              a read scored only where hand label and fine map agree; depth shifts, depth means and the published-depth oracle on our pieces
read_comparison.py           two reads of the same pieces: faces taken, sure ink, ink too wide for a stroke, letter-like windows, scrambled control
./run.sh calibration.ink.invention_guard WEIGHTS PATCH PIECES
./run.sh calibration.ink.own_mesh_windows build | WEIGHTS SCROLL
./run.sh calibration.ink.tile_grid_scoring MODEL
```
