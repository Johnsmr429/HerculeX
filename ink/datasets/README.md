# ink/datasets

`labelled_cases.py` builds the labelled ink cases that the letter atlas (`website/atlas_build.py`) reads. Each case
lives in the library's `datasets/labelled_cases/CASE/`: `case.json`, `layers/` (a brick store), `ink.npy`,
`supervised.npy` and, where it was registered, `registration.json`; `manifest.jsonl` lists them.

```sh
./run.sh ink.datasets.labelled_cases
```

CASE is SCROLL-SEGMENT-VOXEL-DISTANCE-ENERGY. `case.json` holds case_id, sample, voxel_um, box, truth and partition
(development or holdout, fixed by a hash of scroll and segment).
