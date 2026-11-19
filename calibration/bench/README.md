# calibration/bench

The bench scores surfaces from the level fit against frozen truth built from published meshes. It answers one
question the same way for every surface: how much of the truth does a surface cover on the right winding, and where
does it slip?

## Controls

`controls.json` defines the controls. It was frozen on 15 November 2026 and is changed only by adding a version:

- Only scans of 7.91 to 9.366 µm.
- The split is by scroll. Training: PHerc0139, PHerc0172, PHerc1667 and PHercParis4. Holdout A: PHerc0814,
  PHerc0500P2, PHerc0841, PHerc0009B, PHerc1447 and PHerc0343P.
- Validation bands, for choosing models only and never trained on: PHerc0139 z 5500-5650 and PHercParis4 z 7680-7936.
- Holdout zones are scored only with `--holdout` from a clean commit of the ml module and the bench. Every such run
  is appended to `bench/holdout-log.jsonl`. Holdout packages are never rebuilt.

A truth package is one control built from the published meshes alone: the vertices with their normals, a continuous
winding per vertex, flags for pressed, duplicate and conflicting truth, and the sha256 of every source mesh and every
file written.

## Files

| File | Description |
| --- | --- |
| `controls.json` | The controls, their scans, zones and roles. |
| `package.py` | Builds, verifies and lists the truth packages. |
| `truth.py` | Gives the truth of a control, from a package or from a v1 sheet control, with a continuous winding per vertex. |
| `score.py` | Scores surfaces against a package: distance, winding, area, recall at pressed and loose truth, joins and folds. |
| `surface_score.py`, `winding_score.py` | Score extracted surfaces, and score a winding field at the truth vertices. |
| `nearest.py` | Finds the nearest points on the same sheet, within a depth along a normal and a reach across it. |
| `score_check.py` | Checks the scorer on surfaces whose answer is known, made from a package's own meshes. |
| `ridge.py`, `qa.py` | Measure how far the truth lies from the CT ridge, and report on the training truth per package and per mesh. |
| `results.py` | Keeps every scored run and its metrics in one SQLite file. |
| `v1_sources.py` | Puts level-fit tiles back together into the source pieces they were cut from. |

## Commands

```sh
./run.sh calibration.bench.package list                     # controls and whether their package exists
./run.sh calibration.bench.package verify [CONTROL ...]     # package files and source meshes against their sha256
./run.sh calibration.bench.score_check PHerc0139-dev        # the scorer on known answers; exits 1 if a bound fails
./run.sh calibration.bench.score CONTROL PIECES ... [--zones val] [--label NAME]
./run.sh calibration.bench.results report                   # the winding network against v1, one row per control
```

`PIECES` is a surface folder, a unit folder of `piece-*` surfaces from either fit, or a folder of `.npz` surfaces. A
surface's winding comes from its `w.tif` when present, else from the `winding` in its `meta.json`.

## Outputs

These folders are in `bench/` in the outputs folder:

| Path | Contents |
| --- | --- |
| `controls/CONTROL/` | The frozen truth package: `truth.npz`, `crossings.npz`, `axis.json`, `meshes/` and `control.json`. |
| `scores/CONTROL/RUN/` | One scored run: per-vertex ranks and depths, surface labels and `summary.json`. |
| `results.sqlite` | Every scored run and its metrics. |
| `holdout-log.jsonl` | Every holdout run. |
| `qa/`, `checks/` | Truth QA per package, and the truth against the CT ridge. |

Meshes that the library's mirror of the open data holds as empty files were fetched again into `cache/open_data/` in
the outputs folder; `package.py` reads that folder first.
