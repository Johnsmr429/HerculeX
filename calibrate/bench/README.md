# calibrate/bench

Scores unwrapped pieces against frozen ground-truth packages built from published Vesuvius meshes: coverage on the correct winding, distance error, pressed/loose recall, joins, and folds.

## Commands

```sh
./run.sh calibrate.bench.package list                         # list controls and package status
./run.sh calibrate.bench.package verify [CONTROL ...]         # verify package and source-mesh SHA-256 digests
./run.sh calibrate.bench.score_check PHerc0139-dev            # self-test scorer on known perturbations
./run.sh calibrate.bench.score CONTROL PIECES ... [--zones val] [--label NAME]
./run.sh calibrate.bench.results report                       # compare recorded runs per control
```

`PIECES` accepts a surface folder, a unit folder of `piece-*` meshes from either fit, or `.npz` surfaces. Winding comes from `w.tif` if present, otherwise `winding` in `meta.json`.

## Controls and holdout policy

`controls.json` defines frozen 7.91–9.366 µm controls split strictly by scroll:

- **Training:** `PHerc0139`, `PHerc0172`, `PHerc1667`, `PHercParis4`.
- **Validation bands (model selection only, never trained on):** `PHerc0139` z 5500–5650 and `PHercParis4` z 7680–7936.
- **Holdout A:** `PHerc0814`, `PHerc0500P2`, `PHerc0841`, `PHerc0009B`, `PHerc1447`, `PHerc0343P`. Score holdout zones only with `--holdout` from a clean commit; runs append to `bench/holdout-log.jsonl` and holdout packages are never rebuilt.

## Outputs (`data/outputs/bench/`)

| Path | Contents |
| --- | --- |
| `controls/CONTROL/` | Frozen truth package (`truth.npz`, `crossings.npz`, `axis.json`, `meshes/`, `control.json`). |
| `scores/CONTROL/RUN/` | Per-run scored ranks, depths, surface labels, and `summary.json`. |
| `results.sqlite`, `holdout-log.jsonl` | Recorded benchmark metrics and immutable holdout log. |
| `qa/`, `checks/` | Truth-to-CT ridge QA and control diagnostics. |

## Files

| File | Role |
| --- | --- |
| `controls.json`, `package.c`, `truth.c` | Control definitions, SHA-256 package builder/verifier, and truth loader. |
| `score.c`, `nearest.c`, `score_check.c` | Surface scorer, normal-cone nearest-point matcher, and scorer self-check. |
| `surface_score.c`, `winding_score.c`, `v1_sources.c` | Extracted-surface/winding-field scorers and tile reassembler. |
| `ridge.c`, `qa.c`, `results.c` | CT ridge QA and SQLite results reporting. |
