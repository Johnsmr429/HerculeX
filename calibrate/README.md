# calibrate

The `calibrate` package scores the unwrapping stages on controls with known truth. Use it to measure a replacement
for any stage against the stage that it replaces.

| Folder | Contents |
| --- | --- |
| `unwrap/` | Scores sheet detection and unwrapping against published meshes and against a synthetic scroll of known windings. |
| `bench/` | Scores the pieces of either winding fit against frozen truth packages, on the same controls and splits. |

The sheet controls are stored in the library's `controls/` folder, and the bench's truth packages in the outputs
folder's `bench/`. Each study writes its working files to `calibration/STUDY/` in the library and its measured results
to [`records/surface/`](../records/README.md). Record a number only after you run its control alongside it.
