# calibration

Unwrapping scored on controls with known truth, so a replacement for any stage can be measured against the stage it
replaces.

```text
unwrap/   sheet detection and unwrapping against published-mesh truth and a synthetic scroll of known windings
```

Controls come from the library's `controls/`; each study writes `calibration/STUDY/` there and its measured results
into `records/surface/`. A number goes into a record only after its control has been run alongside it.
