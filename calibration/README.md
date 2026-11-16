# calibration

Each stage is scored on controls with known truth, so a replacement for any stage can be measured against the stage it replaces.

```text
unwrap/   sheet detection and unwrapping against published-mesh truth and a synthetic scroll
ink/      ink models on labelled crops left out of training, own meshes, invention controls
text/     whether maps hold rows of writing; blind pairs and spot tests for the eye
survey/   every scroll read by a model not trained on it, ranked for the prizes
```

Controls come from the library's `controls/` and `datasets/`; each study writes `calibration/STUDY/` there and its measured results into `records/ink/` or `records/surface/`.
A number goes into a record only after its control has been run alongside it.
