# records

Stores committed calibration measurements as JSON beside the code. Record a number here only together with the control it was measured against; place large artifacts in `data/library/calibration/STUDY/`.

| Path | Contents |
| --- | --- |
| `surface/` | Unwrapping control measurements, per-piece band truth, and band surveys against published meshes. |
| `umbilici/` *(optional, not bundled in OSS release)* | Local centerline overrides (`chosen.json`) and community centerlines (`vesuvius-center-lines/`). |
