# Scans

The scan stage discovers available CT scans and published products, then downloads only the data required by later stages.

## Modules

| Path | Purpose |
| --- | --- |
| `catalog/` | This module indexes open-data scans, volumes, segments, and published products. |
| `transfer/` | This module downloads OME-Zarr levels, published mirrors, and fragment data. |

Catalogs and source data live under `${HERCULEX_LIBRARY:-data/library}/catalog/` and `raw/`. Temporary CT brick stores live under `cache/bricks/` and are removed after use unless the reading stage retains them.

Run long downloads through the job board with the `--link` resource flag.
