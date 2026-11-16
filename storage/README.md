# Storage

This module defines every data root and the binary formats used by the pipeline.

## Files

| File | Purpose |
| --- | --- |
| `data_roots.py` | Defines the library, outputs, runtime, and stage paths. |
| `brick_store.py` | Stores a CT level or surface volume as raw bricks with an offset table. |
| `node_store.py` | Compresses sheet nodes and cells without changing their values. |
| `route_database.py` | Rebuilds a SQLite index of completed stage units from their records. |
| `tile_store.py` | Packs a sheet image pyramid and its byte-range index. |

## Data roots

The library defaults to `data/library/` and contains catalogs, source data, model weights, controls, studies, and append-only hand marks. Set `HERCULEX_LIBRARY` to move it.

The outputs default to `data/outputs/` and contain unwrapping artifacts, sampled stacks, ink maps, and the generated website. Set `HERCULEX_OUTPUTS` to move it. Set `HERCULEX_DATA` to move both default folders together.

Use the path helpers in `data_roots.py` instead of embedding absolute paths. `HERCULEX_BRICKS` can place the temporary CT brick cache on a separate volume.
