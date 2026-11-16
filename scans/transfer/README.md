# Scan transfer

This module downloads scan volumes and published products over HTTP.

## Files

| File | Purpose |
| --- | --- |
| `parallel_fetch.py` | Performs concurrent HTTP requests with connection reuse and retries. |
| `fetch_volume.py` | Streams one OME-Zarr level into a local brick store. |
| `fetch_atlas.py` | Downloads a coarse CT level for each scroll. |
| `fetch_fragment.py` | Downloads a fragment's surface volume and infrared truth. |
| `fetch_products.py` | Mirrors published open-data meshes and maps. |
| `fetch_server.py` | Mirrors products available only from `dl.ash2txt.org`. |
| `sharded_labels.py` | Reads nonempty chunks from a sharded Zarr v3 label mask. |

## Commands

```sh
./run.sh scans.transfer.fetch_atlas OUT [TARGET_UM]
./run.sh scans.transfer.fetch_products walk|fetch|verify
./run.sh scans.transfer.fetch_server plan|fetch|verify
```

Published mirrors are stored under `${HERCULEX_LIBRARY:-data/library}/published/`. `HERCULEX_FETCH_CONNECTIONS` sets the HTTP connection count and defaults to `32`.
