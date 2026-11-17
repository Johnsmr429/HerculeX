# scans/transfer

These modules download data over HTTP into brick stores and mirrors.

| File | Description |
| --- | --- |
| `parallel_fetch.py` | Runs many HTTP GET requests over kept-alive connections, with byte ranges and retries. |
| `fetch_volume.py` | Streams one OME-Zarr level of a scan into a brick store. |
| `fetch_atlas.py` | Fetches every scroll at about 75 µm per voxel, into one folder per scroll under `OUT`. |
| `fetch_fragment.py` | Fetches a fragment's surface volume and its infrared truth images. |
| `fetch_products.py` | Mirrors the published meshes and maps into the library's `published/open_data/`, packed losslessly. |
| `fetch_server.py` | Mirrors the products that only dl.ash2txt.org holds into the library's `published/dl_ash2txt/`. |
| `sharded_labels.py` | Reads a 2D zarr v3 mask stored in blosc-compressed shards. |

## Commands

```sh
./run.sh scans.transfer.fetch_atlas OUT [VOXEL_UM]
./run.sh scans.transfer.fetch_products walk|fetch|verify
./run.sh scans.transfer.fetch_server plan|fetch|verify
```

Each mirror runs in three steps. `walk` or `plan` lists what to fetch, `fetch` downloads it and resumes where it
stopped, and `verify` accounts for every planned file and compares a random sample with the source. Build the
inventories in [`scans/catalog`](../catalog/README.md) first.

`HERCULEX_FETCH_CONNECTIONS` sets the default number of connections that `parallel_fetch` opens (default `32`). Sheet
detection reads the same variable with a default of `16`.
