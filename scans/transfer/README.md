# scans/transfer

These modules download data over HTTP into brick stores and mirrors.

| File | Description |
| --- | --- |
| `fetch_volume.c` | Streams one OME-Zarr level of a scan into a brick store. |
| `fetch_atlas.c` | Fetches every scroll at about 75 µm per voxel, into one folder per scroll under `OUT`. |
| `fetch_fragment.c` | Fetches a fragment's surface volume and its infrared truth images. |
| `mirror.c` | Mirrors published products into `published/open_data/` and `published/dl_ash2txt/`, packed losslessly. |

## Commands

```sh
./run.sh scans.transfer.fetch_volume PATH LEVEL OUT [--root URL] [--rows FIRST LAST] [--connections N] [--list]
./run.sh scans.transfer.fetch_atlas OUT [VOXEL_UM]
./run.sh scans.transfer.fetch_fragment FRAGMENT PACKAGE WORKING OUT
./run.sh scans.transfer.fetch_products walk|fetch|status|verify [--root URL]
./run.sh scans.transfer.fetch_server plan|fetch|status|verify [--root URL]
```

Each mirror runs in three steps. `walk` or `plan` lists what to fetch, `fetch` downloads it and resumes where it
stopped, and `verify` accounts for every planned file and compares a random sample with the source. Build the
inventories in [`scans/catalog`](../catalog/README.md) first. `status` reads the stored plan and receipts without contacting the source.

Native volume fetches use libcurl. Use `--connections N` to change the default of `32`. Sheet detection uses
`HERCULEX_FETCH_CONNECTIONS` and defaults to `16`.
