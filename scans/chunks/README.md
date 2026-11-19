# scans/chunks

The chunk store reads one level of a remote OME-Zarr array, usually a scan's CT at level 0, and keeps every chunk it
fetches. The ml module and the bench read the CT through it. The level fit reads brick stores from
[`scans/transfer`](../transfer/README.md) instead.

| File | Description |
| --- | --- |
| `volume.py` | Reads a box or the chunks around points of one array level, fetching missing chunks first. |
| `chunk_store.py` | Keeps the decompressed chunks of one array level in an append-only pack file and its SQLite index. |
| `array_registry.py` | Records the remote arrays, the resolver's answers and small documents in `arrays.sqlite`. |
| `resolve_scan.py` | Finds where a scan's CT lives: the open-data bucket first, then the access roots in the scan catalog. |
| `bucket.py` | Lists and reads the open-data bucket. |
| `fetch.py` | Fetches a box or the chunks around points, and reports or evicts what the store holds. |
| `fetch_urls.py`, `fetch_urls.c` | Fetch many URLs at once over libcurl. |
| `read_box.py`, `read_box.c`, `chunks_near_points.c` | Copy a box out of the pack file, and list the chunks near a set of points. |

## Commands

```sh
./run.sh scans.chunks.fetch SCROLL SCAN ct --box z0:z1,y0:y1,x0:x1     # fetch a box of the CT at level 0
./run.sh scans.chunks.fetch resolve SCROLL SCAN [--refresh]           # where the scan's CT lives
./run.sh scans.chunks.fetch usage                                     # what the store holds, per array
./run.sh scans.chunks.fetch evict SCROLL/NAME[/LEVEL]                 # remove an array from the store
```

Fetches use the network link, so run long ones as board jobs with `--link`.

## Outputs

The store is `cache/chunks/` in the outputs folder: `SCROLL/NAME/LEVEL/` per array level (`chunks.pack` and
`index.sqlite`) and `arrays.sqlite`. `HERCULEX_CHUNKS` moves it. A fetch stops when less than its minimum of free disk
space is left.
