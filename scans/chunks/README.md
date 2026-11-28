# scans/chunks

The chunk store reads one level of a remote OME-Zarr array and saves each fetched chunk. Synthetic
realism and slab sampling use this cache. The level fit reads brick stores from
[`scans/transfer`](../transfer/README.md).

| File | Description |
| --- | --- |
| `cached_ct.c` | Fetches and reads CT chunks, resolves sources, and manages the cache. |
| `read_box.c` | Copy a box out of the pack file, and list the chunks near a set of points. |

## Commands

```sh
./run.sh scans.chunks.fetch SCROLL SCAN ct --box z0:z1,y0:y1,x0:x1     # fetch a box of the CT at level 0
./run.sh scans.chunks.fetch resolve SCROLL SCAN [--refresh]           # list the scan's resolved arrays
./run.sh scans.chunks.fetch usage                                     # what the store holds, per array
./run.sh scans.chunks.fetch evict SCROLL/NAME[/LEVEL]                 # remove an array from the store
```

Use `--level L` for another pyramid level or `--points FILE.npy [--margin M]` in place of `--box`.
`--connections N` sets the HTTP concurrency for that fetch.

The CT source is the scan's `ome-zarr` product in the library's `catalog/metadata.json`.
Run `./run.sh catalog` to download the catalogue on a fresh clone. Resolution keeps the stored source in
`arrays.sqlite` in sync with that catalogue.

Fetches use the network link, so run long ones as board jobs with `--link`.

## Outputs

The store is `cache/chunks/` in the outputs folder: `SCROLL/NAME/LEVEL/` per array level (`chunks.pack` and
`index.sqlite`) and `arrays.sqlite`. `HERCULEX_CHUNKS` moves it. A fetch stops when less than its minimum of free disk
space is left.
