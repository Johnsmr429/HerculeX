# scans/transfer

Downloads over HTTP into brick stores and mirrors.

```text
parallel_fetch.py   many GETs over kept-alive connections, ranges and retries
fetch_volume.py     one OME-Zarr level into a brick store
fetch_atlas.py      every scroll at about 75 µm, into raw/SCROLL/atlas_VOXEL/
fetch_fragment.py   a fragment's surface volume and infrared truth
fetch_products.py   published meshes and maps, packed losslessly, into published/open_data/
fetch_server.py     products only dl.ash2txt.org holds, into published/dl_ash2txt/
sharded_labels.py   a 2-D zarr v3 mask from blosc shards

./run.sh scans.transfer.fetch_atlas OUT [VOXEL_UM]
./run.sh scans.transfer.fetch_products walk|fetch|verify
./run.sh scans.transfer.fetch_server plan|fetch|verify
```
