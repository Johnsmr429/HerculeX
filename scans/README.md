# scans

Stage 1: listing the available scans and downloading them.

```text
catalog/    indexes of the open-data bucket and dl.ash2txt.org; the scan chosen for each scroll
transfer/   range fetches of OME-Zarr levels into brick stores; published mirrors; fragments
```

Writes in the library: `catalog/`, `raw/SCROLL/atlas_VOXEL/`, `cache/bricks/SCROLL/STORE/` (removed once read), `published/open_data/`, `published/dl_ash2txt/`.
Downloads share one network link; long fetches go on the job board with `--link`.
