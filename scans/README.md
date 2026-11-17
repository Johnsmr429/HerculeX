# scans

The `scans` package lists the available scans and downloads them (stage 1).

| Folder | Contents |
| --- | --- |
| `catalog/` | Indexes the Vesuvius Challenge open-data bucket and dl.ash2txt.org, and chooses the scan to survey for each scroll. |
| `transfer/` | Fetches OME-Zarr levels by byte range into brick stores, mirrors published products, and fetches fragments. |

These tools write to the following folders in the library:

| Folder | Contents |
| --- | --- |
| `catalog/` | The scan catalog and the product inventories. |
| `cache/bricks/SCROLL/STORE/` | CT bricks fetched for a read; each store is removed once it's read. `HERCULEX_BRICKS` moves this cache. |
| `published/open_data/` | A mirror of the products published in the open-data bucket. |
| `published/dl_ash2txt/` | A mirror of the products that only dl.ash2txt.org holds. |

All downloads share one network link. Run long fetches as board jobs with `--link`.
