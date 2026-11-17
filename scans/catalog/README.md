# scans/catalog

These modules index what you can fetch, and keep the indexes in the library's `catalog/` folder.

| File | Description |
| --- | --- |
| `scan_catalog.py` | Indexes the scans, volumes and segment products listed in the bucket's metadata, which it downloads to `metadata.json` on first use. |
| `scan_choice.py` | Chooses the scan that each scroll is surveyed on, and orders the scrolls by prize. |
| `product_inventory.py` | Records the size of every published product from the bucket listing, in `products.jsonl`. |
| `server_inventory.py` | Lists the files that only dl.ash2txt.org holds, in `server.jsonl`. |

```sh
./run.sh scans.catalog.product_inventory   # rebuild products.jsonl and print totals by product kind
./run.sh scans.catalog.server_inventory    # rebuild server.jsonl and print totals by folder
```

The mirrors in [`scans/transfer`](../transfer/README.md) plan their downloads from these two inventories.
