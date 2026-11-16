# Scan catalog

This module builds local indexes of scans and published products that HerculeX can fetch.

## Files

| File | Purpose |
| --- | --- |
| `scan_catalog.py` | Indexes scans, volumes, and segment products from the open-data metadata. |
| `scan_choice.py` | Selects the survey scan for each scroll. |
| `product_inventory.py` | Records the size and completeness of products in the open-data bucket. |
| `server_inventory.py` | Indexes files available from `dl.ash2txt.org` but absent from the bucket metadata. |

## Commands

```sh
./run.sh scans.catalog.product_inventory
./run.sh scans.catalog.server_inventory
```

The commands write JSON and JSON Lines indexes under `${HERCULEX_LIBRARY:-data/library}/catalog/`.
