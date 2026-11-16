# scans/catalog

Indexes of what can be fetched, kept in the library's `catalog/`.

```text
scan_catalog.py        scans, volumes and segment products from the bucket's metadata.json
scan_choice.py         the scan each scroll is surveyed on, in prize order
product_inventory.py   bytes of every published product from the bucket listing
server_inventory.py    files only dl.ash2txt.org holds

./run.sh scans.catalog.product_inventory
./run.sh scans.catalog.server_inventory
```
