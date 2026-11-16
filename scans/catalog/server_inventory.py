"""Files on dl.ash2txt.org, which the bucket's metadata does not index."""

import json
import re
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from scans.catalog.product_inventory import page_of
from storage.data_roots import SCAN_CATALOG

SERVER = "https://dl.ash2txt.org"
PATH = SCAN_CATALOG / "server.jsonl"
START = ("/full-scrolls/", "/fragments/", "/datasets/", "/community-uploads/", "/other/", "/ml-models/")
ROW = re.compile(r'<td class="link"><a href="([^"]+)"[^>]*>.*?</a></td><td class="size">([^<]*)</td>')
UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}
NOT_WALKED = {
    "volumes",
    "volumes_masked",
    "volumes_small",
    "volumes_zarr",
    "volumes_zarr_standardized",
    "volume_grids",
    "normal_grids",
    "thaumato_outputs",
    "layers",
    "surface_volume",
    "representations",
}
STORE_MARKERS = {".zgroup", ".zarray", "zarr.json"}
MAX_DEPTH = 9


def is_store(folder):
    name = folder.rstrip("/").rsplit("/", 1)[1]
    return name in NOT_WALKED or name.endswith(".zarr")


def size_of(text):
    parts = text.split()
    return int(float(parts[0]) * UNITS[parts[1]]) if len(parts) == 2 and parts[1] in UNITS else 0


def entries(path):
    """Files and folders of one index page."""
    try:
        page = page_of(SERVER + path)
    except OSError as error:
        return [], [], str(error)[:80]
    rows = [(urllib.parse.unquote(h), s) for h, s in ROW.findall(page) if not h.startswith(("/", "..", "?", "http"))]
    return (
        [(path + h, size_of(s)) for h, s in rows if not h.endswith("/")],
        [path + h for h, _ in rows if h.endswith("/")],
        None,
    )


def build():
    files, stores, failed, frontier = [], [], [], list(START)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for depth in range(MAX_DEPTH):
            pages = list(pool.map(lambda p: entries(urllib.parse.quote(p)), frontier))
            found = []
            for path, (below, folders, error) in zip(frontier, pages):
                if error:
                    failed.append(path)
                if any(f.rsplit("/", 1)[1] in STORE_MARKERS for f, _ in below):
                    stores.append(path)
                    continue
                files += [(urllib.parse.unquote(f), s) for f, s in below]
                for folder in map(urllib.parse.unquote, folders):
                    (stores if is_store(folder) else found).append(folder)
            print("depth", depth, "pages", len(frontier), "files", len(files), "stores", len(stores), flush=True)
            frontier = found
            PATH.write_text("".join(json.dumps({"path": p, "bytes": s}) + "\n" for p, s in files))
            PATH.with_suffix(".stores.json").write_text(
                json.dumps({"stores": stores, "failed": failed, "unwalked": frontier}, indent=0)
            )
            if not frontier:
                break
    return files, stores, failed, frontier


if __name__ == "__main__":
    files, stores, failed, unwalked = build()
    total = Counter()
    for path, size in files:
        total["/".join(path.split("/")[1:4])] += size
    for key, size in total.most_common(40):
        print(f"{size / 1e9:12.2f} GB  {key}")
    print(
        len(files),
        "files",
        round(sum(s for _, s in files) / 1e12, 2),
        "TB",
        len(stores),
        "stores",
        len(failed),
        "failed",
        len(unwalked),
        "unwalked",
    )
