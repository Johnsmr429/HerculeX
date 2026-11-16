"""Bytes of every published product, from the bucket's own listing."""

import html
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from scans.catalog.scan_catalog import BUCKET, https_root, load
from storage.data_roots import SCAN_CATALOG

PATH = SCAN_CATALOG / "products.jsonl"
MAX_PAGES = 3
KEY = re.compile(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", re.DOTALL)


def page_of(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode()
        except OSError:
            time.sleep(2**attempt)
    return urllib.request.urlopen(url, timeout=120).read().decode()


def listing(root, prefix, delimiter=None, pages=MAX_PAGES):
    """Keys and sizes under a prefix, and the folders one level down when a delimiter is given."""
    keys, folders, token = [], [], None
    for _ in range(pages):
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        page = page_of(f"{root}/?{urllib.parse.urlencode(query)}")
        if "<ListBucketResult " not in page or "</ListBucketResult>" not in page:
            raise OSError(f"invalid S3 listing for {prefix}")
        keys += [(html.unescape(k), int(s)) for k, s in KEY.findall(page)]
        folders += [html.unescape(f) for f in re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", page)]
        found = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", page)
        if not found:
            return keys, folders, "<IsTruncated>false</IsTruncated>" in page
        token = found.group(1)
    return keys, folders, False


def products(samples):
    """Every published product as (sample, owner, kind, root, path)."""
    rows = []
    for name, sample in sorted(samples.items()):
        for group in ("volumes", "segments"):
            for owner, item in sample[group].items():
                for data in item.get("data", []):
                    origin = data["origins"][0]
                    rows.append((name, owner, data["type"], https_root(origin), origin["path"]))
    return rows


def describe(row):
    sample, owner, kind, root, path = row
    record = {"sample": sample, "owner": owner, "kind": kind, "root": root, "path": path}
    if root != BUCKET:
        return {**record, "complete": False}
    try:
        keys, _, complete = listing(root, path)
    except OSError as error:
        return {**record, "error": str(error)[:120]}
    return {**record, "keys": len(keys), "bytes": sum(s for _, s in keys), "complete": complete}


def build():
    rows = products(load())
    with ThreadPoolExecutor(max_workers=16) as pool:
        described = list(pool.map(describe, rows))
    PATH.write_text("".join(json.dumps(d) + "\n" for d in described))
    return described


if __name__ == "__main__":
    described = build()
    bytes_by, count_by, open_by = Counter(), Counter(), Counter()
    for row in described:
        count_by[row["kind"]] += 1
        bytes_by[row["kind"]] += row.get("bytes", 0)
        open_by[row["kind"]] += "error" in row or not row.get("complete", False)
    for kind, count in count_by.most_common():
        print(f"{kind:28s} {count:5d} {bytes_by[kind] / 1e9:10.2f} GB  unsized {open_by[kind]}")
