"""Small reads from the open-data bucket: S3 listings and JSON documents."""

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from scans.catalog.scan_catalog import BUCKET

USER_AGENT = "herculex-chunks/1"
KEY = re.compile(r"<Contents>.*?<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>.*?</Contents>", re.DOTALL)
FOLDER = re.compile(r"<CommonPrefixes><Prefix>([^<]*)</Prefix></CommonPrefixes>")
TOKEN = re.compile(r"<NextContinuationToken>([^<]*)</NextContinuationToken>")


def get(url, retries=4, timeout=60):
    """Bytes of one object; FileNotFoundError on 404, PermissionError on 403 (the listable bucket answers 404)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(url) from error
            if error.code == 403:
                raise PermissionError(url) from error
            if attempt == retries:
                raise
        except OSError:
            if attempt == retries:
                raise
        time.sleep(min(30, 2**attempt))
    raise OSError(url)


def get_json(url):
    return json.loads(get(url))


def list_prefix(prefix, delimiter="/", root=BUCKET, max_pages=64):
    """Keys with sizes and sub-folders directly under a prefix (S3 ListObjectsV2)."""
    keys, folders, token = [], [], None
    for _ in range(max_pages):
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        page = get(f"{root}/?{urllib.parse.urlencode(query)}").decode()
        if "</ListBucketResult>" not in page:
            raise OSError(f"invalid S3 listing for {prefix}")
        keys += [(html.unescape(k), int(s)) for k, s in KEY.findall(page)]
        folders += [html.unescape(f) for f in FOLDER.findall(page)]
        found = TOKEN.search(page)
        if not found:
            return keys, folders
        token = html.unescape(found.group(1))
    raise OSError(f"S3 listing for {prefix} longer than {max_pages} pages")
