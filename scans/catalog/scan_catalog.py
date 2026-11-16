"""Index of every scan, volume and segment product in the open-data bucket's metadata."""

import gzip
import json
import sys
import urllib.request
from pathlib import Path

from storage.data_roots import SCAN_CATALOG

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
GRAND_PRIZE = ["0125", "0191", "0211", "0257", "0268", "0358", "0800", "0813", "0826", "1203", "1218", "1447", "1545"]
FIRST_LETTERS = [
    "0125",
    "0175A",
    "0175B",
    "0191",
    "0211",
    "0257",
    "0268",
    "0358",
    "0483A",
    "0483B",
    "0490A",
    "0490B",
    "0800",
    "0813",
    "0826",
    "0846A",
    "0846B",
    "1203",
    "1218",
    "1545",
]
NARROW_WINDOW = {"20241024131838"}


def fetch_metadata(path):
    raw = urllib.request.urlopen(f"{BUCKET}/metadata.min.json", timeout=120).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)


def load(path=SCAN_CATALOG / "metadata.json"):
    if not Path(path).exists():
        fetch_metadata(path)
    return json.loads(Path(path).read_text())["samples"]


def https_root(origin):
    """HTTPS root of one origin."""
    url = origin["access_roots"][0]["url"]
    if url.startswith("s3://"):
        return f"https://{url[5:]}.s3.us-east-1.amazonaws.com"
    return url.rstrip("/")


def volumes(samples):
    """One row per reconstructed volume."""
    rows = []
    for name, sample in sorted(samples.items()):
        for volume_id, volume in sample["volumes"].items():
            products = {d["type"]: d["origins"][0]["path"] for d in volume["data"]}
            roots = {d["type"]: https_root(d["origins"][0]) for d in volume["data"]}
            rows.append(
                {
                    "sample": name,
                    "type": sample["sample"]["properties"].get("type"),
                    "volume": volume_id,
                    "voxel_um": volume["properties"].get("pixel_size_um"),
                    "energy_kev": volume["properties"].get("energy_keV"),
                    "products": products,
                    "roots": roots,
                }
            )
    return rows


def survey_volumes(samples):
    """The volumes a survey may choose from."""
    return [row for row in volumes(samples) if row["volume"] not in NARROW_WINDOW]


def segments(samples):
    """One row per segment, its products by type."""
    rows = []
    for name, sample in sorted(samples.items()):
        for segment_id, segment in sample["segments"].items():
            products = {}
            for data in segment.get("data", []):
                products.setdefault(data["type"], []).append(data["origins"][0]["path"])
            rows.append(
                {"sample": name, "segment": segment_id, "suffix": segment.get("suffix", ""), "products": products}
            )
    return rows


if __name__ == "__main__":
    samples = load()
    for row in segments(samples):
        for path in row["products"].get("ink-labels", []):
            print(row["sample"], row["segment"], row["suffix"], path.split("/ink-labels/")[1])
    print(len(volumes(samples)), "volumes", len(segments(samples)), "segments", file=sys.stderr)
