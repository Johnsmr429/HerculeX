"""Fetch a fragment's surface volume and its infrared truth."""

import re
import sys
import urllib.request
from pathlib import Path

from scans.transfer.parallel_fetch import fetch_all

SERVER = "https://dl.ash2txt.org/fragments"
TRUTH = ("inklabels.png", "ir.png", "mask.png")


def listing(url):
    page = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r'href="([^"?/][^"]*)"', page)


def fetch_fragment(frag, package, working, out):
    base = f"{SERVER}/{frag}/{package}.volpkg/working/{working}"
    out = Path(out)
    (out / "surface_volume").mkdir(parents=True, exist_ok=True)
    names = [n for n in listing(f"{base}/") if n in TRUTH]
    names += [f"surface_volume/{n}" for n in listing(f"{base}/surface_volume/") if n.endswith(".tif")]
    todo = [(n, n) for n in names if not (out / n).exists()]

    def deliver(name, body):
        if body is None:
            raise OSError(f"missing {name}")
        temporary = out / f"{name}.part"
        temporary.write_bytes(body)
        temporary.replace(out / name)
        print(name, len(body), flush=True)

    return fetch_all(base, todo, deliver, connections=8)


if __name__ == "__main__":
    print(fetch_fragment(*sys.argv[1:5]))
