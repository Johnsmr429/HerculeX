"""Refresh published viewer HTML without recomposing tiles or changing saved marks."""

import json
import re

from storage.data_roots import BROWSER_ASSETS, SHEETS

META = re.compile(r"const M\s*=\s*(.*?),\s*TOP\s*=", re.DOTALL)
FLOOR = re.compile(r"floor\s*=\s*kept\.floor\s*\?\?\s*([\d.]+)")
SCAN = re.compile(r"scan\s*=\s*kept\.scan\s*\?\?\s*([\d.]+)")


def refresh(root=None, titles=None, extra=None):
    """Every viewer rebuilt from its template; `titles` (page folder name to title) retitles those pages and `extra`
    (page folder name to meta fields, such as outlines) adds or replaces fields kept in their meta from then on."""
    updated = []
    for target in sorted((root or SHEETS).glob("*/index.html")):
        old = target.read_text()
        match = META.search(old)
        if not match:
            continue
        meta = json.loads(match[1])
        meta["title"] = (titles or {}).get(target.parent.name, meta["title"])
        meta |= (extra or {}).get(target.parent.name, {})
        template = "sheet_viewer.html"
        floor, scan = FLOOR.search(old).group(1), SCAN.search(old).group(1)
        html = (
            (BROWSER_ASSETS / template)
            .read_text()
            .replace("__TITLE__", meta["title"])
            .replace("__META__", json.dumps(meta))
            .replace("__FLOOR__", floor)
            .replace("__SCAN__", scan)
        )
        if html != old:
            pending = target.with_suffix(".html.part")
            pending.write_text(html)
            pending.replace(target)
            updated.append(target)
    return updated


if __name__ == "__main__":
    for path in refresh():
        print(path)
