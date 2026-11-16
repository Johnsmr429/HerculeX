"""Refresh published viewer HTML without recomposing tiles or changing saved marks."""

import json
import re

from storage.data_roots import BROWSER_ASSETS, SHEETS


def refresh(root=None, titles=None, extra=None):
    """Every viewer rebuilt from its template; `titles` (page folder name to title) retitles those pages and `extra`
    (page folder name to meta fields, such as outlines) adds or replaces fields kept in their meta from then on."""
    updated = []
    for target in sorted((root or SHEETS).glob("*/index.html")):
        old = target.read_text()
        match = re.search(r"const M=(.*?),TOP=", old)
        if not match:
            continue
        meta = json.loads(match[1])
        meta["title"] = (titles or {}).get(target.parent.name, meta["title"])
        meta |= (extra or {}).get(target.parent.name, {})
        template = "two_unwraps.html" if meta["scan"].startswith("two-unwraps/") else "sheet_viewer.html"
        floor = re.search(r"floor=kept.floor\?\?([\d.]+)", old).group(1)
        scan = re.search(r"scan=kept.scan\?\?([\d.]+)", old).group(1)
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
