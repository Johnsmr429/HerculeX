"""Pages of pictures shown place by place, each place's pictures switched in place, in the site's one look."""

import html
import json
import re
import sys
from pathlib import Path

from storage.data_roots import BROWSER_ASSETS, SHEETS

PICTURE = re.compile(r"(\d+)-(.+)\.(?:png|jpg)")


def places_of(folder):
    """Each subfolder of a page as a place, its pictures in their numbered order, titled from their file names."""
    places = []
    for sub in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        found = sorted(
            (int(m[1]), m[2].replace("-", " "), f"{sub.name}/{f.name}")
            for f in sub.iterdir()
            if (m := PICTURE.fullmatch(f.name))
        )
        if found:
            places.append({"name": sub.name, "pictures": [{"title": t, "src": s} for _, t, s in found]})
    return places


def write(folder, title, back="../index.html", places=None):
    """A page of places from its pictures on disc unless they are given."""
    meta = {"title": title, "back": back, "places": places_of(folder) if places is None else places}
    page = (BROWSER_ASSETS / "panel_page.html").read_text()
    (Path(folder) / "index.html").write_text(
        page.replace("__TITLE__", html.escape(title)).replace("__META__", json.dumps(meta).replace("</", "<\\/"))
    )


def title_of(folder):
    """The title a page already in the folder carries, else the folder's name."""
    path = Path(folder) / "index.html"
    found = re.search(r"<title>(.*?)</title>", path.read_text()) if path.exists() else None
    return html.unescape(found[1]) if found else Path(folder).name


if __name__ == "__main__":
    for name in sys.argv[1:]:
        write(SHEETS / name, title_of(SHEETS / name))
        print("page", SHEETS / name / "index.html", flush=True)
