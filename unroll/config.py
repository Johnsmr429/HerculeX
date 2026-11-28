import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HX = next(p for p in Path(__file__).resolve().parents if (p / "run.sh").exists() and (p / "level" / "level").is_dir())
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"


def data_root(name):
    return Path(subprocess.check_output([str(HX / "run.sh"), "paths", name], cwd=HX, text=True).strip())


def m7_url(volume):
    paths = [
        origin["path"]
        for entry in volume.get("data", [])
        if entry.get("type") == "surface-prediction-zarr"
        for origin in entry.get("origins", [])
        if "-m7-" in origin["path"]
    ]
    return BUCKET + sorted(paths)[-1].rstrip("/") + "/0" if paths else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scroll")
    ap.add_argument("scan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m7", default="")
    args = ap.parse_args()
    library = data_root("library")
    outputs = data_root("outputs")
    catalog = json.loads((library / "catalog" / "metadata.json").read_text())
    sample = catalog["samples"].get(args.scroll) or {}
    volume = (sample.get("volumes") or {}).get(args.scan)
    if volume is None:
        sys.exit(
            f"{args.scroll} {args.scan} is not in the scan catalog: run ./run.sh catalog {args.scroll}"
        )
    voxel = float(volume["properties"]["pixel_size_um"])
    if not 7.5 <= voxel <= 10.0:
        sys.exit(
            f"{args.scroll} {args.scan} is a {voxel} um scan; the unroll's settings are in voxels of 8.6 to 9.4 um: "
            f"use the scroll's 8.64 or 9.362 um scan"
        )
    level = outputs / "unwrapping" / "level" / args.scroll / args.scan
    curl_path = level / "curl.json"
    axis_path = level / "sheet" / "axis.json"
    curl = json.loads(curl_path.read_text()) if curl_path.exists() else {}
    if curl.get("rule") != "set by hand" or abs(int(curl.get("curl", 0))) != 1:
        sys.exit(
            f"{args.scroll} {args.scan} has no hand-set curl: check the scroll's winding by eye and run "
            f"./run.sh curl {args.scroll} {args.scan} 1 (or -1)"
        )
    if not axis_path.exists():
        sys.exit(
            f"{args.scroll} {args.scan} has no umbilicus in {axis_path}: "
            f"run ./run.sh level.level.umbilicus {args.scroll} {args.scan}"
        )
    url = args.m7 or m7_url(volume)
    if not url:
        sys.exit(
            f"{args.scroll} {args.scan} has no published m7 prediction: pass --m7 URL of a surface zarr level"
        )
    shape = json.load(urllib.request.urlopen(url.rstrip("/") + "/.zarray"))["shape"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(axis_path, out / "axis.json")
    source = level / "sheet" / "axis-source.json"
    if source.exists():
        shutil.copyfile(source, out / "axis-source.json")
    settings = {
        "scroll": args.scroll,
        "scan": args.scan,
        "voxel_um": voxel,
        "curl": int(curl["curl"]),
        "m7_url": url,
        "shape": ",".join(str(int(v)) for v in shape),
    }
    (out / "scan.json").write_text(json.dumps(settings, indent=1) + "\n")
    print(json.dumps(settings))


if __name__ == "__main__":
    main()
