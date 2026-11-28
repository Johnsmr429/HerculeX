import argparse
import concurrent.futures
import ctypes
import ctypes.util
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import numpy.lib.format as npy_fmt

try:
    import blosc as _blosc_mod
except ImportError:
    _blosc_mod = None
    _candidates = [
        ctypes.util.find_library("blosc"),
        "/opt/homebrew/lib/libblosc.dylib",
        "/usr/local/lib/libblosc.dylib",
        "/usr/lib/x86_64-linux-gnu/libblosc.so.1",
    ]
    _found = [p for p in _candidates if p and (os.path.exists(p) or not p.startswith("/"))]
    if not _found:
        sys.exit("fetch: install the blosc Python package or the blosc library")
    _libblosc = ctypes.CDLL(_found[0])
    _libblosc.blosc_decompress_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libblosc.blosc_decompress_ctx.restype = ctypes.c_int


def decompress(raw, nbytes):
    if _blosc_mod is not None:
        return np.frombuffer(_blosc_mod.decompress(raw), dtype=np.uint8)
    out = np.empty(nbytes, dtype=np.uint8)
    rc = _libblosc.blosc_decompress_ctx(raw, out.ctypes.data, nbytes, 1)
    if rc != nbytes:
        raise RuntimeError(f"blosc returned {rc} bytes, expected {nbytes}")
    return out


def zarr_layout(url):
    with urllib.request.urlopen(url.rstrip("/") + "/.zarray", timeout=60) as r:
        meta = json.load(r)
    if meta.get("dtype") not in ("|u1", "<u1", ">u1"):
        sys.exit(f"fetch: {url} holds {meta.get('dtype')}, not uint8")
    compressor = (meta.get("compressor") or {}).get("id")
    if compressor != "blosc":
        sys.exit(f"fetch: {url} is compressed with {compressor}, not blosc")
    return tuple(int(c) for c in meta["chunks"]), int(meta.get("fill_value") or 0)


def drop_cache(fd, offset, length):
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def fetch_box(url, box, out_path, workers=32, mask=None):
    (z0, z1), (y0, y1), (x0, x1) = box
    (cz, cy, cx), fill = zarr_layout(url)
    Z, Y, X = z1 - z0, y1 - y0, x1 - x0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    f = open(out_path, "wb")
    npy_fmt.write_array_header_1_0(f, {"descr": "|u1", "fortran_order": False, "shape": (Z, Y, X)})
    data_off = f.tell()
    if Z * Y * X > 0:
        f.seek(data_off + Z * Y * X - 1)
        f.write(b"\x00")
        f.flush()
    gz_list = list(range(z0 // cz, (z1 - 1) // cz + 1)) if Z > 0 else []
    gy_list = list(range(y0 // cy, (y1 - 1) // cy + 1)) if Y > 0 else []
    gx_list = list(range(x0 // cx, (x1 - 1) // cx + 1)) if X > 0 else []
    parsed = urllib.parse.urlsplit(url.rstrip("/"))
    use_http = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    base_path = parsed.path.rstrip("/")
    local = threading.local()
    open_conns = []
    conn_lock = threading.Lock()

    def thread_conn():
        conn = getattr(local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            conn = cls(parsed.netloc, timeout=60)
            local.conn = conn
            with conn_lock:
                open_conns.append(conn)
        return conn

    def reset_conn():
        conn = getattr(local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
            local.conn = None

    def read_chunk(gz, gy, gx):
        if not use_http:
            chunk_url = f"{url.rstrip('/')}/{gz}/{gy}/{gx}"
            with urllib.request.urlopen(chunk_url, timeout=60) as r:
                return r.read()
        path = f"{base_path}/{gz}/{gy}/{gx}"
        conn = thread_conn()
        conn.request("GET", path, headers={"Connection": "keep-alive"})
        resp = conn.getresponse()
        body = resp.read()
        if resp.status == 200:
            return body
        if resp.status in (403, 404):
            return None
        if resp.status in (301, 302, 303, 307, 308):
            location = resp.getheader("Location")
            if location:
                with urllib.request.urlopen(location, timeout=60) as r:
                    return r.read()
        raise RuntimeError(f"HTTP {resp.status}")

    def get_chunk(key):
        gz, gy, gx = key
        for attempt in range(6):
            try:
                raw = read_chunk(gz, gy, gx)
                if raw is None:
                    return key, None
                return key, decompress(raw, cz * cy * cx).reshape(cz, cy, cx)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    return key, None
            except (urllib.error.URLError, http.client.HTTPException, OSError, RuntimeError):
                reset_conn()
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"fetch: could not read {url.rstrip('/')}/{gz}/{gy}/{gx} after 6 tries")

    mask_shape = mask.shape if mask is not None else None
    yi = np.minimum(np.arange(y0, y1) // 16, mask_shape[1] - 1) if mask is not None else None
    xi = np.minimum(np.arange(x0, x1) // 16, mask_shape[2] - 1) if mask is not None else None

    def chunk_needed(sz0, sz1, sy0, sy1, sx0, sx1):
        if mask is None:
            return True
        mz0 = min(sz0 // 16, mask_shape[0] - 1)
        mz1 = min((sz1 - 1) // 16, mask_shape[0] - 1)
        my0 = min(sy0 // 16, mask_shape[1] - 1)
        my1 = min((sy1 - 1) // 16, mask_shape[1] - 1)
        mx0 = min(sx0 // 16, mask_shape[2] - 1)
        mx1 = min((sx1 - 1) // 16, mask_shape[2] - 1)
        return bool(mask[mz0 : mz1 + 1, my0 : my1 + 1, mx0 : mx1 + 1].any())

    t0 = time.time()
    hits = skipped = kept_vox = total_vox = 0
    plane = Y * X
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for gz in gz_list:
                sz0, sz1 = max(z0, gz * cz), min(z1, (gz + 1) * cz)
                slab_keys = []
                for gy in gy_list:
                    sy0, sy1 = max(y0, gy * cy), min(y1, (gy + 1) * cy)
                    for gx in gx_list:
                        sx0, sx1 = max(x0, gx * cx), min(x1, (gx + 1) * cx)
                        if chunk_needed(sz0, sz1, sy0, sy1, sx0, sx1):
                            slab_keys.append((gz, gy, gx))
                        else:
                            skipped += 1
                slab = np.full((sz1 - sz0, Y, X), 0 if mask is not None else fill, dtype=np.uint8)
                if slab_keys:
                    futures = [ex.submit(get_chunk, key) for key in slab_keys]
                    for fut in concurrent.futures.as_completed(futures):
                        (_, gy, gx), arr = fut.result()
                        if arr is None:
                            continue
                        hits += 1
                        sy0, sy1 = max(y0, gy * cy), min(y1, (gy + 1) * cy)
                        sx0, sx1 = max(x0, gx * cx), min(x1, (gx + 1) * cx)
                        slab[:, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = arr[
                            sz0 - gz * cz : sz1 - gz * cz,
                            sy0 - gy * cy : sy1 - gy * cy,
                            sx0 - gx * cx : sx1 - gx * cx,
                        ]
                if mask is not None:
                    for iz in range(sz1 - sz0):
                        inside = mask[min((sz0 + iz) // 16, mask_shape[0] - 1)][np.ix_(yi, xi)]
                        total_vox += int(np.count_nonzero(slab[iz]))
                        slab[iz] = np.where(inside, slab[iz], 0)
                        kept_vox += int(np.count_nonzero(slab[iz]))
                f.seek(data_off + (sz0 - z0) * plane)
                f.write(memoryview(slab))
                f.flush()
                drop_cache(f.fileno(), data_off + (sz0 - z0) * plane, (sz1 - sz0) * plane)
    finally:
        f.close()
        for conn in open_conns:
            try:
                conn.close()
            except OSError:
                pass
    total = len(gz_list) * len(gy_list) * len(gx_list)
    print(
        f"fetched {Z}x{Y}x{X} ({hits} of {total} chunks, {skipped} masked out) in {time.time() - t0:.1f}s -> {out_path}",
        flush=True,
    )
    return kept_vox, total_vox


def fetch_unit(work, name, out_path, workers=32):
    work = Path(work)
    plan = {u["name"]: u for u in json.loads((work / "units_plan.json").read_text())}
    u = plan[name]
    z0, z1, y0, y1, x0, x1 = u["z0"], u["z1"], u["y0"], u["y1"], u["x0"], u["x1"]
    url = json.loads((work / "scan.json").read_text())["m7_url"]
    m = np.load(work / "papyrus_mask_L4.npz")
    shape = tuple(m["shape"])
    mask = np.unpackbits(m["mask"], axis=-1)[..., : shape[2]].astype(bool)
    t = time.time()
    kept, total = fetch_box(url, ((z0, z1), (y0, y1), (x0, x1)), out_path, workers=workers, mask=mask)
    print(f"masked {name}: kept {kept} of {total} m7 voxels ({time.time() - t:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Fetch a box of a published m7 surface prediction from S3 into a uint8 .npy, masked to the papyrus for a unit."
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    unit = sub.add_parser("unit", help="one planned unit, masked to the papyrus")
    unit.add_argument("name")
    unit.add_argument("out")
    unit.add_argument("--work", required=True, help="folder with units_plan.json, scan.json, papyrus_mask_L4.npz")
    unit.add_argument("--workers", type=int, default=32)
    box = sub.add_parser("box", help="any box of a zarr level, unmasked")
    box.add_argument("url")
    box.add_argument("box", help="z0:z1,y0:y1,x0:x1")
    box.add_argument("out")
    box.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    if args.mode == "unit":
        fetch_unit(args.work, args.name, args.out, args.workers)
    else:
        ranges = tuple(tuple(int(v) for v in part.split(":")) for part in args.box.split(","))
        fetch_box(args.url, ranges, args.out, args.workers)


if __name__ == "__main__":
    main()
