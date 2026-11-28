import contextlib
import json
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
repo_studio = os.path.normpath(os.path.join(here, "..", "studio"))
sys.path.insert(0, repo_studio)
sys.path.insert(0, here)
from model import DeepResidualUNet


def load_safetensors(path):
    file = open(path, "rb")
    mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
    header_size = struct.unpack_from("<Q", mapping, 0)[0]
    header = json.loads(mapping[8 : 8 + header_size])
    base = 8 + header_size
    types = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
    }
    state = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        first, last = spec["data_offsets"]
        count = int(np.prod(spec["shape"], dtype=np.int64))
        state[name] = torch.frombuffer(
            mapping, dtype=types[spec["dtype"]], count=count, offset=base + first
        ).reshape(spec["shape"])
    return state, mapping, file


def crop(volume, y, x):
    h, w = volume.shape[-2:]
    tile = volume[:, max(0, y) : min(h, y + 128), max(0, x) : min(w, x + 128)]
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    if any(a or b for a, b in pad):
        mode = "reflect" if min(h, w) > 1 else "edge"
        tile = np.pad(tile, pad, mode=mode)
    return tile


def autocast_for(device, dtype=None):
    if device.type == "cuda":
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.inference_mode()
def predict(model, volume, device, batch=8, dtype="worker"):
    volume = volume[:16]
    h, w = volume.shape[-2:]
    out = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    axis = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    kernel = (axis[:, None] * axis[None, :]).astype(np.float32)
    empty = (volume == 0).all(0)
    ys = sorted(set(range(0, h, 64)) | {max(h - 96, 0)})
    xs = sorted(set(range(0, w, 64)) | {max(w - 96, 0)})
    points = [(y, x) for y in ys for x in xs if not empty[y : y + 96, x : x + 96].all()]
    if dtype == "fp32":
        ctx = contextlib.nullcontext()
    elif dtype == "fp16":
        ctx = autocast_for(device, torch.float16)
    else:
        ctx = autocast_for(device)
    with ctx:
        for start in range(0, len(points), batch):
            group = points[start : start + batch]
            raw = np.stack([crop(volume, y - 16, x - 16) for y, x in group])
            tensor = torch.from_numpy(raw[:, None]).to(device).float().clamp_(0, 200).div_(255)
            logits = model(tensor)
            maps = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
            for (y, x), one in zip(group, maps):
                hh, ww = min(96, h - y), min(96, w - x)
                out[y : y + hh, x : x + ww] += one[:hh, :ww] * kernel[:hh, :ww]
                weight[y : y + hh, x : x + ww] += kernel[:hh, :ww]
    np.divide(out, np.maximum(weight, 1e-12), out=out, where=weight > 0)
    out[empty] = 0
    return out


def metrics(ref, test, label):
    diff = np.abs(ref - test)
    r_flat = ref.ravel().astype(np.float64)
    t_flat = test.ravel().astype(np.float64)
    r_c = r_flat - r_flat.mean()
    t_c = t_flat - t_flat.mean()
    corr = float(np.dot(r_c, t_c) / (np.linalg.norm(r_c) * np.linalg.norm(t_c) + 1e-12))
    print(
        f"[{label}] mae={diff.mean():.6f} p99={np.percentile(diff, 99):.6f} "
        f"max={diff.max():.6f} corr={corr:.6f}",
        flush=True,
    )
    for thr in (0.3, 0.5, 0.7):
        a = ref >= thr
        b = test >= thr
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        dice = (2.0 * inter) / max(int(a.sum() + b.sum()), 1)
        iou = inter / max(union, 1)
        print(
            f"  thr={thr:.1f}: ref_pos={int(a.sum())} test_pos={int(b.sum())} dice={dice:.5f} iou={iou:.5f}",
            flush=True,
        )


def main():
    weights = sys.argv[1]
    binary = sys.argv[2]
    vol_path = sys.argv[3] if len(sys.argv) > 3 else None
    device = torch.device("cuda")
    model = DeepResidualUNet()
    state, mapping, wf = load_safetensors(weights)
    model.load_state_dict(state, strict=True)
    del state
    mapping.close()
    wf.close()
    model = model.to(device).eval()

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(12345)
        tiles = rng.uniform(0.0, 200.0 / 255.0, size=(2, 1, 16, 128, 128)).astype(np.float32)
        tiles_in = os.path.join(tmp, "tiles_in.npy")
        np.save(tiles_in, tiles)
        with torch.inference_mode():
            t_gpu = torch.from_numpy(tiles).to(device)
            ref_fp32 = model(t_gpu).cpu().numpy()
            with autocast_for(device, torch.float16):
                ref_fp16 = model(t_gpu).float().cpu().numpy()
        metrics(ref_fp32, ref_fp16, "PyTorch FP32 vs PyTorch FP16 (tiles)")

        for mode in ("f222", "f444"):
            out_p = os.path.join(tmp, f"tiles_{mode}.npy")
            subprocess.run(
                [binary, "tiles", weights, tiles_in, out_p, "--batch", "2", "--mode", mode],
                check=True,
            )
            got = np.load(out_p)
            metrics(ref_fp32, got, f"PyTorch FP32 vs studio_cuda({mode}) (tiles)")
            metrics(ref_fp16, got, f"PyTorch FP16 vs studio_cuda({mode}) (tiles)")

        if vol_path and os.path.exists(vol_path):
            vol = np.load(vol_path)
            print(f"\nLoaded volume {vol_path}: shape={vol.shape} dtype={vol.dtype}", flush=True)
        else:
            vol = np.zeros((16, 512, 512), dtype=np.uint8)
            yy, xx = np.mgrid[0:512, 0:512]
            for z in range(16):
                pat = 90 + 40 * np.sin(xx / 9.0 + z * 0.2) * np.cos(yy / 11.0 - z * 0.15)
                pat += 35 * (((xx // 40 + yy // 40) % 2 == 0) & ((xx % 40 < 12) | (yy % 40 < 12)))
                vol[z] = np.clip(pat + rng.normal(0, 12, (512, 512)), 10, 220).astype(np.uint8)
            vol_path = os.path.join(tmp, "vol.npy")
            np.save(vol_path, vol)

        _ = predict(model, vol, device, batch=8, dtype="worker")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_worker = predict(model, vol, device, batch=8, dtype="worker")
        torch.cuda.synchronize()
        pt_worker_ms = (time.perf_counter() - t0) * 1e3

        _ = predict(model, vol, device, batch=8, dtype="fp16")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_fp16_map = predict(model, vol, device, batch=8, dtype="fp16")
        torch.cuda.synchronize()
        pt_fp16_ms = (time.perf_counter() - t0) * 1e3

        print(
            f"\nPyTorch worker.py ({vol.shape}): {pt_worker_ms:.2f} ms | "
            f"PyTorch FP16: {pt_fp16_ms:.2f} ms",
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

        for mode in ("f222", "f444"):
            out_p = os.path.join(tmp, f"map_{mode}.npy")
            res = subprocess.run(
                [
                    binary,
                    "run",
                    weights,
                    vol_path,
                    out_p,
                    "--batch",
                    "8",
                    "--mode",
                    mode,
                    "--profile",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(res.stdout.strip(), flush=True)
            got_map = np.load(out_p)
            metrics(ref_worker, got_map, f"Predict PyTorch worker.py vs studio_cuda({mode})")
            metrics(ref_fp16_map, got_map, f"Predict PyTorch FP16 vs studio_cuda({mode})")


if __name__ == "__main__":
    main()
