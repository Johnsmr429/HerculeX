import contextlib
import json
import mmap
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepResidualUNet


def exact(stream, size):
    chunks = []
    left = size
    while left:
        part = stream.read(left)
        if not part:
            return None
        chunks.append(part)
        left -= len(part)
    return b"".join(chunks)


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
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
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
        if last < first:
            raise ValueError(f"invalid tensor offsets for {name}")
    return state, mapping, file


def device_of():
    requested = os.environ.get("HERCULEX_STUDIO_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crop(volume, y, x):
    h, w = volume.shape[-2:]
    tile = volume[:, max(0, y) : min(h, y + 128), max(0, x) : min(w, x + 128)]
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    if any(a or b for a, b in pad):
        mode = "reflect" if min(h, w) > 1 else "edge"
        tile = np.pad(tile, pad, mode=mode)
    return tile


class Folded(torch.nn.Module):

    def __init__(self, conv):
        super().__init__()
        out, channels = conv.weight.shape[:2]
        weight = conv.weight.detach().permute(0, 2, 1, 3, 4).reshape(out, 3 * channels, 3, 3)
        self.register_buffer("weight", weight.contiguous())

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1))
        stacked = torch.cat([padded[:, :, k : k + depth] for k in range(3)], dim=1)
        stacked = stacked.permute(0, 2, 1, 3, 4).reshape(batch * depth, 3 * channels, height, width)
        flat = torch.nn.functional.conv2d(stacked, self.weight.to(stacked.dtype), padding=1)
        return flat.reshape(batch, depth, -1, height, width).permute(0, 2, 1, 3, 4)


def folded(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if (
                isinstance(child, torch.nn.Conv3d)
                and child.kernel_size == (3, 3, 3)
                and child.stride == (1, 1, 1)
                and child.groups == 1
            ):
                setattr(module, name, Folded(child))
    return model


def autocast_for(device):
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    if device.type == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def predict(model, volume, device):
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
    batch = int(os.environ.get("HERCULEX_STUDIO_BATCH", "1" if device.type == "mps" else "8"))
    with autocast_for(device):
        for start in range(0, len(points), batch):
            group = points[start : start + batch]
            raw = np.stack([crop(volume, y - 16, x - 16) for y, x in group])
            tensor = torch.from_numpy(raw[:, None]).to(device, non_blocking=True).float().clamp_(0, 200).div_(255)
            logits = model(tensor)
            maps = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
            for (y, x), one in zip(group, maps):
                hh, ww = min(96, h - y), min(96, w - x)
                out[y : y + hh, x : x + ww] += one[:hh, :ww] * kernel[:hh, :ww]
                weight[y : y + hh, x : x + ww] += kernel[:hh, :ww]
    np.divide(out, np.maximum(weight, 1e-12), out=out, where=weight > 0)
    out[empty] = 0
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


device = device_of()
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
model = DeepResidualUNet()
state, mapping, weight_file = load_safetensors(sys.argv[1])
model.load_state_dict(state, strict=True)
del state
mapping.close()
weight_file.close()
if device.type == "mps":
    model = folded(model)
model = model.to(device).eval()
source = sys.stdin.buffer
target = sys.stdout.buffer
target.write(b"HXOK")
target.flush()
while header := exact(source, 12):
    depth, height, width = struct.unpack("=III", header)
    raw = exact(source, depth * height * width)
    if raw is None:
        break
    volume = np.frombuffer(raw, np.uint8).reshape(depth, height, width)
    result = predict(model, volume, device)
    target.write(result.astype("<f4", copy=False).tobytes())
    target.flush()
