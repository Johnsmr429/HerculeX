"""The organisers' 2 um ink model (scrollprize/ink_canonical_2um) on this Mac.

A 3D ResNet-152 with a 3D U-Net decoder that collapses depth by attention: it reads 62 layers one 2.4 um voxel
apart, 256 x 256 px tiles (0.6 mm), and answers at a quarter of the tile's resolution. The network and the reading
rule are the published ones (villa, ink-detection/optimized_inference, MIT: model_resnet3d_3d_decoder.py,
models/resnetall.py, inference.py), written out here so the module names match the checkpoint: values clipped to
0..200 and divided by 200, tiles blended by a Hann window, the sigmoid of the logits resized to the tile.

    FineReader(weights).read(stack, stride=128)   stack (62, height, width) uint8, the first layer nearest the reader
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LAYERS = 62
TILE = 256
CLIP = 200


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class Backbone(nn.Module):
    def __init__(self, blocks=(3, 8, 36, 3)):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(1, 64, (7, 7, 7), (1, 2, 2), (3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.layer1 = self.stage(64, blocks[0], 1)
        self.layer2 = self.stage(128, blocks[1], 2)
        self.layer3 = self.stage(256, blocks[2], 2)
        self.layer4 = self.stage(512, blocks[3], 2)

    def stage(self, planes, count, stride):
        down = None
        if stride != 1 or self.in_planes != planes * 4:
            down = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * 4, 1, stride, bias=False), nn.BatchNorm3d(planes * 4)
            )
        layers = [Bottleneck(self.in_planes, planes, stride, down)]
        self.in_planes = planes * 4
        layers += [Bottleneck(self.in_planes, planes) for _ in range(1, count)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        return [a, b, c, self.layer4(c)]


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        return F.relu(self.gn2(self.conv2(out)) + self.shortcut(x))


class DepthCollapse(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn_conv = nn.Conv3d(in_ch, 1, 1)

    def forward(self, x):
        return (x * F.softmax(self.attn_conv(x), dim=2)).sum(dim=2)


class Decoder(nn.Module):
    def __init__(self, encoder=(256, 512, 1024, 2048), decoder=(64, 128, 256, 512)):
        super().__init__()
        self.channel_reduce = nn.ModuleList(
            nn.Sequential(nn.Conv3d(e, d, 1, bias=False), nn.GroupNorm(min(32, d), d), nn.ReLU())
            for e, d in zip(encoder, decoder)
        )
        self.decoder_blocks = nn.ModuleList(ResBlock(decoder[i] + decoder[i - 1], decoder[i - 1]) for i in (3, 2, 1))
        self.depth_collapse = DepthCollapse(decoder[0])
        self.logit = nn.Conv2d(decoder[0], 1, 1)

    def forward(self, maps):
        feats = [reduce(m) for reduce, m in zip(self.channel_reduce, maps)]
        x = feats[3]
        for block, skip in zip(self.decoder_blocks, (feats[2], feats[1], feats[0])):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.logit(self.depth_collapse(x))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.decoder = Decoder()
        self.normalization = nn.BatchNorm3d(1)

    def forward(self, x):
        return self.decoder(self.backbone(self.normalization(x)))


def loaded(weights):
    state = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    clean = {}
    for key, value in state.items():
        while key.startswith(("model.", "module.")):
            key = key.split(".", 1)[1]
        clean[key] = value
    network = Network()
    wanted = set(network.state_dict())
    left = {k for k in clean if k not in wanted and not k.startswith(("backbone.fc.", "decoder.aux_head"))}
    missing = wanted - set(clean)
    if left or missing:
        raise ValueError(f"checkpoint does not fit: unexpected {sorted(left)[:5]}, missing {sorted(missing)[:5]}")
    network.load_state_dict({k: v for k, v in clean.items() if k in wanted})
    return network.eval()


def places(length, tile, stride):
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, stride))
    if out[-1] != length - tile:
        out.append(length - tile)
    return out


class FineReader:
    def __init__(self, weights, device="mps", half=True, batch=2):
        self.device, self.half, self.batch = device, half, batch
        self.network = loaded(weights).to(device)
        if half:
            self.network.half()

    @torch.no_grad()
    def tiles(self, block):
        """Ink probability (n, 64, 64) of (n, 62, 256, 256) uint8 tiles."""
        x = torch.from_numpy(block).to(self.device)
        x = (x.half() if self.half else x.float()).clamp_(0, CLIP).div_(CLIP)[:, None]
        return torch.sigmoid(self.network(x).float())[:, 0]

    def read(self, stack, stride=128, report=None):
        """Probability of ink of a (62, height, width) uint8 stack at its own resolution."""
        depth, height, width = stack.shape
        assert depth == LAYERS, depth
        pad_y, pad_x = max(TILE - height, 0), max(TILE - width, 0)
        if pad_y or pad_x:
            stack = np.pad(stack, ((0, 0), (0, pad_y), (0, pad_x)))
        _, h, w = stack.shape
        total = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        weight = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        hann = torch.from_numpy(np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)).to(self.device) + 1e-4
        spots = [(y, x) for y in places(h, TILE, stride) for x in places(w, TILE, stride)]
        spots = [(y, x) for y, x in spots if stack[depth // 2, y : y + TILE, x : x + TILE].any()]
        for start in range(0, len(spots), self.batch):
            group = spots[start : start + self.batch]
            block = np.stack([stack[:, y : y + TILE, x : x + TILE] for y, x in group])
            maps = F.interpolate(self.tiles(block)[:, None], size=(TILE, TILE), mode="bilinear", align_corners=False)[
                :, 0
            ]
            for (y, x), tile in zip(group, maps):
                total[y : y + TILE, x : x + TILE] += tile * hann
                weight[y : y + TILE, x : x + TILE] += hann
            if report and (start // self.batch) % 50 == 0:
                report(start, len(spots))
        out = (total / weight.clamp_min(1e-6)).cpu().numpy()
        return out[:height, :width]
