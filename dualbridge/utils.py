import copy
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image

from .unet import UNet


def load_config(path, overrides=()):
    """YAML config with optional dotted overrides, e.g. train.batch_size=4."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_unet(cfg):
    u = dict(cfg["unet"])
    u["image_size"] = cfg["data"]["image_size"]
    u["cond_channels"] = 3 if cfg["model"].get("condition", True) else 0
    for k in ("channel_mult", "attention_resolutions"):
        u[k] = tuple(u[k])
    return UNet(**u)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def load_network(cfg, ckpt_path, device, use_ema=True):
    net = build_unet(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state["ema"] if use_ema and "ema" in state else state["model"])
    return net.eval()


def to_uint8(x):
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()


def save_images(x, folder, names):
    os.makedirs(folder, exist_ok=True)
    for img, name in zip(to_uint8(x), names):
        Image.fromarray(img).save(os.path.join(folder, name))


def save_grid(rows, path, pad=2):
    """rows: list of (N, 3, H, W) tensors; each becomes one row of the grid."""
    arrs = [to_uint8(r) for r in rows]
    n, h, w = arrs[0].shape[0], arrs[0].shape[1], arrs[0].shape[2]
    grid = np.full((len(arrs) * (h + pad) + pad, n * (w + pad) + pad, 3), 255, np.uint8)
    for r, a in enumerate(arrs):
        for c in range(n):
            y0, x0 = pad + r * (h + pad), pad + c * (w + pad)
            grid[y0:y0 + h, x0:x0 + w] = a[c]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(grid).save(path)
