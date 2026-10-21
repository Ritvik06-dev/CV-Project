"""Time-conditioned UNet (ADM-style, simplified).

The official code uses the OpenAI/LDM UNetModel with concat conditioning:
the source image y is concatenated to x_t on the channel axis (in_channels = 6).
This is a smaller re-implementation of the same structure: residual blocks with
scale-shift timestep modulation, self-attention at chosen resolutions, and
strided-conv / nearest-upsample resampling.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def norm(ch):
    groups = min(32, ch)
    while ch % groups:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout):
        super().__init__()
        self.in_layers = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_ch))
        self.out_norm = norm(out_ch)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class Attention(nn.Module):
    def __init__(self, ch, head_ch=64):
        super().__init__()
        self.heads = max(1, ch // head_ch)
        self.norm = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, image_size=64, in_channels=3, out_channels=3, cond_channels=3,
                 model_channels=64, channel_mult=(1, 2, 2, 4), num_res_blocks=2,
                 attention_resolutions=(16,), dropout=0.0):
        super().__init__()
        self.cond_channels = cond_channels
        emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.model_channels = model_channels

        self.inp = nn.Conv2d(in_channels + cond_channels, model_channels, 3, padding=1)
        self.down = nn.ModuleList()
        chans, ch, res = [model_channels], model_channels, image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, model_channels * mult, emb_dim, dropout)]
                ch = model_channels * mult
                if res in attention_resolutions:
                    layers.append(Attention(ch))
                self.down.append(nn.ModuleList(layers))
                chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), Attention(ch), ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + chans.pop(), model_channels * mult, emb_dim, dropout)]
                ch = model_channels * mult
                if res in attention_resolutions:
                    layers.append(Attention(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    layers.append(nn.Conv2d(ch, ch, 3, padding=1))
                    res *= 2
                self.up.append(nn.ModuleList(layers))

        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(layers, h, emb):
        for layer in layers:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x, t, cond=None):
        if self.cond_channels:
            x = torch.cat([x, cond], dim=1)
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        h = self.inp(x)
        skips = [h]
        for layers in self.down:
            h = self._run(layers, h, emb)
            skips.append(h)
        h = self._run(self.mid, h, emb)
        for layers in self.up:
            h = self._run(layers, torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)
