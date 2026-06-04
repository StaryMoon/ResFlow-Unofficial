from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        device = t.device
        dtype = t.dtype
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=device, dtype=dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """Compact conditional U-Net used as the velocity field estimator."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(image_channels * 2, base_channels, kernel_size=3, padding=1)
        self.down = ResBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.middle = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up = ResBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t.to(dtype=x_t.dtype))
        h0 = self.input(torch.cat([x_t, condition], dim=1))
        h1 = self.down(h0, time_emb)
        h2 = self.downsample(h1)
        h2 = self.middle(h2, time_emb)
        h3 = self.upsample(h2)
        if h3.shape[-2:] != h1.shape[-2:]:
            h3 = F.interpolate(h3, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up(torch.cat([h3, h1], dim=1), time_emb)
        return self.output(F.silu(self.output_norm(h)))
