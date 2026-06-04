from __future__ import annotations

import torch
import torch.nn.functional as F


def make_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")
    dtype = dtype or torch.float32
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur(x: torch.Tensor, kernel_size: int = 9, sigma: float = 1.5) -> torch.Tensor:
    channels = x.shape[1]
    kernel = make_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=x.device,
        dtype=x.dtype,
    )
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)


def downsample_upsample(x: torch.Tensor, scale: int = 2, mode: str = "bicubic") -> torch.Tensor:
    if scale <= 1:
        return x
    low = F.interpolate(x, scale_factor=1 / scale, mode=mode, align_corners=False)
    return F.interpolate(low, size=x.shape[-2:], mode=mode, align_corners=False)


def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.03, clamp: bool = True) -> torch.Tensor:
    noisy = x + torch.randn_like(x) * sigma
    return noisy.clamp(0, 1) if clamp else noisy


def random_degrade(
    clean: torch.Tensor,
    blur_kernel_size: int = 9,
    blur_sigma: float = 1.5,
    noise_sigma: float = 0.03,
    scale: int = 2,
) -> torch.Tensor:
    degraded = gaussian_blur(clean, kernel_size=blur_kernel_size, sigma=blur_sigma)
    degraded = downsample_upsample(degraded, scale=scale)
    degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
    return degraded
