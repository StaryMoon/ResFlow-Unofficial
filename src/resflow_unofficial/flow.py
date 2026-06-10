from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .unet import SimpleUNet


class RestorationFlow(nn.Module):
    """Compact restoration-flow module.

    The path starts at the degraded image and moves toward the clean image.
    A conditional network predicts the velocity field along that path.
    """

    def __init__(self, image_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.velocity = SimpleUNet(
            image_channels=image_channels,
            base_channels=base_channels,
        )

    @staticmethod
    def interpolate(clean: torch.Tensor, degraded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.ndim < clean.ndim:
            t = t[..., None]
        return (1 - t) * degraded + t * clean

    def forward(self, x_t: torch.Tensor, degraded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity(x_t=x_t, condition=degraded, t=t)

    def training_loss(self, clean: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        if clean.shape != degraded.shape:
            raise ValueError("clean and degraded tensors must have the same shape.")
        batch = clean.shape[0]
        t = torch.rand(batch, device=clean.device, dtype=clean.dtype)
        x_t = self.interpolate(clean=clean, degraded=degraded, t=t)
        target_velocity = clean - degraded
        predicted_velocity = self.forward(x_t=x_t, degraded=degraded, t=t)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, degraded: torch.Tensor, steps: int = 4, clamp: bool = True) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        x = degraded.clone()
        batch = degraded.shape[0]
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full(
                (batch,),
                fill_value=index / steps,
                device=degraded.device,
                dtype=degraded.dtype,
            )
            velocity = self.forward(x_t=x, degraded=degraded, t=t)
            x = x + dt * velocity
            if clamp:
                x = x.clamp(0, 1)
        return x
