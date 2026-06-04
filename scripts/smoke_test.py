from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from resflow_unofficial import RestorationFlow, random_degrade


def main() -> None:
    torch.manual_seed(2026)
    clean = torch.rand(2, 3, 64, 64)
    degraded = random_degrade(clean)

    model = RestorationFlow(image_channels=3, base_channels=16)
    loss = model.training_loss(clean, degraded)
    loss.backward()
    restored = model.sample(degraded, steps=2)

    print(f"loss: {loss.item():.6f}")
    print(f"restored: {restored.shape}")


if __name__ == "__main__":
    main()
