# ResFlow-Unofficial

> Unofficial PyTorch implementation starter for **Reversing Flow for Image Restoration** (CVPR 2025).
>
> If this repo saves you reading / reproduction time, please give it a star and follow [@StaryMoon](https://github.com/StaryMoon). It helps me keep building open reproduction starters for recent restoration papers.

## Status

This repository is an **independent, unofficial, work-in-progress reproduction starter**.

- Paper: [Reversing Flow for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Reversing_Flow_for_Image_Restoration_CVPR_2025_paper.html)
- Venue: CVPR 2025, pp. 7545-7558
- Official code: the CVPR page says the authors plan to release code; this repository is not affiliated with the authors.
- Reproduction status: **benchmarks are not reproduced yet**.

The first release focuses on making the core idea easy to inspect:

- conditional velocity network for image restoration
- degradation operators for toy experiments
- flow-matching style training objective
- few-step Euler sampling loop from degraded image to restored image
- smoke-test script for quick sanity checks

## What This Is

ResFlow models image restoration as a deterministic path from low-quality images to high-quality images and learns a velocity field along that path. This repo implements a compact starter version of that idea:

```text
degraded image x_0  -- learned velocity field -->  restored image x_1
```

The implementation is intentionally small so students and researchers can read it quickly, modify it, and plug in real datasets later.

## What This Is Not

This is not the official codebase and not a drop-in reproduction of the paper's reported numbers.

Current limitations:

- exact architecture from the paper is not implemented yet
- auxiliary-variable details are simplified in v0.1.0
- no official checkpoints
- no full dataset pipeline
- no PSNR / SSIM benchmark table yet

## Quick Start

```bash
git clone https://github.com/StaryMoon/ResFlow-Unofficial.git
cd ResFlow-Unofficial
pip install -r requirements.txt
python scripts/smoke_test.py
```

Expected output:

```text
loss: ...
restored: torch.Size([2, 3, 64, 64])
```

## Minimal Usage

```python
import torch

from resflow_unofficial import RestorationFlow, random_degrade

clean = torch.rand(2, 3, 64, 64)
degraded = random_degrade(clean)

model = RestorationFlow(image_channels=3, base_channels=32)
loss = model.training_loss(clean, degraded)
loss.backward()

restored = model.sample(degraded, steps=4)
```

## Roadmap

- [ ] Match the exact ResFlow architecture more closely.
- [ ] Add auxiliary-state modeling instead of the simplified conditional path.
- [ ] Add dataset loaders for deraining, deblurring, denoising, and low-light enhancement.
- [ ] Add PSNR / SSIM evaluation scripts.
- [ ] Reproduce a small benchmark table.
- [ ] Add pretrained checkpoints when experiments become stable.

## Why I Built This

Image restoration papers often become useful only after someone turns the math into a runnable repo. This project is meant to be that first public scaffold: clear enough to read, small enough to fork, and honest about what is still missing.

## Citation

If you use the method, please cite the original paper:

```bibtex
@InProceedings{Qin_2025_CVPR,
  author = {Qin, Haina and Luo, Wenyang and Wang, Libin and Zheng, Dandan and Chen, Jingdong and Yang, Ming and Li, Bing and Hu, Weiming},
  title = {Reversing Flow for Image Restoration},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {7545--7558}
}
```

## License

This repository is released under the MIT License. The paper and any official materials remain owned by their respective authors / publishers.
