from __future__ import annotations

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0.0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10.0 * torch.log10(max_val**2 / mse)


def psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    pred = pred.clamp(0.0, max_val)
    target = target.clamp(0.0, max_val)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(max_val**2 / mse)
