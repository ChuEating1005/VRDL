"""Losses for image restoration."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier L1 (Lap pyramid loss). Smooth approximation of L1 around 0,
    well-behaved gradients for IR. eps=1e-3 matches Restormer/MPRNet."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps2).mean()


class FFTLoss(nn.Module):
    """L1 distance between real+imag FFT spectra. Targets periodic structure
    (rain streaks, snowflake textures) directly in the frequency domain.
    FFTformer / Restormer-FFT use this exact form."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.rfft2(pred.float(), norm="ortho")
        ft = torch.fft.rfft2(target.float(), norm="ortho")
        return (fp.real - ft.real).abs().mean() + (fp.imag - ft.imag).abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self, kind: str = "l1", fft_weight: float = 0.0, charb_eps: float = 1e-3):
        super().__init__()
        if kind == "l1":
            self.spatial = nn.L1Loss()
        elif kind == "charbonnier":
            self.spatial = CharbonnierLoss(eps=charb_eps)
        else:
            raise ValueError(f"unknown spatial loss: {kind}")
        self.fft_weight = fft_weight
        self.fft = FFTLoss() if fft_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.spatial(pred, target)
        if self.fft is not None:
            loss = loss + self.fft_weight * self.fft(pred, target)
        return loss


def build_loss(kind: str = "l1", fft_weight: float = 0.0, charb_eps: float = 1e-3) -> nn.Module:
    return CombinedLoss(kind=kind, fft_weight=fft_weight, charb_eps=charb_eps)
