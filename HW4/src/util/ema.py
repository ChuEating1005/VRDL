"""Exponential moving average of model parameters."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ps, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if ps.dtype.is_floating_point:
                ps.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                ps.copy_(p)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
