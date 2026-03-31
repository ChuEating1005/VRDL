"""Utility functions for HW1 Image Classification."""

import os
import random
import shutil

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks running mean and count for a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    predicted = output.argmax(dim=1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)


def save_checkpoint(state: dict, is_best: bool, output_dir: str):
    filepath = os.path.join(output_dir, "last.pth")
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(output_dir, "best.pth")
        shutil.copy(filepath, best_path)
        print(f"  ★ New best model saved (val_acc={state.get('best_val_acc', 0):.4f})")
