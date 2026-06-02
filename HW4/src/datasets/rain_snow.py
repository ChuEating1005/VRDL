from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class RainSnowPairedDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        patch_size: int = 128,
        split: str = "train",
        indices: list[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.degraded_dir = self.root / "degraded"
        self.clean_dir = self.root / "clean"
        self.patch_size = patch_size
        self.split = split

        idxs = indices if indices is not None else list(range(1, 1601))
        self.samples: list[tuple[Path, Path, str]] = []
        for de_type in ("rain", "snow"):
            for i in idxs:
                deg = self.degraded_dir / f"{de_type}-{i}.png"
                clean = self.clean_dir / f"{de_type}_clean-{i}.png"
                self.samples.append((deg, clean, de_type))

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_and_augment(
        self, deg: torch.Tensor, clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, h, w = deg.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            deg = torch.nn.functional.pad(deg, (0, pad_w, 0, pad_h), mode="reflect")
            clean = torch.nn.functional.pad(clean, (0, pad_w, 0, pad_h), mode="reflect")
            _, h, w = deg.shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        deg = deg[:, top : top + ps, left : left + ps]
        clean = clean[:, top : top + ps, left : left + ps]
        if random.random() < 0.5:
            deg = torch.flip(deg, dims=[2])
            clean = torch.flip(clean, dims=[2])
        if random.random() < 0.5:
            deg = torch.flip(deg, dims=[1])
            clean = torch.flip(clean, dims=[1])
        k = random.randint(0, 3)
        if k:
            deg = torch.rot90(deg, k, dims=[1, 2])
            clean = torch.rot90(clean, k, dims=[1, 2])
        return deg, clean

    def __getitem__(self, idx: int) -> dict:
        deg_path, clean_path, de_type = self.samples[idx]
        deg = _to_tensor(Image.open(deg_path))
        clean = _to_tensor(Image.open(clean_path))
        if self.split == "train":
            deg, clean = self._crop_and_augment(deg, clean)
        return {
            "degraded": deg,
            "clean": clean,
            "de_type": de_type,
            "name": deg_path.name,
        }


class TestDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.files = sorted(
            self.root.glob("*.png"),
            key=lambda p: int(p.stem),
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        img = _to_tensor(Image.open(path))
        return {"degraded": img, "name": path.name}
