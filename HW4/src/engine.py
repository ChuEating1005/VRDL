from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .datasets import RainSnowPairedDataset, TestDataset
from .models import build_model
from .util import EMA, build_loss, psnr_batch


@dataclass
class TrainCfg:
    data_root: str = "data/train"
    out_dir: str = "outputs/baseline"
    epochs: int = 120
    batch_size: int = 8
    patch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    val_fraction: float = 0.05
    num_workers: int = 8
    seed: int = 0
    log_every: int = 50
    val_every: int = 1
    amp: bool = True
    grad_clip: float = 0.0
    resume: str | None = None
    wandb_project: str | None = "vrdl-hw4"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    model_name: str = "promptir"
    model_kwargs: dict = field(default_factory=dict)
    loss: str = "l1"
    fft_weight: float = 0.0
    charb_eps: float = 1e-3
    ema_decay: float = 0.0
    extra: dict = field(default_factory=dict)


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_fraction)))
    val = sorted(i + 1 for i in perm[:n_val])
    train_ = sorted(i + 1 for i in perm[n_val:])
    return train_, val


def _build_loaders(cfg: TrainCfg) -> tuple[DataLoader, DataLoader]:
    train_idx, val_idx = _split_indices(1600, cfg.val_fraction, cfg.seed)
    train_ds = RainSnowPairedDataset(
        cfg.data_root, patch_size=cfg.patch_size, split="train", indices=train_idx
    )
    val_ds = RainSnowPairedDataset(
        cfg.data_root, patch_size=cfg.patch_size, split="val", indices=val_idx
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


def _lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    import math

    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        deg = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(deg)
        pred = pred.float()
        clean = clean.float()
        total += psnr_batch(pred, clean).sum().item()
        count += deg.size(0)
    model.train()
    return total / max(1, count)


def train(cfg: TrainCfg) -> dict:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    wandb_run = None
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or out_dir.name,
            dir=str(out_dir),
            config=cfg.__dict__,
            mode=cfg.wandb_mode,
            resume="allow",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model_name, **cfg.model_kwargs).to(device)
    loss_fn = build_loss(cfg.loss, fft_weight=cfg.fft_weight, charb_eps=cfg.charb_eps)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    train_loader, val_loader = _build_loaders(cfg)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    start_epoch = 0
    best_psnr = -1.0
    global_step = 0
    if cfg.resume and Path(cfg.resume).exists():
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        best_psnr = ckpt.get("best_psnr", -1.0)
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        print(f"resumed from {cfg.resume} at epoch {start_epoch}, best PSNR {best_psnr:.3f}")

    history: list[dict] = []
    lr = cfg.lr
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        loss_accum = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            deg = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            lr = _lr_at(global_step, total_steps, warmup_steps, cfg.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                pred = model(deg)
                loss = loss_fn(pred, clean)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            if ema is not None:
                ema.update(model)
            loss_accum += loss.item()
            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": loss.item(), "train/lr": lr},
                        step=global_step,
                    )
            global_step += 1
        avg_loss = loss_accum / max(1, len(train_loader))
        elapsed = time.time() - t0

        record: dict = {"epoch": epoch, "loss": avg_loss, "time_sec": elapsed, "lr": lr}
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            val_psnr = evaluate(model, val_loader, device)
            record["val_psnr"] = val_psnr
            writer.add_scalar("val/psnr", val_psnr, epoch)
            if ema is not None:
                val_psnr_ema = evaluate(ema.shadow, val_loader, device)
                record["val_psnr_ema"] = val_psnr_ema
                writer.add_scalar("val/psnr_ema", val_psnr_ema, epoch)
            score = max(val_psnr, record.get("val_psnr_ema", val_psnr))
            if score > best_psnr:
                best_psnr = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "optim": optim.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_psnr": best_psnr,
                        "cfg": cfg.__dict__,
                    },
                    out_dir / "best.pt",
                )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": avg_loss,
                    "train/epoch_time_sec": elapsed,
                    **({"val/psnr": record["val_psnr"]} if "val_psnr" in record else {}),
                    **({"val/psnr_ema": record["val_psnr_ema"]} if "val_psnr_ema" in record else {}),
                    "val/best_psnr": best_psnr,
                },
                step=global_step,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "optim": optim.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_psnr": best_psnr,
                "cfg": cfg.__dict__,
            },
            out_dir / "last.pt",
        )
        history.append(record)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"[epoch {epoch}] loss={avg_loss:.4f} val_psnr={record.get('val_psnr', float('nan')):.3f} "
            f"best={best_psnr:.3f} lr={lr:.2e} time={elapsed:.1f}s"
        )
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return {"best_psnr": best_psnr, "history": history}


def _d4_transforms() -> list:
    pairs: list = []
    for k in range(4):
        for flip in (False, True):
            def make(k_=k, flip_=flip):
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    y = torch.rot90(x, k_, dims=[2, 3])
                    if flip_:
                        y = torch.flip(y, dims=[3])
                    return y

                def inv(y: torch.Tensor) -> torch.Tensor:
                    if flip_:
                        y = torch.flip(y, dims=[3])
                    return torch.rot90(y, -k_, dims=[2, 3])

                return fwd, inv

            pairs.append(make())
    return pairs


def _build_model_from_cfg(cfg: dict) -> nn.Module:
    name = cfg.get("model_name", "promptir")
    kwargs = cfg.get("model_kwargs") or cfg.get("extra") or {}
    return build_model(name, **kwargs)


@torch.no_grad()
def predict(
    ckpt_path: str,
    test_dir: str,
    out_path: str,
    device: str | None = None,
    tta: int = 0,
    use_ema: bool = True,
) -> str:
    import numpy as np

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    model = _build_model_from_cfg(saved_cfg).to(dev)
    state = ckpt.get("ema") if (use_ema and "ema" in ckpt and ckpt["ema"] is not None) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {'EMA' if state is ckpt.get('ema') else 'model'} weights from {ckpt_path}")

    transforms = _d4_transforms()
    if tta == 0:
        transforms = transforms[:1]
    elif tta == 4:
        transforms = [transforms[0], transforms[1], transforms[4], transforms[5]]

    ds = TestDataset(test_dir)
    out: dict[str, "np.ndarray"] = {}
    for i in tqdm(range(len(ds)), desc="predict"):
        sample = ds[i]
        x = sample["degraded"].unsqueeze(0).to(dev)
        preds = []
        for fwd, inv in transforms:
            with torch.amp.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                y = model(fwd(x))
            preds.append(inv(y.float()))
        pred = torch.stack(preds, dim=0).mean(dim=0).clamp(0.0, 1.0)[0].cpu().numpy()
        arr = (pred * 255.0 + 0.5).astype("uint8")
        out[sample["name"]] = arr
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"wrote {len(out)} predictions to {out_path}")
    return out_path


def average_checkpoints(ckpt_paths: list[str], out_path: str) -> str:
    assert ckpt_paths, "no checkpoints given"
    state_sum: dict | None = None
    ema_sum: dict | None = None
    n = 0
    for p in ckpt_paths:
        c = torch.load(p, map_location="cpu", weights_only=False)
        m = c["model"]
        if state_sum is None:
            state_sum = {k: v.clone().float() for k, v in m.items()}
        else:
            for k in state_sum:
                state_sum[k] += m[k].float()
        if "ema" in c and c["ema"] is not None:
            e = c["ema"]
            if ema_sum is None:
                ema_sum = {k: v.clone().float() for k, v in e.items()}
            else:
                for k in ema_sum:
                    ema_sum[k] += e[k].float()
        n += 1
        last_cfg = c.get("cfg", {})
    assert state_sum is not None
    avg = {k: (v / n).to(dtype=state_sum[k].dtype) for k, v in state_sum.items()}
    ema_avg = None
    if ema_sum is not None:
        ema_avg = {k: (v / n).to(dtype=ema_sum[k].dtype) for k, v in ema_sum.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": avg, "ema": ema_avg, "cfg": last_cfg, "n_averaged": n}, out_path)
    print(f"averaged {n} ckpts → {out_path}")
    return out_path
