from __future__ import annotations

import argparse
import sys

import yaml

from .engine import TrainCfg, average_checkpoints, predict, train


def _load_cfg(path: str | None, overrides: dict) -> TrainCfg:
    data: dict = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return TrainCfg(**data)


def _cmd_train(args: argparse.Namespace) -> None:
    overrides = {
        "data_root": args.data_root,
        "out_dir": args.out_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
    }
    if args.no_wandb:
        overrides["wandb_project"] = None
    cfg = _load_cfg(args.config, overrides)
    train(cfg)


def _cmd_predict(args: argparse.Namespace) -> None:
    predict(
        ckpt_path=args.ckpt,
        test_dir=args.test_dir,
        out_path=args.out,
        tta=args.tta,
        use_ema=not args.no_ema,
    )


def _cmd_ensemble(args: argparse.Namespace) -> None:
    average_checkpoints(args.ckpts, args.out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("vrdl-hw4")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--config", type=str, default=None)
    pt.add_argument("--data-root", type=str, default=None)
    pt.add_argument("--out-dir", type=str, default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--patch-size", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--num-workers", type=int, default=None)
    pt.add_argument("--resume", type=str, default=None)
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--wandb-project", type=str, default=None)
    pt.add_argument("--wandb-run-name", type=str, default=None)
    pt.add_argument("--wandb-mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    pt.add_argument("--no-wandb", action="store_true")
    pt.set_defaults(func=_cmd_train)

    pp = sub.add_parser("predict")
    pp.add_argument("--ckpt", type=str, required=True)
    pp.add_argument("--test-dir", type=str, default="data/test/degraded")
    pp.add_argument("--out", type=str, default="pred.npz")
    pp.add_argument("--tta", type=int, default=0, choices=[0, 4, 8])
    pp.add_argument("--no-ema", action="store_true")
    pp.set_defaults(func=_cmd_predict)

    pe = sub.add_parser("ensemble")
    pe.add_argument("--ckpts", nargs="+", required=True)
    pe.add_argument("--out", type=str, required=True)
    pe.set_defaults(func=_cmd_ensemble)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
