"""Run direct multi-checkpoint TTA inference and fuse raw candidates once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.hw3_cells import HW3TestDataset, collate_fn
from src.engine import _fuse_tta_candidates, _predict_candidates_with_tta
from src.main import load_checkpoint_if_needed, make_model
from src.util.misc import get_device, save_json, set_seed


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 5-fold checkpoint ensemble + TTA inference")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--submission", default="outputs/test-results-ensemble-tta.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-samples", type=int, default=0)

    parser.add_argument("--backbone", choices=["resnet50", "resnet101", "convnext_tiny"], default="resnet50")
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--use-pafpn", action="store_true")
    parser.add_argument("--mask-loss", choices=["bce", "dice"], default="bce")
    parser.add_argument("--small-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detections-per-img", type=int, default=1000)
    parser.add_argument("--box-score-thresh", type=float, default=0.05)
    parser.add_argument("--model-min-size", type=int, default=512)
    parser.add_argument("--model-max-size", type=int, default=1024)
    parser.add_argument("--rpn-pre-nms-top-n-train", type=int, default=1000)
    parser.add_argument("--rpn-post-nms-top-n-train", type=int, default=500)
    parser.add_argument("--rpn-pre-nms-top-n-test", type=int, default=1000)
    parser.add_argument("--rpn-post-nms-top-n-test", type=int, default=500)
    parser.add_argument("--box-batch-size-per-image", type=int, default=256)

    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-vhflip", action="store_true")
    parser.add_argument("--tta-scales", default="1.0")
    parser.add_argument("--tta-fusion", choices=["nms", "mask"], default="mask")
    parser.add_argument("--tta-iou-threshold", type=float, default=0.5)
    parser.add_argument("--tta-fusion-mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    for checkpoint in args.checkpoints:
        args.checkpoint = checkpoint
        model = make_model(args, device)
        load_checkpoint_if_needed(model, checkpoint, device)
        model.eval()
        models.append(model)
    return models


@torch.inference_mode()
def predict_ensemble(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = get_device(args.device)
    dataset = HW3TestDataset(args.data_root)
    if args.debug_samples > 0:
        dataset.infos = dataset.infos[: args.debug_samples]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    models = load_models(args, device)
    tta_scales = tuple(float(x) for x in args.tta_scales.split(",") if x)
    records: list[dict[str, Any]] = []
    for images, infos in tqdm(loader, desc="ensemble infer"):
        image = images[0].to(device)
        info = infos[0]
        candidates: list[dict[str, Any]] = []
        for model in models:
            candidates.extend(
                _predict_candidates_with_tta(
                    model=model,
                    image=image,
                    image_id=int(info.image_id),
                    mask_threshold=args.mask_threshold,
                    score_threshold=args.score_threshold,
                    tta_hflip=args.tta_hflip,
                    tta_vhflip=args.tta_vhflip,
                    tta_scales=tta_scales,
                    clear_cuda_cache=device.type == "cuda",
                )
            )
        records.extend(
            _fuse_tta_candidates(
                candidates,
                fusion=args.tta_fusion,
                iou_threshold=args.tta_iou_threshold,
                mask_threshold=args.tta_fusion_mask_threshold,
            )
        )
        del image, candidates
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    records = predict_ensemble(args)
    save_json(records, args.submission)
    print(f"Saved {len(records)} records to {args.submission}")


if __name__ == "__main__":
    main()
