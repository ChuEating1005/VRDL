"""Fuse multiple COCO instance-segmentation result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.engine import _fuse_tta_candidates


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble COCO result JSONs with mask fusion")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input COCO result JSON files")
    parser.add_argument("--output", required=True, help="Output fused COCO result JSON")
    parser.add_argument("--fusion", choices=["nms", "mask"], default="mask")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def record_to_candidate(record: dict[str, Any]) -> dict[str, Any]:
    x, y, w, h = record["bbox"]
    return {
        "image_id": int(record["image_id"]),
        "category_id": int(record["category_id"]),
        "score": float(record["score"]),
        "box_xyxy": [float(x), float(y), float(x + w), float(y + h)],
        "segmentation": record["segmentation"],
    }


def load_candidates(paths: list[str], score_threshold: float) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            if float(record["score"]) < score_threshold:
                continue
            candidate = record_to_candidate(record)
            by_image[int(candidate["image_id"])].append(candidate)
    return by_image


def main() -> None:
    args = get_args()
    by_image = load_candidates(args.inputs, args.score_threshold)
    fused_records: list[dict[str, Any]] = []
    for image_id in tqdm(sorted(by_image), desc="ensemble"):
        fused_records.extend(
            _fuse_tta_candidates(
                by_image[image_id],
                fusion=args.fusion,
                iou_threshold=args.iou_threshold,
                mask_threshold=args.mask_threshold,
            )
        )
    fused_records.sort(key=lambda row: (int(row["image_id"]), -float(row["score"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(fused_records, f)
    print(f"Saved {len(fused_records)} fused records to {output}")


if __name__ == "__main__":
    main()
