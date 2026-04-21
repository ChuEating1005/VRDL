from scipy.optimize import linear_sum_assignment

import torch
from torch import Tensor, nn

from src.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# NYCUHW2Detection keeps ann["category_id"] unchanged, so the matcher uses
# the dataset-emitted label convention as-is. For the provided digits COCO data,
# that means labels stay 1..10 and pred_logits should expose at least 11 classes.


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        assert cost_class > 0 and cost_bbox > 0 and cost_giou > 0
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        bs, num_queries = pred_logits.shape[:2]

        out_prob = pred_logits.flatten(0, 1).sigmoid()
        out_bbox = pred_boxes.flatten(0, 1)

        sizes = [len(t["boxes"]) for t in targets]
        tgt_ids = torch.cat([t["labels"] for t in targets], dim=0)
        tgt_bbox = torch.cat([t["boxes"] for t in targets], dim=0)

        # DINO uses sigmoid focal-style classification cost instead of softmax NLL.
        neg_cost_class = (1 - self.focal_alpha) * (out_prob**self.focal_gamma) * (
            -(1 - out_prob + 1e-8).log()
        )
        pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (
            -(out_prob + 1e-8).log()
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        cost_splits = C.split(sizes, dim=-1)

        indices = []
        for i, size in enumerate(sizes):
            if size == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            row_ind, col_ind = linear_sum_assignment(cost_splits[i][i])
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )

        return indices


def build_matcher(args) -> HungarianMatcher:
    return HungarianMatcher(
        cost_class=getattr(args, "set_cost_class", 2.0),
        cost_bbox=getattr(args, "set_cost_bbox", 5.0),
        cost_giou=getattr(args, "set_cost_giou", 2.0),
        focal_alpha=getattr(args, "focal_alpha", 0.25),
        focal_gamma=getattr(args, "focal_gamma", 2.0),
    )
