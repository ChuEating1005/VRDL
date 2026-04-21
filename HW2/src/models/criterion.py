# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportUntypedFunctionDecorator=false, reportUnknownLambdaType=false, reportMissingParameterType=false
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dn_components import compute_dn_loss
from .matcher import build_matcher
from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..util.misc import accuracy, get_world_size, is_dist_avail_and_initialized


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    # Standard RetinaNet/DETR sigmoid focal formulation; caller multiplies by num_queries for DINO scaling.
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        losses: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.losses = losses if losses is not None else ["labels", "boxes", "cardinality"]

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(src, batch_idx) for batch_idx, (src, _) in enumerate(indices)],
            dim=0,
        )
        src_idx = torch.cat([src for (src, _) in indices], dim=0)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(tgt, batch_idx) for batch_idx, (_, tgt) in enumerate(indices)],
            dim=0,
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices], dim=0)
        return batch_idx, tgt_idx

    def get_loss(
        self,
        loss: str,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
        **kwargs,
    ) -> dict[str, Tensor]:
        loss_map: dict[str, Callable[..., dict[str, Tensor]]] = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "cardinality": self.loss_cardinality,
        }
        if loss not in loss_map:
            raise ValueError(f"Unknown loss: {loss}")
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
        log: bool = True,
    ) -> dict[str, Tensor]:
        src_logits = outputs["pred_logits"]
        # Dataset/matcher keep COCO category_id as-is: foreground labels occupy indices 1..10, index 0 stays unused.
        target_classes_onehot = torch.zeros_like(src_logits)

        matched_target_classes = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            tgt_classes = targets[batch_idx]["labels"][tgt_idx]
            target_classes_onehot[batch_idx, src_idx, tgt_classes] = 1.0
            matched_target_classes.append(tgt_classes)

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes=num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ) * src_logits.shape[1]

        losses = {"loss_ce": loss_ce}
        if log:
            if matched_target_classes:
                idx = self._get_src_permutation_idx(indices)
                target_classes_o = torch.cat(matched_target_classes, dim=0)
                losses["class_error"] = 100.0 - accuracy(src_logits[idx], target_classes_o)[0]
            else:
                losses["class_error"] = src_logits.new_zeros(())
        return losses

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        del indices, num_boxes
        pred_logits = outputs["pred_logits"]
        card_pred = (pred_logits.sigmoid().max(-1)[0] > 0.5).sum(1)
        tgt_lengths = torch.as_tensor([len(t["labels"]) for t in targets], device=pred_logits.device)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        loss_giou = (
            1.0
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(tgt_boxes),
                )
            )
        ).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        outputs_without_aux = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes_tensor = torch.as_tensor(
            [num_boxes],
            dtype=outputs_without_aux["pred_logits"].dtype,
            device=outputs_without_aux["pred_logits"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_tensor)
        num_boxes = torch.clamp(num_boxes_tensor / get_world_size(), min=1).item()

        losses: dict[str, Tensor] = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        for layer_idx, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux_outputs, targets)
            for loss in self.losses:
                kwargs = {"log": False} if loss == "labels" else {}
                aux_loss_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, **kwargs)
                losses.update({f"{key}_{layer_idx}": value for key, value in aux_loss_dict.items()})

        enc_outputs = outputs.get("enc_outputs")
        if enc_outputs is not None:
            enc_indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                if loss == "cardinality":
                    continue
                kwargs = {"log": False} if loss == "labels" else {}
                enc_loss_dict = self.get_loss(loss, enc_outputs, targets, enc_indices, num_boxes, **kwargs)
                losses.update({f"{key}_enc": value for key, value in enc_loss_dict.items()})

        # DN supervision comes from dn_components.compute_dn_loss; keep it unweighted here.
        losses.update(
            compute_dn_loss(
                outputs.get("dn_outputs"),
                targets,
                self.num_classes,
                outputs.get("dn_meta"),
                self.focal_alpha,
                self.focal_gamma,
            )
        )
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs: dict, target_sizes: Tensor, topk: int = 300) -> list[dict[str, Tensor]]:
        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]

        batch_size, _, num_classes = out_logits.shape
        prob = out_logits.sigmoid()
        topk = min(topk, prob.shape[1] * prob.shape[2])
        topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), topk, dim=1)

        scores = topk_values
        topk_boxes = topk_indexes // num_classes
        labels = topk_indexes % num_classes

        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).expand(-1, -1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for score, label, box in zip(scores, labels, boxes):
            results.append({"scores": score, "labels": label, "boxes": box})
        return results


def build_criterion_and_postprocess(args, num_classes: int) -> tuple[SetCriterion, PostProcess]:
    matcher = build_matcher(args)
    cls_loss_coef = getattr(args, "cls_loss_coef", 2.0)
    bbox_loss_coef = getattr(args, "bbox_loss_coef", 5.0)
    giou_loss_coef = getattr(args, "giou_loss_coef", 2.0)

    base_weight_dict = {
        "loss_ce": cls_loss_coef,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }
    weight_dict = dict(base_weight_dict)

    num_aux = max(getattr(args, "dec_layers", 6) - 1, 0)
    for layer_idx in range(num_aux):
        weight_dict.update({f"{key}_{layer_idx}": value for key, value in base_weight_dict.items()})

    weight_dict.update({f"{key}_enc": value for key, value in base_weight_dict.items()})
    weight_dict.update(
        {
            "loss_ce_dn": cls_loss_coef,
            "loss_bbox_dn": bbox_loss_coef,
            "loss_giou_dn": giou_loss_coef,
        }
    )
    for layer_idx in range(num_aux):
        weight_dict.update(
            {
                f"loss_ce_dn_aux_{layer_idx}": cls_loss_coef,
                f"loss_bbox_dn_aux_{layer_idx}": bbox_loss_coef,
                f"loss_giou_dn_aux_{layer_idx}": giou_loss_coef,
            }
        )

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=getattr(args, "focal_alpha", 0.25),
        focal_gamma=getattr(args, "focal_gamma", 2.0),
        losses=["labels", "boxes", "cardinality"],
    )
    postprocessors = PostProcess()
    return criterion, postprocessors
