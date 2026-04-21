# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""CDN utilities adapted from IDEA-Research DINO.

Labels follow this project's convention: foreground classes are 1..10 and
class index 0 stays unused; DN negatives use all-zero sigmoid targets.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from src.util.misc import inverse_sigmoid


def _none_tuple():
    return None, None, None, None


def _sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float,
    gamma: float,
    normalizer: float,
) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = loss * alpha_t
    return loss.sum() / max(normalizer, 1.0)


def _build_dn_targets(
    targets: list[dict],
    dn_meta: dict,
    num_classes: int,
    device: torch.device,
    cls_dtype: torch.dtype,
    box_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = len(targets)
    single_pad = int(dn_meta["single_pad"])
    num_dn_group = int(dn_meta["num_dn_group"])
    pad_size = int(dn_meta["pad_size"])

    cls_targets = torch.zeros(batch_size, pad_size, num_classes, device=device, dtype=cls_dtype)
    box_targets = torch.zeros(batch_size, pad_size, 4, device=device, dtype=box_dtype)
    pos_mask = torch.zeros(batch_size, pad_size, device=device, dtype=torch.bool)

    for batch_idx, target in enumerate(targets):
        num_gt = int(target["labels"].numel())
        if num_gt == 0:
            continue

        labels = target["labels"].to(device=device, dtype=torch.long)
        boxes = target["boxes"].to(device=device, dtype=box_dtype)
        for group_idx in range(num_dn_group):
            group_start = group_idx * 2 * single_pad
            pos_slice = slice(group_start, group_start + num_gt)
            cls_targets[batch_idx, pos_slice, labels] = 1.0
            box_targets[batch_idx, pos_slice] = boxes
            pos_mask[batch_idx, pos_slice] = True

    return cls_targets, box_targets, pos_mask


def _compute_single_dn_loss(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    targets: list[dict],
    num_classes: int,
    dn_meta: dict,
    focal_alpha: float,
    focal_gamma: float,
) -> dict[str, Tensor]:
    cls_targets, box_targets, pos_mask = _build_dn_targets(
        targets,
        dn_meta,
        num_classes,
        pred_logits.device,
        pred_logits.dtype,
        pred_boxes.dtype,
    )
    num_pos = float(pos_mask.sum().item())

    losses = {
        "loss_ce_dn": _sigmoid_focal_loss(pred_logits, cls_targets, focal_alpha, focal_gamma, num_pos)
        * pred_logits.shape[1],
        "loss_bbox_dn": pred_boxes.sum() * 0.0,
        "loss_giou_dn": pred_boxes.sum() * 0.0,
    }
    if pos_mask.any():
        pos_pred_boxes = pred_boxes[pos_mask]
        pos_tgt_boxes = box_targets[pos_mask]
        losses["loss_bbox_dn"] = F.l1_loss(pos_pred_boxes, pos_tgt_boxes, reduction="sum") / max(num_pos, 1.0)
        losses["loss_giou_dn"] = (
            1.0
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pos_pred_boxes),
                    box_cxcywh_to_xyxy(pos_tgt_boxes),
                )
            )
        ).sum() / max(num_pos, 1.0)
    return losses


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    if not training or dn_args is None:
        return _none_tuple()

    targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
    if dn_number <= 0 or not targets:
        return _none_tuple()

    known_num = [int(t["labels"].numel()) for t in targets]
    single_pad = max(known_num, default=0)
    if single_pad == 0:
        return _none_tuple()

    scalar = max(dn_number // max(single_pad * 2, 1), 1)
    device = label_enc.weight.device
    batch_size = len(targets)

    labels = torch.cat([t["labels"].to(device=device, dtype=torch.long) for t in targets], dim=0)
    boxes = torch.cat([t["boxes"].to(device=device) for t in targets], dim=0)
    batch_idx = torch.cat(
        [torch.full((num_gt,), idx, device=device, dtype=torch.long) for idx, num_gt in enumerate(known_num)],
        dim=0,
    )
    if labels.numel() == 0:
        return _none_tuple()

    repeat_factor = 2 * scalar
    known_labels = labels.repeat(repeat_factor)
    known_bboxs = boxes.repeat(repeat_factor, 1)
    known_bid = batch_idx.repeat(repeat_factor)
    known_labels_expanded = known_labels.clone()
    known_bbox_expand = known_bboxs.clone()

    if label_noise_ratio > 0:
        chosen_indice = torch.nonzero(
            torch.rand_like(known_labels_expanded.float()) < (label_noise_ratio * 0.5),
            as_tuple=False,
        ).flatten()
        if chosen_indice.numel() > 0:
            new_label = torch.randint(
                0,
                num_classes,
                (chosen_indice.numel(),),
                device=device,
                dtype=torch.long,
            )
            known_labels_expanded.scatter_(0, chosen_indice, new_label)

    pad_size = single_pad * repeat_factor
    num_known = boxes.shape[0]
    positive_idx = torch.arange(num_known, device=device, dtype=torch.long).unsqueeze(0).repeat(scalar, 1)
    positive_idx += (torch.arange(scalar, device=device, dtype=torch.long) * num_known * 2).unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + num_known

    if box_noise_scale > 0:
        # Official DINO CDN noise: positives use rand_part in [0, 1), negatives use [1, 2),
        # both with random sign on xyxy corners, so negatives are always farther away.
        known_bbox_xyxy = box_cxcywh_to_xyxy(known_bboxs)
        diff = torch.zeros_like(known_bboxs)
        diff[:, :2] = known_bboxs[:, 2:] / 2
        diff[:, 2:] = known_bboxs[:, 2:] / 2

        rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.long).to(known_bboxs.dtype)
        rand_sign = rand_sign * 2.0 - 1.0
        rand_part = torch.rand_like(known_bboxs)
        rand_part[negative_idx] += 1.0
        rand_part = rand_part * rand_sign

        known_bbox_xyxy = (known_bbox_xyxy + rand_part * diff * box_noise_scale).clamp(min=0.0, max=1.0)
        known_bbox_expand = box_xyxy_to_cxcywh(known_bbox_xyxy)

    input_label_embed = label_enc(known_labels_expanded.long())
    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    input_query_label = torch.zeros(
        batch_size,
        pad_size,
        hidden_dim,
        device=device,
        dtype=input_label_embed.dtype,
    )
    input_query_bbox = torch.zeros(batch_size, pad_size, 4, device=device, dtype=input_bbox_embed.dtype)

    map_known_indice = torch.cat(
        [torch.arange(num_gt, device=device, dtype=torch.long) for num_gt in known_num],
        dim=0,
    )
    map_known_indice = torch.cat(
        [map_known_indice + single_pad * idx for idx in range(repeat_factor)],
        dim=0,
    )
    input_query_label[(known_bid, map_known_indice)] = input_label_embed
    input_query_bbox[(known_bid, map_known_indice)] = input_bbox_embed

    total_len = pad_size + num_queries
    # True means blocked, matching nn.MultiheadAttention attn_mask convention.
    attn_mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)
    attn_mask[pad_size:, :pad_size] = True
    for group_idx in range(scalar):
        start = group_idx * 2 * single_pad
        end = start + 2 * single_pad
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:pad_size] = True

    dn_meta = {
        "single_pad": single_pad,
        "num_dn_group": scalar,
        "pad_size": pad_size,
    }
    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss=True, _set_aux_loss: Callable | None = None):
    if not dn_meta or dn_meta.get("pad_size", 0) <= 0:
        return outputs_class, outputs_coord, None

    pad_size = int(dn_meta["pad_size"])
    output_known_class = outputs_class[:, :, :pad_size, :]
    output_known_coord = outputs_coord[:, :, :pad_size, :]
    outputs_class = outputs_class[:, :, pad_size:, :]
    outputs_coord = outputs_coord[:, :, pad_size:, :]

    dn_outputs = {
        "pred_logits": output_known_class[-1],
        "pred_boxes": output_known_coord[-1],
    }
    if aux_loss:
        if _set_aux_loss is not None:
            dn_outputs["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        else:
            dn_outputs["aux_outputs"] = [
                {"pred_logits": cls_layer, "pred_boxes": box_layer}
                for cls_layer, box_layer in zip(output_known_class[:-1], output_known_coord[:-1])
            ]
    return outputs_class, outputs_coord, dn_outputs


def compute_dn_loss(dn_outputs, targets, num_classes, dn_meta, focal_alpha, focal_gamma):
    device = None
    if dn_outputs is not None and "pred_logits" in dn_outputs:
        device = dn_outputs["pred_logits"].device
    elif targets:
        device = targets[0]["labels"].device
    else:
        device = torch.device("cpu")

    zero = torch.zeros((), device=device)
    losses = {
        "loss_ce_dn": zero,
        "loss_bbox_dn": zero,
        "loss_giou_dn": zero,
    }
    if dn_outputs is None or not dn_meta or dn_meta.get("pad_size", 0) <= 0:
        return losses

    main_losses = _compute_single_dn_loss(
        dn_outputs["pred_logits"],
        dn_outputs["pred_boxes"],
        targets,
        num_classes,
        dn_meta,
        focal_alpha,
        focal_gamma,
    )
    losses.update(main_losses)

    for idx, aux_output in enumerate(dn_outputs.get("aux_outputs", [])):
        aux_losses = _compute_single_dn_loss(
            aux_output["pred_logits"],
            aux_output["pred_boxes"],
            targets,
            num_classes,
            dn_meta,
            focal_alpha,
            focal_gamma,
        )
        losses[f"loss_ce_dn_aux_{idx}"] = aux_losses["loss_ce_dn"]
        losses[f"loss_bbox_dn_aux_{idx}"] = aux_losses["loss_bbox_dn"]
        losses[f"loss_giou_dn_aux_{idx}"] = aux_losses["loss_giou_dn"]

    return losses
