# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportPrivateUsage=false, reportMissingParameterType=false, reportAttributeAccessIssue=false, reportUntypedFunctionDecorator=false
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.backbone import Joiner, build_backbone
from src.models.deformable_transformer import MLP, DeformableTransformer, build_deformable_transformer
from src.models.dn_components import dn_post_process, prepare_for_cdn
from src.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list


class DINO(nn.Module):
    def __init__(
        self,
        backbone: Joiner,
        transformer: DeformableTransformer,
        num_classes: int = 11,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        aux_loss: bool = True,
        with_box_refine: bool = True,
        two_stage: bool = True,
        mixed_selection: bool = True,
        dn_number: int = 100,
        dn_label_noise_ratio: float = 0.5,
        dn_box_noise_scale: float = 1.0,
    ):
        super().__init__()
        if not hasattr(transformer, "d_model"):
            raise AttributeError("DeformableTransformer must expose a d_model attribute for DINO.")
        if not with_box_refine:
            raise ValueError("DINO requires with_box_refine=True.")

        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.mixed_selection = mixed_selection
        self.dn_number = dn_number
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_box_noise_scale = dn_box_noise_scale

        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for in_channels in backbone.num_channels:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )

        in_channels = backbone.num_channels[-1]
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim

        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.constant_(proj[0].bias, 0)

        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)  # focal prior for stable classification start
        self.class_embed = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(transformer.decoder.num_layers)]
        )
        self.bbox_embed = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(transformer.decoder.num_layers)]
        )

        for class_embed in self.class_embed:
            nn.init.constant_(class_embed.bias, bias_value)
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight, 0)  # zero deltas make refinement start as identity
            nn.init.constant_(bbox_embed.layers[-1].bias, 0)

        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

    def forward(
        self,
        samples: NestedTensor | list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]] | dict[str, Tensor] | dict]:
        if isinstance(samples, (list, tuple)):
            samples = nested_tensor_from_tensor_list(list(samples))

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            src = self.input_proj[lvl](feat.tensors)
            mask = feat.mask
            if mask is None:
                mask = torch.zeros(
                    (src.shape[0], src.shape[-2], src.shape[-1]),
                    dtype=torch.bool,
                    device=src.device,
                )
            srcs.append(src)
            masks.append(mask)

        for lvl in range(len(features), self.num_feature_levels):
            if lvl == len(features):
                src = self.input_proj[lvl](features[-1].tensors)
            else:
                src = self.input_proj[lvl](srcs[-1])

            if samples.mask is None:
                mask = torch.zeros(
                    (src.shape[0], src.shape[-2], src.shape[-1]),
                    dtype=torch.bool,
                    device=src.device,
                )
            else:
                mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            pos.append(self.backbone[1](NestedTensor(src, mask)).to(src.dtype))
            srcs.append(src)
            masks.append(mask)

        if self.training and targets is not None:
            dn_args = (targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale)
            dn_query_label, dn_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args,
                training=True,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.transformer.d_model,
                label_enc=self.label_enc,
            )
        else:
            dn_query_label, dn_query_bbox, attn_mask, dn_meta = None, None, None, None

        hs, init_ref, inter_ref, enc_class, enc_coord_unact = self.transformer(
            srcs,
            masks,
            pos,
            query_embed=None,
            dn_query_label=dn_query_label,
            dn_query_bbox=dn_query_bbox,
            attn_mask=attn_mask,
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            ref = init_ref if lvl == 0 else inter_ref[lvl - 1]
            ref = inverse_sigmoid(ref)
            outputs_class = self.class_embed[lvl](hs[lvl])
            delta = self.bbox_embed[lvl](hs[lvl])
            outputs_coord = (delta + ref).sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if dn_meta is not None:
            outputs_class, outputs_coord, dn_outputs = dn_post_process(
                outputs_class,
                outputs_coord,
                dn_meta,
                aux_loss=self.aux_loss,
            )
        else:
            dn_outputs = None

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_class,
                "pred_boxes": enc_coord_unact.sigmoid(),
            }
        if dn_outputs is not None:
            out["dn_outputs"] = dn_outputs  # DN outputs are split from matching queries after decoder prediction.
            out["dn_meta"] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor) -> list[dict[str, Tensor]]:
        return [
            {"pred_logits": cls_layer, "pred_boxes": box_layer}
            for cls_layer, box_layer in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def build_dino(args) -> DINO:
    backbone = build_backbone(args)
    transformer = build_deformable_transformer(args)
    return DINO(
        backbone=backbone,
        transformer=transformer,
        num_classes=getattr(args, "num_classes", 11),
        num_queries=getattr(args, "num_queries", 900),
        num_feature_levels=getattr(args, "num_feature_levels", 4),
        aux_loss=getattr(args, "aux_loss", True),
        with_box_refine=getattr(args, "with_box_refine", True),
        two_stage=getattr(args, "two_stage", True),
        mixed_selection=getattr(args, "mixed_selection", True),
        dn_number=getattr(args, "dn_number", 100),
        dn_label_noise_ratio=getattr(args, "dn_label_noise_ratio", 0.5),
        dn_box_noise_scale=getattr(args, "dn_box_noise_scale", 1.0),
    )
