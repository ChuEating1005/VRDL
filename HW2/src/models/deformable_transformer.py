# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false, reportPrivateUsage=false, reportMissingParameterType=false, reportAttributeAccessIssue=false
# ------------------------------------------------------------------------
# Adapted from IDEA-Research DINO / Deformable DETR transformer logic.
# ------------------------------------------------------------------------

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.ops.modules import MSDeformAttn
from src.util.misc import inverse_sigmoid


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {name}.")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if idx < self.num_layers - 1 else layer(x)
        return x


def _meshgrid(y: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    try:
        return torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        return torch.meshgrid(y, x)


def gen_encoder_output_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor,
    spatial_shapes: Tensor,
) -> tuple[Tensor, Tensor]:
    n_batch, _, _ = memory.shape
    proposals = []
    cur = 0
    for lvl, (height, width) in enumerate(spatial_shapes):
        level_mask = memory_padding_mask[:, cur : cur + height * width].view(n_batch, height, width, 1)
        valid_h = torch.sum(~level_mask[:, :, 0, 0], dim=1)
        valid_w = torch.sum(~level_mask[:, 0, :, 0], dim=1)

        grid_y, grid_x = _meshgrid(
            torch.linspace(0, int(height) - 1, int(height), dtype=torch.float32, device=memory.device),
            torch.linspace(0, int(width) - 1, int(width), dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], dim=1).view(n_batch, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(n_batch, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * (0.05 * (2.0**lvl))
        proposals.append(torch.cat([grid, wh], dim=-1).view(n_batch, -1, 4))
        cur += int(height * width)

    output_proposals = torch.cat(proposals, dim=1)
    output_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(dim=-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1.0 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_valid, float("inf"))

    output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~output_valid, 0.0)
    return output_memory, output_proposals


def gen_sineembed_for_position(pos_tensor: Tensor, dim_t_factor: int = 128) -> Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(dim_t_factor, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim_t_factor)

    embeds = []
    for idx in range(pos_tensor.size(-1)):
        pos = pos_tensor[..., idx] * scale
        pos = pos[..., None] / dim_t
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
        embeds.append(pos)

    if pos_tensor.size(-1) == 2:
        return torch.cat((embeds[1], embeds[0]), dim=-1)
    if pos_tensor.size(-1) == 4:
        return torch.cat((embeds[1], embeds[0], embeds[2], embeds[3]), dim=-1)
    raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor | None,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = _meshgrid(
                torch.linspace(0.5, int(height) - 0.5, int(height), dtype=torch.float32, device=device),
                torch.linspace(0.5, int(width) - 0.5, int(width), dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), dim=-1))
        reference_points = torch.cat(reference_points_list, dim=1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor | None,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: Tensor,
        level_start_index: Tensor,
        src_padding_mask: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt_bqc = tgt.transpose(0, 1)
        query_bqc = self.with_pos_embed(tgt_bqc, None if query_pos is None else query_pos.transpose(0, 1))
        tgt2 = self.cross_attn(
            query_bqc,
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2.transpose(0, 1)))
        return self.forward_ffn(tgt)


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = True,
        d_model: int = 256,
        query_dim: int = 4,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.norm = norm if norm is not None else nn.LayerNorm(d_model)
        self.d_model = d_model
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.look_forward_twice = True

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        src: Tensor,
        src_spatial_shapes: Tensor,
        src_level_start_index: Tensor,
        src_valid_ratios: Tensor,
        query_pos: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = tgt
        static_query_pos = None
        if query_pos is not None:
            static_query_pos = query_pos.transpose(0, 1) if query_pos.shape[:2] == tgt.shape[:2] else query_pos

        intermediate = []
        intermediate_reference_points = []

        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [src_valid_ratios, src_valid_ratios], dim=-1
                )[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            query_sine_embed = gen_sineembed_for_position(reference_points)
            dynamic_query_pos = self.ref_point_head(query_sine_embed)
            if static_query_pos is not None:
                dynamic_query_pos = dynamic_query_pos + static_query_pos

            # Decoder self-attention runs in (Q, B, C); MSDeformAttn uses (B, Q, C).
            output = layer(
                output,
                dynamic_query_pos.transpose(0, 1),
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                self_attn_mask,
            )

            if self.bbox_embed is not None:
                delta = self.bbox_embed[layer_idx](output.transpose(0, 1))
                new_reference_points = (delta + inverse_sigmoid(reference_points)).sigmoid()
                # LFT: keep the graph through reference updates instead of detaching here.
                reference_points = new_reference_points if self.look_forward_twice else new_reference_points.detach()

            intermediate.append(self.norm(output).transpose(0, 1))
            intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return intermediate[-1].unsqueeze(0), intermediate_reference_points[-1].unsqueeze(0)


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        two_stage: bool = True,
        two_stage_num_proposals: int = 900,
        mixed_selection: bool = True,
        look_forward_twice: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.mixed_selection = mixed_selection
        self.look_forward_twice = look_forward_twice

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm=nn.LayerNorm(d_model),
            return_intermediate=True,
            d_model=d_model,
            query_dim=4,
            num_feature_levels=num_feature_levels,
        )
        self.decoder.look_forward_twice = look_forward_twice

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.tgt_embed = nn.Embedding(num_queries, d_model)

        if not two_stage:
            self.reference_points = nn.Embedding(num_queries, 4)

        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        for module in self.modules():
            if hasattr(module, "_reset_parameters") and module is not self:
                if isinstance(module, MSDeformAttn):
                    module._reset_parameters()
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.normal_(self.tgt_embed.weight)
        if hasattr(self, "reference_points"):
            nn.init.uniform_(self.reference_points.weight, 0.0, 1.0)
            self.reference_points.weight.data = inverse_sigmoid(self.reference_points.weight.data)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], dim=1)
        valid_w = torch.sum(~mask[:, 0, :], dim=1)
        return torch.stack([valid_w.float() / width, valid_h.float() / height], dim=-1)

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        query_embed: Tensor | None = None,
        dn_query_label: Tensor | None = None,
        dn_query_bbox: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        batch_size = srcs[0].shape[0]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, height, width = src.shape
            spatial_shapes.append((height, width))
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed + self.level_embed[lvl].view(1, 1, -1))
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes_tensor.new_zeros((1,)), spatial_shapes_tensor.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], dim=1)

        memory = self.encoder(
            src_flatten,
            spatial_shapes_tensor,
            level_start_index,
            valid_ratios,
            pos=lvl_pos_embed_flatten,
            padding_mask=mask_flatten,
        )

        enc_outputs_class = None
        enc_outputs_coord_unact = None
        static_query_pos = None

        if self.two_stage:
            if self.decoder.class_embed is None or self.decoder.bbox_embed is None:
                raise ValueError("decoder.class_embed and decoder.bbox_embed must be set before two-stage forward")

            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes_tensor)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            class_head = self.decoder.class_embed[-1]
            bbox_head = self.decoder.bbox_embed[-1]
            enc_outputs_class = class_head(output_memory)
            enc_outputs_coord_unact = bbox_head(output_memory) + output_proposals

            topk = min(self.two_stage_num_proposals, enc_outputs_class.shape[1])
            topk_indices = torch.topk(enc_outputs_class.max(dim=-1)[0], topk, dim=1)[1]
            gather_boxes = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
            gather_hidden = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)

            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, gather_boxes)
            reference_points = topk_coords_unact.sigmoid()
            topk_memory = torch.gather(output_memory, 1, gather_hidden)

            pos_trans_out = self.pos_trans_norm(self.pos_trans(gen_sineembed_for_position(reference_points)))
            query_pos_seed, tgt_from_pos = pos_trans_out.split(self.d_model, dim=2)
            static_query_pos = query_pos_seed.transpose(0, 1)

            # Mixed Query Selection: positions come from encoder proposals, content stays learned.
            if self.mixed_selection:
                tgt = self.tgt_embed.weight[:topk].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                tgt = tgt_from_pos if query_embed is None else query_embed.unsqueeze(0).expand(batch_size, -1, -1)
                if tgt.shape[1] != topk:
                    tgt = topk_memory
        else:
            num_queries = self.num_queries if query_embed is None else query_embed.shape[0]
            tgt = self.tgt_embed.weight[:num_queries].unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points.weight[:num_queries].unsqueeze(0).expand(batch_size, -1, -1).sigmoid()

        if dn_query_label is not None and dn_query_bbox is not None:
            reference_points = torch.cat([dn_query_bbox.sigmoid(), reference_points], dim=1)
            tgt = torch.cat([dn_query_label, tgt], dim=1)
            if static_query_pos is not None:
                dn_query_pos = torch.zeros(
                    batch_size,
                    dn_query_label.shape[1],
                    self.d_model,
                    device=tgt.device,
                    dtype=tgt.dtype,
                )
                static_query_pos = torch.cat([dn_query_pos, static_query_pos.transpose(0, 1)], dim=1).transpose(0, 1)

        init_reference = reference_points
        hs, inter_references = self.decoder(
            tgt=tgt.transpose(0, 1),
            reference_points=reference_points,
            src=memory,
            src_spatial_shapes=spatial_shapes_tensor,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=static_query_pos,
            src_padding_mask=mask_flatten,
            self_attn_mask=attn_mask,
        )
        return hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact


def build_deformable_transformer(args: object) -> DeformableTransformer:
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        mixed_selection=args.mixed_selection,
        look_forward_twice=args.look_forward_twice,
    )
