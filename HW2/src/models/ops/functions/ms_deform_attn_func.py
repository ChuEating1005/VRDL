import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def _load_extension():
    try:
        return importlib.import_module("MultiScaleDeformableAttention")
    except ModuleNotFoundError:
        ops_dir = Path(__file__).resolve().parents[1]
        ops_dir_str = str(ops_dir)
        if ops_dir_str not in sys.path:
            sys.path.insert(0, ops_dir_str)
        return importlib.import_module("MultiScaleDeformableAttention")


class MSDeformAttnFunction(Function):
    # The compiled CUDA op only supports fp32. Force inputs to fp32 under autocast
    # so AMP training (bf16/fp16) works transparently. Output is cast back to the
    # autocast dtype by the autograd engine automatically.
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ext = _load_extension()
        ctx.im2col_step = int(im2col_step)
        value = value.contiguous()
        value_spatial_shapes = value_spatial_shapes.contiguous()
        value_level_start_index = value_level_start_index.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        output = ext.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        ext = _load_extension()
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_output = grad_output.to(value.dtype).contiguous()
        grad_value, grad_sampling_loc, grad_attn_weight = ext.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
):
    n, _, n_heads, channels = value.shape
    _, n_query, _, n_levels, n_points, _ = sampling_locations.shape
    if torch.is_tensor(value_spatial_shapes):
        spatial_shapes = value_spatial_shapes.tolist()
    else:
        spatial_shapes = list(value_spatial_shapes)

    split_sizes = [int(h) * int(w) for h, w in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_values = []

    for level, (height, width) in enumerate(spatial_shapes):
        height = int(height)
        width = int(width)
        value_level = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(n * n_heads, channels, height, width)
        )
        grid_level = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(
            value_level,
            grid_level,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_values.append(sampled)

    attention = attention_weights.transpose(1, 2).reshape(
        n * n_heads,
        1,
        n_query,
        n_levels * n_points,
    )
    output = (
        torch.stack(sampled_values, dim=-2).flatten(-2) * attention
    ).sum(-1)
    output = output.view(n, n_heads * channels, n_query)
    return output.transpose(1, 2).contiguous()
