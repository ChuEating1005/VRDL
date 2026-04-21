import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ..functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch


def _is_power_of_2(value: int) -> bool:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid input for _is_power_of_2: {value!r}")
    return (value & (value - 1)) == 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")

        d_per_head = d_model // n_heads
        if not _is_power_of_2(d_per_head):
            warnings.warn(
                "MSDeformAttn is fastest when d_model // n_heads is a power of 2.",
                stacklevel=2,
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().amax(dim=-1, keepdim=True)
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for point_idx in range(self.n_points):
            grid_init[:, :, point_idx, :] *= point_idx + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        constant_(self.attention_weights.weight, 0.0)
        constant_(self.attention_weights.bias, 0.0)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.0)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        n, len_q, _ = query.shape
        n_in, len_in, _ = input_flatten.shape
        if n != n_in:
            raise ValueError(f"batch size mismatch: query batch={n}, input batch={n_in}")

        device = input_flatten.device
        input_spatial_shapes = input_spatial_shapes.to(device=device, dtype=torch.long)
        input_level_start_index = input_level_start_index.to(device=device, dtype=torch.long)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.to(device=device, dtype=torch.bool)

        expected_len = int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item())
        if expected_len != len_in:
            raise ValueError(f"input_flatten length {len_in} does not match spatial shapes total {expected_len}")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(n, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]],
                dim=-1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        if value.is_cuda:
            output = MSDeformAttnFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = ms_deform_attn_core_pytorch(
                value,
                input_spatial_shapes,
                sampling_locations,
                attention_weights,
            )
        return self.output_proj(output)
