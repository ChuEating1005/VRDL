#pragma once

#include <torch/extension.h>

#include <vector>

at::Tensor ms_deform_attn_cpu_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int im2col_step);

std::vector<at::Tensor> ms_deform_attn_cpu_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int im2col_step);
