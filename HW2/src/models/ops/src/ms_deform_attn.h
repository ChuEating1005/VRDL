#pragma once

#include <torch/extension.h>

#include <vector>

#include "ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "ms_deform_attn_cuda.h"
#endif

inline at::Tensor ms_deform_attn_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int im2col_step) {
    if (value.is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc,
            attn_weight,
            im2col_step);
#else
        TORCH_CHECK(false, "MultiScaleDeformableAttention was built without CUDA support");
#endif
    }
    return ms_deform_attn_cpu_forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step);
}

inline std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int im2col_step) {
    if (value.is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc,
            attn_weight,
            grad_output,
            im2col_step);
#else
        TORCH_CHECK(false, "MultiScaleDeformableAttention was built without CUDA support");
#endif
    }
    return ms_deform_attn_cpu_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        im2col_step);
}
