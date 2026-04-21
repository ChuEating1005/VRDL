// Adapted from the Apache-2.0 Deformable-DETR reference implementation.

#include "../ms_deform_attn_cuda.h"
#include "ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

namespace {

void check_cuda_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight) {
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index must be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");

    TORCH_CHECK(spatial_shapes.scalar_type() == at::kLong, "spatial_shapes must be int64");
    TORCH_CHECK(level_start_index.scalar_type() == at::kLong, "level_start_index must be int64");
    TORCH_CHECK(value.scalar_type() == sampling_loc.scalar_type(), "value and sampling_loc must have same dtype");
    TORCH_CHECK(value.scalar_type() == attn_weight.scalar_type(), "value and attn_weight must have same dtype");
}

}  // namespace

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int im2col_step) {
    check_cuda_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight);

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);
    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(im2col_step_ > 0, "im2col_step must be positive");
    TORCH_CHECK(batch % im2col_step_ == 0, "batch must be divisible by im2col_step");

    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());
    auto output_chunks = output.view({batch / im2col_step_, im2col_step_, num_query, num_heads, channels});

    const auto stream = at::cuda::getDefaultCUDAStream();
    const int64_t per_value_size = static_cast<int64_t>(spatial_size) * num_heads * channels;
    const int64_t per_sample_loc_size = static_cast<int64_t>(num_query) * num_heads * num_levels * num_point * 2;
    const int64_t per_attn_weight_size = static_cast<int64_t>(num_query) * num_heads * num_levels * num_point;

    for (int chunk = 0; chunk < batch / im2col_step_; ++chunk) {
        auto output_chunk = output_chunks.select(0, chunk);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            value.scalar_type(),
            "ms_deform_attn_forward_cuda",
            [&] {
                ms_deformable_im2col_cuda<scalar_t>(
                    stream.stream(),
                    value.data_ptr<scalar_t>() + chunk * im2col_step_ * per_value_size,
                    spatial_shapes.data_ptr<int64_t>(),
                    level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + chunk * im2col_step_ * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + chunk * im2col_step_ * per_attn_weight_size,
                    im2col_step_,
                    spatial_size,
                    num_heads,
                    channels,
                    num_levels,
                    num_query,
                    num_point,
                    output_chunk.data_ptr<scalar_t>());
            });
    }

    return output.view({batch, num_query, num_heads * channels});
}

std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int im2col_step) {
    check_cuda_inputs(value, spatial_shapes, level_start_index, sampling_loc, attn_weight);
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == value.scalar_type(), "grad_output must match value dtype");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);
    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(im2col_step_ > 0, "im2col_step must be positive");
    TORCH_CHECK(batch % im2col_step_ == 0, "batch must be divisible by im2col_step");

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    auto grad_output_chunks = grad_output.view({batch / im2col_step_, im2col_step_, num_query, num_heads, channels});
    const auto stream = at::cuda::getDefaultCUDAStream();
    const int64_t per_value_size = static_cast<int64_t>(spatial_size) * num_heads * channels;
    const int64_t per_sample_loc_size = static_cast<int64_t>(num_query) * num_heads * num_levels * num_point * 2;
    const int64_t per_attn_weight_size = static_cast<int64_t>(num_query) * num_heads * num_levels * num_point;

    for (int chunk = 0; chunk < batch / im2col_step_; ++chunk) {
        auto grad_output_chunk = grad_output_chunks.select(0, chunk);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            value.scalar_type(),
            "ms_deform_attn_backward_cuda",
            [&] {
                ms_deformable_col2im_cuda<scalar_t>(
                    stream.stream(),
                    grad_output_chunk.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>() + chunk * im2col_step_ * per_value_size,
                    spatial_shapes.data_ptr<int64_t>(),
                    level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + chunk * im2col_step_ * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + chunk * im2col_step_ * per_attn_weight_size,
                    im2col_step_,
                    spatial_size,
                    num_heads,
                    channels,
                    num_levels,
                    num_query,
                    num_point,
                    grad_value.data_ptr<scalar_t>() + chunk * im2col_step_ * per_value_size,
                    grad_sampling_loc.data_ptr<scalar_t>() + chunk * im2col_step_ * per_sample_loc_size,
                    grad_attn_weight.data_ptr<scalar_t>() + chunk * im2col_step_ * per_attn_weight_size);
            });
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}
