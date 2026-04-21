// Adapted from the Apache-2.0 Deformable-DETR reference implementation.

#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/Atomic.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int kCudaNumThreads = 256;

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(int n, int num_threads) {
    return (n + num_threads - 1) / num_threads;
}

template <typename scalar_t, typename accscalar_t>
__device__ accscalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t* bottom_data,
    int height,
    int width,
    int num_heads,
    int channels,
    accscalar_t h,
    accscalar_t w,
    int head,
    int channel) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const accscalar_t lh = h - h_low;
    const accscalar_t lw = w - w_low;
    const accscalar_t hh = accscalar_t(1) - lh;
    const accscalar_t hw = accscalar_t(1) - lw;

    const int w_stride = num_heads * channels;
    const int h_stride = width * w_stride;
    const int base_ptr = head * channels + channel;

    accscalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        v1 = static_cast<accscalar_t>(bottom_data[h_low * h_stride + w_low * w_stride + base_ptr]);
    }
    accscalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        v2 = static_cast<accscalar_t>(bottom_data[h_low * h_stride + w_high * w_stride + base_ptr]);
    }
    accscalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        v3 = static_cast<accscalar_t>(bottom_data[h_high * h_stride + w_low * w_stride + base_ptr]);
    }
    accscalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        v4 = static_cast<accscalar_t>(bottom_data[h_high * h_stride + w_high * w_stride + base_ptr]);
    }

    const accscalar_t w1 = hh * hw;
    const accscalar_t w2 = hh * lw;
    const accscalar_t w3 = lh * hw;
    const accscalar_t w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename scalar_t, typename accscalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t* bottom_data,
    int height,
    int width,
    int num_heads,
    int channels,
    accscalar_t h,
    accscalar_t w,
    int head,
    int channel,
    accscalar_t top_grad,
    accscalar_t attn_weight,
    scalar_t* grad_value,
    accscalar_t* grad_sampling_loc,
    accscalar_t* grad_attn_weight) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const accscalar_t lh = h - h_low;
    const accscalar_t lw = w - w_low;
    const accscalar_t hh = accscalar_t(1) - lh;
    const accscalar_t hw = accscalar_t(1) - lw;

    const int w_stride = num_heads * channels;
    const int h_stride = width * w_stride;
    const int base_ptr = head * channels + channel;

    const accscalar_t w1 = hh * hw;
    const accscalar_t w2 = hh * lw;
    const accscalar_t w3 = lh * hw;
    const accscalar_t w4 = lh * lw;
    const accscalar_t top_grad_value = top_grad * attn_weight;

    accscalar_t grad_h_weight = 0;
    accscalar_t grad_w_weight = 0;
    accscalar_t sampled = 0;

    if (h_low >= 0 && w_low >= 0) {
        const int ptr = h_low * h_stride + w_low * w_stride + base_ptr;
        const accscalar_t v = static_cast<accscalar_t>(bottom_data[ptr]);
        sampled += w1 * v;
        grad_h_weight -= hw * v;
        grad_w_weight -= hh * v;
        gpuAtomicAdd(grad_value + ptr, static_cast<scalar_t>(w1 * top_grad_value));
    }
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr = h_low * h_stride + w_high * w_stride + base_ptr;
        const accscalar_t v = static_cast<accscalar_t>(bottom_data[ptr]);
        sampled += w2 * v;
        grad_h_weight -= lw * v;
        grad_w_weight += hh * v;
        gpuAtomicAdd(grad_value + ptr, static_cast<scalar_t>(w2 * top_grad_value));
    }
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr = h_high * h_stride + w_low * w_stride + base_ptr;
        const accscalar_t v = static_cast<accscalar_t>(bottom_data[ptr]);
        sampled += w3 * v;
        grad_h_weight += hw * v;
        grad_w_weight -= lh * v;
        gpuAtomicAdd(grad_value + ptr, static_cast<scalar_t>(w3 * top_grad_value));
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr = h_high * h_stride + w_high * w_stride + base_ptr;
        const accscalar_t v = static_cast<accscalar_t>(bottom_data[ptr]);
        sampled += w4 * v;
        grad_h_weight += lw * v;
        grad_w_weight += lh * v;
        gpuAtomicAdd(grad_value + ptr, static_cast<scalar_t>(w4 * top_grad_value));
    }

    grad_attn_weight[0] = top_grad * sampled;
    grad_sampling_loc[0] = static_cast<accscalar_t>(width) * grad_w_weight * top_grad_value;
    grad_sampling_loc[1] = static_cast<accscalar_t>(height) * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    int n,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    scalar_t* data_col) {
    using accscalar_t = at::acc_type<scalar_t, true>;

    CUDA_KERNEL_LOOP(index, n) {
        int tmp = index;
        const int c_col = tmp % channels;
        tmp /= channels;
        const int sampling_index = tmp;
        const int head = tmp % num_heads;
        tmp /= num_heads;
        tmp /= num_query;
        const int batch = tmp;

        const int head_stride = num_heads * channels;
        const int value_offset = batch * spatial_size * head_stride;
        int weight_ptr = sampling_index * num_levels * num_point;
        int loc_ptr = weight_ptr * 2;

        accscalar_t col = 0;
        for (int level = 0; level < num_levels; ++level) {
            const int spatial_h = static_cast<int>(data_spatial_shapes[level * 2]);
            const int spatial_w = static_cast<int>(data_spatial_shapes[level * 2 + 1]);
            const int level_start_id = static_cast<int>(data_level_start_index[level]);
            const scalar_t* value_ptr = data_value + value_offset + level_start_id * head_stride;

            for (int point = 0; point < num_point; ++point) {
                const accscalar_t loc_w = static_cast<accscalar_t>(data_sampling_loc[loc_ptr]);
                const accscalar_t loc_h = static_cast<accscalar_t>(data_sampling_loc[loc_ptr + 1]);
                const accscalar_t weight = static_cast<accscalar_t>(data_attn_weight[weight_ptr]);
                const accscalar_t h_im = loc_h * spatial_h - accscalar_t(0.5);
                const accscalar_t w_im = loc_w * spatial_w - accscalar_t(0.5);

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    col += ms_deform_attn_im2col_bilinear<scalar_t, accscalar_t>(
                        value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        head,
                        c_col) * weight;
                }

                ++weight_ptr;
                loc_ptr += 2;
            }
        }
        data_col[index] = static_cast<scalar_t>(col);
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel(
    int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
    using accscalar_t = at::acc_type<scalar_t, true>;

    CUDA_KERNEL_LOOP(index, n) {
        int tmp = index;
        const int c_col = tmp % channels;
        tmp /= channels;
        const int sampling_index = tmp;
        const int head = tmp % num_heads;
        tmp /= num_heads;
        tmp /= num_query;
        const int batch = tmp;

        const accscalar_t top_grad = static_cast<accscalar_t>(grad_col[index]);
        const int head_stride = num_heads * channels;
        const int value_offset = batch * spatial_size * head_stride;
        int weight_ptr = sampling_index * num_levels * num_point;
        int loc_ptr = weight_ptr * 2;

        for (int level = 0; level < num_levels; ++level) {
            const int spatial_h = static_cast<int>(data_spatial_shapes[level * 2]);
            const int spatial_w = static_cast<int>(data_spatial_shapes[level * 2 + 1]);
            const int level_start_id = static_cast<int>(data_level_start_index[level]);
            const scalar_t* value_ptr = data_value + value_offset + level_start_id * head_stride;
            scalar_t* grad_value_ptr = grad_value + value_offset + level_start_id * head_stride;

            for (int point = 0; point < num_point; ++point) {
                const accscalar_t loc_w = static_cast<accscalar_t>(data_sampling_loc[loc_ptr]);
                const accscalar_t loc_h = static_cast<accscalar_t>(data_sampling_loc[loc_ptr + 1]);
                const accscalar_t weight = static_cast<accscalar_t>(data_attn_weight[weight_ptr]);
                const accscalar_t h_im = loc_h * spatial_h - accscalar_t(0.5);
                const accscalar_t w_im = loc_w * spatial_w - accscalar_t(0.5);

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    accscalar_t grad_loc[2] = {0, 0};
                    accscalar_t grad_weight[1] = {0};
                    ms_deform_attn_col2im_bilinear<scalar_t, accscalar_t>(
                        value_ptr,
                        spatial_h,
                        spatial_w,
                        num_heads,
                        channels,
                        h_im,
                        w_im,
                        head,
                        c_col,
                        top_grad,
                        weight,
                        grad_value_ptr,
                        grad_loc,
                        grad_weight);
                    gpuAtomicAdd(grad_sampling_loc + loc_ptr, static_cast<scalar_t>(grad_loc[0]));
                    gpuAtomicAdd(grad_sampling_loc + loc_ptr + 1, static_cast<scalar_t>(grad_loc[1]));
                    gpuAtomicAdd(grad_attn_weight + weight_ptr, static_cast<scalar_t>(grad_weight[0]));
                }

                ++weight_ptr;
                loc_ptr += 2;
            }
        }
    }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    scalar_t* data_col) {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    ms_deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels, kCudaNumThreads), kCudaNumThreads, 0, stream>>>(
        num_kernels,
        data_value,
        data_spatial_shapes,
        data_level_start_index,
        data_sampling_loc,
        data_attn_weight,
        batch_size,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        data_col);
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    ms_deformable_col2im_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels, kCudaNumThreads), kCudaNumThreads, 0, stream>>>(
        num_kernels,
        grad_col,
        data_value,
        data_spatial_shapes,
        data_level_start_index,
        data_sampling_loc,
        data_attn_weight,
        batch_size,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        grad_value,
        grad_sampling_loc,
        grad_attn_weight);
}
