import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ops.functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch
from src.models.ops.modules import MSDeformAttn


def level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    return torch.cat(
        [areas.new_zeros(1), areas.cumsum(0)[:-1]],
        dim=0,
    )


def normalized_attention_weights(shape, *, device, dtype):
    weights = torch.rand(shape, device=device, dtype=dtype)
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True)
    return weights


def assert_close(name, actual, expected, atol, rtol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}")


def forward_reference_test():
    device = torch.device("cuda")
    dtype = torch.float32
    spatial_shapes = torch.tensor([[84, 84], [42, 42], [21, 21], [11, 11]], device=device, dtype=torch.long)
    level_start = level_start_index(spatial_shapes)
    batch_size = 2
    num_heads = 8
    channels = 32
    num_query = 300
    num_levels = spatial_shapes.size(0)
    num_points = 4
    total_spatial = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(batch_size, total_spatial, num_heads, channels, device=device, dtype=dtype)
    sampling_locations = torch.rand(
        batch_size,
        num_query,
        num_heads,
        num_levels,
        num_points,
        2,
        device=device,
        dtype=dtype,
    )
    attention_weights = normalized_attention_weights(
        (batch_size, num_query, num_heads, num_levels, num_points),
        device=device,
        dtype=dtype,
    )

    cuda_out = MSDeformAttnFunction.apply(
        value,
        spatial_shapes,
        level_start,
        sampling_locations,
        attention_weights,
        64,
    )
    ref_out = ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    assert_close("forward", cuda_out, ref_out, atol=1e-3, rtol=1e-3)


def backward_reference_test():
    device = torch.device("cuda")
    dtype = torch.float64
    spatial_shapes = torch.tensor([[4, 5], [2, 3]], device=device, dtype=torch.long)
    level_start = level_start_index(spatial_shapes)
    batch_size = 1
    num_heads = 4
    channels = 8
    num_query = 6
    num_levels = spatial_shapes.size(0)
    num_points = 4
    total_spatial = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    base_value = torch.randn(batch_size, total_spatial, num_heads, channels, device=device, dtype=dtype)
    base_sampling = torch.rand(
        batch_size,
        num_query,
        num_heads,
        num_levels,
        num_points,
        2,
        device=device,
        dtype=dtype,
    )
    base_attention = torch.rand(
        batch_size,
        num_query,
        num_heads,
        num_levels,
        num_points,
        device=device,
        dtype=dtype,
    )

    value_cuda = base_value.clone().requires_grad_(True)
    sampling_cuda = base_sampling.clone().requires_grad_(True)
    attention_cuda = base_attention.clone().requires_grad_(True)

    value_ref = base_value.clone().requires_grad_(True)
    sampling_ref = base_sampling.clone().requires_grad_(True)
    attention_ref = base_attention.clone().requires_grad_(True)

    grad_output = torch.randn(batch_size, num_query, num_heads * channels, device=device, dtype=dtype)

    out_cuda = MSDeformAttnFunction.apply(
        value_cuda,
        spatial_shapes,
        level_start,
        sampling_cuda,
        attention_cuda,
        64,
    )
    out_ref = ms_deform_attn_core_pytorch(value_ref, spatial_shapes, sampling_ref, attention_ref)

    out_cuda.backward(grad_output)
    out_ref.backward(grad_output)

    assert_close("backward/output", out_cuda, out_ref, atol=1e-8, rtol=1e-8)
    assert_close("backward/grad_value", value_cuda.grad, value_ref.grad, atol=1e-8, rtol=1e-8)
    assert_close(
        "backward/grad_sampling_locations",
        sampling_cuda.grad,
        sampling_ref.grad,
        atol=5e-8,
        rtol=5e-8,
    )
    assert_close(
        "backward/grad_attention_weights",
        attention_cuda.grad,
        attention_ref.grad,
        atol=1e-8,
        rtol=1e-8,
    )


def module_smoke_test():
    device = torch.device("cuda")
    module = MSDeformAttn().to(device)
    module.eval()

    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5], [3, 3]], device=device, dtype=torch.long)
    level_start = level_start_index(spatial_shapes)
    total_spatial = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    batch_size = 2
    num_query = 32
    query = torch.randn(batch_size, num_query, 256, device=device)
    input_flatten = torch.randn(batch_size, total_spatial, 256, device=device)
    padding_mask = torch.zeros(batch_size, total_spatial, device=device, dtype=torch.bool)
    padding_mask[:, -5:] = True

    reference_points_2d = torch.rand(batch_size, num_query, 4, 2, device=device)
    reference_points_4d = torch.rand(batch_size, num_query, 4, 4, device=device)

    output_2d = module(
        query,
        reference_points_2d,
        input_flatten,
        spatial_shapes,
        level_start,
        padding_mask,
    )
    output_4d = module(
        query,
        reference_points_4d,
        input_flatten,
        spatial_shapes,
        level_start,
        padding_mask,
    )

    assert output_2d.shape == (batch_size, num_query, 256)
    assert output_4d.shape == (batch_size, num_query, 256)


def main() -> int:
    if not torch.cuda.is_available():
        print("FAILED")
        print("CUDA is not available", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    try:
        forward_reference_test()
        backward_reference_test()
        module_smoke_test()
    except Exception as exc:
        print("FAILED")
        print(exc, file=sys.stderr)
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
