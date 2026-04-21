// Adapted from the Apache-2.0 Deformable-DETR reference implementation.

#include "ms_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "MSDeformAttn forward");
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "MSDeformAttn backward");
}
