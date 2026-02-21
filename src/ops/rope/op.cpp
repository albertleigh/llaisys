#include "op.hpp"
#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/rope_cuda.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "ROPE: pos_ids must be int64");

    ASSERT(in->shape().size() == 3, "ROPE: input tensor must be 3-D.");
    ASSERT(out->shape().size() == 3, "ROPE: output tensor must be 3-D.");
    ASSERT(pos_ids->shape().size() == 1, "ROPE: pos_ids tensor must be 1-D.");

    size_t seq_len = in->shape()[0];
    size_t head_dim = in->shape()[2];

    ASSERT(seq_len == pos_ids->shape()[0], "ROPE: seq_len must be equal to pos_ids first dimension.");
    ASSERT(in->shape() == out->shape(), "ROPE: input and output tensor must have the same shape.");
    ASSERT(head_dim % 2 == 0, "ROPE: head_dim must be even number.");

    ASSERT(in->isContiguous() && pos_ids->isContiguous() && out->isContiguous(), "ROPE: input, pos_ids and output tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
