#include "op.hpp"
#include "cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/rearrange_cuda.cuh"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    std::vector<size_t> stride_in(in->strides().begin(), in->strides().end());
    std::vector<size_t> stride_out(out->strides().begin(), out->strides().end());

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rearrange(out->data(), in->data(), out->dtype(), out->shape(), stride_in, stride_out);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->shape(), stride_in, stride_out);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
    return cuda::rearrange(out->data(), in->data(), out->dtype(), out->shape(), stride_in, stride_out);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
