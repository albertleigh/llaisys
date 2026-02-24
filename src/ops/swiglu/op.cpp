#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(gate->shape() == up->shape(), "SwiGLU: gate and up must be of the same shape.");
    ASSERT(up->shape() == out->shape(), "SwiGLU: up and out must be of the same shape.");
    ASSERT(gate->isContiguous() && up->isContiguous() && out->isContiguous(), "SwiGLU: gate, up and out tensors must be contiguous.");

    size_t numel = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
