#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/rms_norm_cuda.cuh"
#endif
#include <array>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(in, weight);
    ASSERT(out->shape().size() == 2, "RMSNorm: output tensor must be 2-D.");
    ASSERT(in->shape().size() == 2, "RMSNorm: input tensor must be 2-D.");
    ASSERT(weight->shape().size() == 1, "RMSNorm: weight tensor must be 1-D.");

    size_t dimi = in->shape()[0];
    size_t dimj = in->shape()[1];

    ASSERT(weight->shape()[0] == dimj, "RMSNorm: weight tensor must have the same size as the second dimension of the input tensor.");
    ASSERT(out->shape()[0] == dimi && out->shape()[1] == dimj, "RMSNorm: output tensor must have the same size as the first dimension of the input tensor.");
    ASSERT(eps > 0, "RMSNorm: epsilon must be greater than zero.");

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), {dimi, dimj});
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), {dimi, dimj});
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), {dimi, dimj});
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
