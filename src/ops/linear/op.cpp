#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/linear_cuda.cuh"
#endif

#include <array>

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }
    ASSERT(in->shape().size() == 2, "Input tensor must be 2D");
    ASSERT(weight->shape().size() == 2, "Weight tensor must be 2D");

    ASSERT(weight->shape()[1] == in->shape()[1], "Weight and in must have the same number of columns");
    ASSERT(in->shape()[0] == out->shape()[0], "In and out must have the same number of columns");
    ASSERT(weight->shape()[0] == out->shape()[1], "Weight and out must have the same number of columns");

    if (bias != nullptr) {
        ASSERT(bias->shape().size() == 1, "Bias tensor must be 1D");
        ASSERT(weight->shape()[0] == bias->shape()[0], "Weight and bias must have the same number of columns");
    }

    size_t dimi = in->shape()[0];
    size_t dimk = in->shape()[1];
    size_t dimj = weight->shape()[0];

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), {dimi, dimk, dimj});
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), {dimi, dimk, dimj});
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), {dimi, dimk, dimj});
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
