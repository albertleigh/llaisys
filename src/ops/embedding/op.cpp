#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "cuda/embedding_cuda.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be int64");
    ASSERT(index->isContiguous(), "Embedding: index must be contiguous.");
    size_t embeding_dim = weight->shape().back();
    ASSERT(out->shape().size() == 2 && out->shape()[1] == embeding_dim, "Embedding: output shape must be [index_size, embedding_dim]");
    ASSERT(index->shape().size() == 1 && index->shape()[0] == out->shape()[0], "Embedding: index must be a 1D tensor with size equal to output first dimension");

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), embeding_dim);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), embeding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), embeding_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
