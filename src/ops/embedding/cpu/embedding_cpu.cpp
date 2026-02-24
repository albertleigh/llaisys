//
// Created by ali on 2/11/26.
//

#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

#include <cstring>

namespace {
template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t size, size_t embedding_dim) {
    for (size_t i = 0; i < size; i++) {
        T *out_row_ptr = out + i * embedding_dim;
        const T *weight_row_ptr = weight + index[i] * embedding_dim;
        std::memcpy(out_row_ptr, weight_row_ptr, embedding_dim * sizeof(T));
    }
}
} // namespace

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t size, size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), size, embedding_dim);
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), size, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), size, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu