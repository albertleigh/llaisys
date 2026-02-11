//
// Created by ali on 2/15/26.
//

#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float gate_ele = utils::cast<float>(gate[i]);
        float up_ele = utils::cast<float>(up[i]);
        float swish_ele = gate_ele / (1.0f + std::exp(-gate_ele));
        float res = up_ele * swish_ele;
        out[i] = utils::cast<T>(res);
    }
}
} // namespace

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(gate), reinterpret_cast<const fp16_t *>(up), numel);
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(gate), reinterpret_cast<const bf16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu