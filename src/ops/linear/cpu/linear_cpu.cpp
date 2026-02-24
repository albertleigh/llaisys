//
// Created by ali on 2/11/26.
//

#include "../../../utils.hpp"
#include "linear_cpu.hpp"
#include "llaisys.h"

#include <array>
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::array<size_t, 3> dims) {
    size_t dimi = dims[0];
    size_t dimk = dims[1];
    size_t dimj = dims[2];

    for (size_t i = 0; i < dimi; ++i) {
        for (size_t j = 0; j < dimj; ++j) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                float sum = 0.0f;
                for (size_t k = 0; k < dimk; ++k) {
                    sum += utils::cast<float>(in[i * dimk + k]) * utils::cast<float>(weight[j * dimk + k]);
                }

                if (bias != nullptr) {
                    sum += utils::cast<float>(bias[j]);
                }
                out[i * dimj + j] = utils::cast<T>(sum);
            } else {
                T sum = 0.0f;
                for (size_t k = 0; k < dimk; ++k) {
                    sum += in[i * dimk + k] * weight[j * dimk + k];
                }

                if (bias != nullptr) {
                    sum += bias[j];
                }
                out[i * dimj + j] = sum;
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, std::array<size_t, 3> dims) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const fp16_t *>(weight), reinterpret_cast<const fp16_t *>(bias), dims);
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), dims);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const bf16_t *>(weight), reinterpret_cast<const bf16_t *>(bias), dims);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu