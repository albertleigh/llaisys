//
// Created by ali on 2/11/26.
//

#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <array>
#include <cmath>
namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, std::array<size_t, 2> dims) {
    size_t dimi = dims[0];
    size_t dimj = dims[1];

    for (size_t i = 0; i < dimi; ++i) {
        float sum_seq = 0.0f;
        for (size_t j = 0; j < dimj; ++j) {
            float val = utils::cast<float>(in[i * dimj + j]);
            sum_seq += val * val;
        }

        float rms = std::sqrt(sum_seq / dimj + eps);
        float inv_rms = 1.0f / rms;
        for (size_t j = 0; j < dimj; ++j) {
            float val = utils::cast<float>(in[i * dimj + j]);
            float w = utils::cast<float>(weight[j]);
            float res = val * w * inv_rms;
            out[i * dimj + j] = utils::cast<T>(res);
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t dtype, std::array<size_t, 2> dims) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const fp16_t *>(weight), eps, dims);
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, dims);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const bf16_t *>(weight), eps, dims);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
