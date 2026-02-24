//
// Created by ali on 2/12/26.
//

#include "rope_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, const std::vector<size_t> &dims) {
    const size_t seq_len = dims[0];
    const size_t num_heads = dims[1];
    const size_t head_dim = dims[2];
    const size_t half_dim = head_dim / 2;

    std::vector<float> denoms(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        double exponent = (2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
        double denom_d = std::pow(static_cast<double>(theta), exponent);
        denoms[i] = static_cast<float>(denom_d);
    }

    for (size_t s = 0; s < seq_len; ++s) {
        const int64_t pos = pos_ids[s];
        float pos_f = static_cast<float>(pos);
        for (size_t h = 0; h < num_heads; ++h) {
            size_t offset = s * (num_heads * head_dim) + h * head_dim;

            const T *src_vec = in + offset;
            T *dst_vec = out + offset;

            for (size_t j = 0; j < half_dim; ++j) {
                float angle = pos_f / denoms[j];

                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                float a = utils::cast<float>(src_vec[j]);
                float b = utils::cast<float>(src_vec[j + half_dim]);

                float a_out = a * cos_val - b * sin_val;
                float b_out = b * cos_val + a * sin_val;

                dst_vec[j] = utils::cast<T>(a_out);
                dst_vec[j + half_dim] = utils::cast<T>(b_out);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t dtype, const std::vector<size_t> &dims) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, dims);
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, dims);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, dims);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
