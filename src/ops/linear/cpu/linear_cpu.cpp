//
// Created by ali on 2/11/26.
//

#include "../../../utils.hpp"
#include "linear_cpu.hpp"
#include "llaisys.h"

#include <array>
#include <cstddef>
#include <cstring>

namespace llaisys::ops::cpu {

// Optimized linear operation with blocking and parallelization
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::array<size_t, 3> dims) {
    size_t dimi = dims[0];
    size_t dimk = dims[1];
    size_t dimj = dims[2];

    // For small matrices, use simple implementation
    constexpr size_t THRESHOLD = 64;

    if (dimi * dimj * dimk < THRESHOLD * THRESHOLD * THRESHOLD) {
        // Small matrix - use simple nested loops
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
        return;
    }

    // Large matrix - use optimized blocked implementation with OpenMP

    // Initialize output with bias
    if (bias != nullptr) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < dimi; ++i) {
            for (size_t j = 0; j < dimj; ++j) {
                out[i * dimj + j] = bias[j];
            }
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < dimi; ++i) {
            std::memset(out + i * dimj, 0, dimj * sizeof(T));
        }
    }

    // Block sizes for cache optimization
    constexpr size_t BLOCK_I = 32;
    constexpr size_t BLOCK_J = 64;
    constexpr size_t BLOCK_K = 256;

    if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
        // For half precision types, accumulate in float and convert at the end
        #pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < dimi; ii += BLOCK_I) {
            for (size_t jj = 0; jj < dimj; jj += BLOCK_J) {
                size_t i_end = (ii + BLOCK_I < dimi) ? ii + BLOCK_I : dimi;
                size_t j_end = (jj + BLOCK_J < dimj) ? jj + BLOCK_J : dimj;

                // Allocate float accumulation buffer for this block
                float acc_buffer[BLOCK_I * BLOCK_J] = {0};

                // Copy initial bias values to accumulation buffer
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        acc_buffer[(i - ii) * BLOCK_J + (j - jj)] = utils::cast<float>(out[i * dimj + j]);
                    }
                }

                for (size_t kk = 0; kk < dimk; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < dimk) ? kk + BLOCK_K : dimk;

                    // Process block with i-k-j order for better cache locality
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t k = kk; k < k_end; ++k) {
                            float in_val = utils::cast<float>(in[i * dimk + k]);

                            // Vectorizable inner loop
                            #pragma omp simd
                            for (size_t j = jj; j < j_end; ++j) {
                                float w_val = utils::cast<float>(weight[j * dimk + k]);
                                acc_buffer[(i - ii) * BLOCK_J + (j - jj)] += in_val * w_val;
                            }
                        }
                    }
                }

                // Convert back to half precision
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        out[i * dimj + j] = utils::cast<T>(acc_buffer[(i - ii) * BLOCK_J + (j - jj)]);
                    }
                }
            }
        }
    } else {
        // For full precision types (float)
        #pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < dimi; ii += BLOCK_I) {
            for (size_t jj = 0; jj < dimj; jj += BLOCK_J) {
                size_t i_end = (ii + BLOCK_I < dimi) ? ii + BLOCK_I : dimi;
                size_t j_end = (jj + BLOCK_J < dimj) ? jj + BLOCK_J : dimj;

                for (size_t kk = 0; kk < dimk; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < dimk) ? kk + BLOCK_K : dimk;

                    // Process block with i-k-j order for better cache locality
                    for (size_t i = ii; i < i_end; ++i) {
                        for (size_t k = kk; k < k_end; ++k) {
                            T in_val = in[i * dimk + k];

                            // Vectorizable inner loop - compiler can auto-vectorize with SIMD
                            #pragma omp simd
                            for (size_t j = jj; j < j_end; ++j) {
                                out[i * dimj + j] += in_val * weight[j * dimk + k];
                            }
                        }
                    }
                }
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