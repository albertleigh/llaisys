//
// Created by ali on 2/11/26.
//

#include "../../../utils.hpp"
#include "linear_cpu.hpp"
#include "llaisys.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <openblas/cblas.h>  // OpenBLAS header from vcpkg

namespace llaisys::ops::cpu {

// Optimized linear operation with OpenBLAS for large matrices
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::array<size_t, 3> dims) {
    size_t M = dims[0];  // rows of input/output (batch size or sequence length)
    size_t K = dims[1];  // cols of input = cols of weight (reduction dimension)
    size_t N = dims[2];  // cols of output = rows of weight (output features)

    // For small matrices, use simple implementation
    constexpr size_t THRESHOLD = 64;

    if (M * N * K < THRESHOLD * THRESHOLD * THRESHOLD) {
        // Small matrix - use simple nested loops
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += utils::cast<float>(in[i * K + k]) * utils::cast<float>(weight[j * K + k]);
                    }
                    if (bias != nullptr) {
                        sum += utils::cast<float>(bias[j]);
                    }
                    out[i * N + j] = utils::cast<T>(sum);
                } else {
                    T sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += in[i * K + k] * weight[j * K + k];
                    }
                    if (bias != nullptr) {
                        sum += bias[j];
                    }
                    out[i * N + j] = sum;
                }
            }
        }
        return;
    }

    // Large matrix - use OpenBLAS for float32, optimized OpenMP for others
    if constexpr (std::is_same_v<T, float>) {
        // Use OpenBLAS SGEMM for maximum performance
        // out(M×N) = in(M×K) * weight^T(N×K)
        // C = alpha * A * B^T + beta * C

        // Initialize output with zeros
        std::memset(out, 0, M * N * sizeof(float));

        // Matrix multiplication: out = in * weight^T
        cblas_sgemm(CblasRowMajor,      // Row-major order
                    CblasNoTrans,        // Don't transpose A (in)
                    CblasTrans,          // Transpose B (weight)
                    M,                   // Rows of A and C
                    N,                   // Cols of B^T and C
                    K,                   // Cols of A, rows of B
                    1.0f,                // alpha
                    in, K,               // A matrix with leading dimension K
                    weight, K,           // B matrix with leading dimension K
                    0.0f,                // beta (don't add to C)
                    out, N);             // C matrix with leading dimension N

        // Add bias if present
        if (bias != nullptr) {
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    out[i * N + j] += bias[j];
                }
            }
        }

        return;
    }

    // For half-precision types, use optimized OpenMP blocked implementation
    // (OpenBLAS doesn't have native fp16/bf16 support)

    // Initialize output with bias
    if (bias != nullptr) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                out[i * N + j] = bias[j];
            }
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {
            std::memset(out + i * N, 0, N * sizeof(T));
        }
    }

    // Block sizes for cache optimization
    constexpr size_t BLOCK_I = 32;
    constexpr size_t BLOCK_J = 64;
    constexpr size_t BLOCK_K = 256;

    // For half precision types, accumulate in float and convert at the end
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < M; ii += BLOCK_I) {
        for (size_t jj = 0; jj < N; jj += BLOCK_J) {
            size_t i_end = (ii + BLOCK_I < M) ? ii + BLOCK_I : M;
            size_t j_end = (jj + BLOCK_J < N) ? jj + BLOCK_J : N;

            // Allocate float accumulation buffer for this block
            float acc_buffer[BLOCK_I * BLOCK_J] = {0};

            // Copy initial bias values to accumulation buffer
            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
                    acc_buffer[(i - ii) * BLOCK_J + (j - jj)] = utils::cast<float>(out[i * N + j]);
                }
            }

            for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                size_t k_end = (kk + BLOCK_K < K) ? kk + BLOCK_K : K;

                // Process block with i-k-j order for better cache locality
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        float in_val = utils::cast<float>(in[i * K + k]);

                        // Vectorizable inner loop
                        #pragma omp simd
                        for (size_t j = jj; j < j_end; ++j) {
                            float w_val = utils::cast<float>(weight[j * K + k]);
                            acc_buffer[(i - ii) * BLOCK_J + (j - jj)] += in_val * w_val;
                        }
                    }
                }
            }

            // Convert back to half precision
            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
                    out[i * N + j] = utils::cast<T>(acc_buffer[(i - ii) * BLOCK_J + (j - jj)]);
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