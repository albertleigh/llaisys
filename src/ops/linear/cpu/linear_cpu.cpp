//
// Created by ali on 2/11/26.
//

#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include "../../../utils/omp_compat.hpp"
#include "llaisys.h"

#include <array>
#include <cstring>
#include <vector>

#ifdef ENABLE_MKL
#include <mkl.h> // Intel MKL (includes mkl_cblas.h)
#else
#include <openblas/cblas.h> // OpenBLAS header from vcpkg
#endif

namespace llaisys::ops::cpu {

// ─── Small-matrix fallback (shared by all backends) ─────────────────────────
template <typename T>
static void linear_naive_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t K, size_t N) {
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
}

// ─── OpenMP blocked half-precision fallback ─────────────────────────────────
template <typename T>
static void linear_blocked_half_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t K, size_t N) {
    // Initialize output with bias
    if (bias != nullptr) {
        OMP_PARALLEL_FOR_COLLAPSE2
        for (omp_idx_t i = 0; i < OMP_CAST(M); ++i) {
            for (omp_idx_t j = 0; j < OMP_CAST(N); ++j) {
                out[i * N + j] = bias[j];
            }
        }
    } else {
        OMP_PARALLEL_FOR
        for (omp_idx_t i = 0; i < OMP_CAST(M); ++i) {
            std::memset(out + i * N, 0, N * sizeof(T));
        }
    }

    // Block sizes for cache optimization
    constexpr size_t BLOCK_I = 32;
    constexpr size_t BLOCK_J = 64;
    constexpr size_t BLOCK_K = 256;

#ifdef _MSC_VER
    // MSVC: no collapse – flatten the 2D block grid into a single loop
    const ptrdiff_t n_blocks_i = static_cast<ptrdiff_t>((M + BLOCK_I - 1) / BLOCK_I);
    const ptrdiff_t n_blocks_j = static_cast<ptrdiff_t>((N + BLOCK_J - 1) / BLOCK_J);
    const ptrdiff_t total_blocks = n_blocks_i * n_blocks_j;

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t ii = static_cast<size_t>(block_idx / n_blocks_j) * BLOCK_I;
        const size_t jj = static_cast<size_t>(block_idx % n_blocks_j) * BLOCK_J;
#else
    // GCC / Clang: use collapse(2) over 2D block grid
#pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < M; ii += BLOCK_I) {
        for (size_t jj = 0; jj < N; jj += BLOCK_J) {
#endif
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
                        OMP_SIMD
                        for (size_t j = jj; j < j_end; ++j) {
                            float w_val = utils::cast<float>(weight[j * K + k]);
                            acc_buffer[(i - ii) * BLOCK_J + (j - jj)] += in_val * w_val;
                        }
                    }
                }
            }

            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
                    out[i * N + j] = utils::cast<T>(acc_buffer[(i - ii) * BLOCK_J + (j - jj)]);
                }
            }
#ifdef _MSC_VER
    }
#else
        }
    }
#endif
}

// ─── Main linear dispatch ───────────────────────────────────────────────────
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::array<size_t, 3> dims) {
    size_t M = dims[0]; // rows of input/output (batch size or sequence length)
    size_t K = dims[1]; // cols of input = cols of weight (reduction dimension)
    size_t N = dims[2]; // cols of output = rows of weight (output features)

    // For small matrices, use simple implementation
    constexpr size_t THRESHOLD = 64;
    if (M * N * K < THRESHOLD * THRESHOLD * THRESHOLD) {
        return linear_naive_(out, in, weight, bias, M, K, N);
    }

    // ── float32 path ────────────────────────────────────────────────────────
    if constexpr (std::is_same_v<T, float>) {
        // Both MKL and OpenBLAS expose cblas_sgemm
        // out(M×N) = in(M×K) * weight^T(N×K)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f, in, static_cast<int>(K),
                    weight, static_cast<int>(K),
                    0.0f, out, static_cast<int>(N));

        if (bias != nullptr) {
            OMP_PARALLEL_FOR_COLLAPSE2
            for (omp_idx_t i = 0; i < OMP_CAST(M); ++i) {
                for (omp_idx_t j = 0; j < OMP_CAST(N); ++j) {
                    out[i * N + j] += bias[j];
                }
            }
        }
        return;
    }

#ifdef ENABLE_MKL
    // ── MKL fp16 path: cblas_hgemm (native half-precision GEMM) ────────────
    if constexpr (std::is_same_v<T, fp16_t>) {
        // fp16_t._v is uint16_t, same layout as MKL_F16 (unsigned short)
        const MKL_F16 *a = reinterpret_cast<const MKL_F16 *>(in);
        const MKL_F16 *b = reinterpret_cast<const MKL_F16 *>(weight);
        MKL_F16 *c = reinterpret_cast<MKL_F16 *>(out);

        // alpha = 1.0h, beta = 0.0h  (as MKL_F16 bit patterns)
        MKL_F16 alpha, beta;
        {
            // IEEE 754 half: 1.0 = 0x3C00, 0.0 = 0x0000
            uint16_t one = 0x3C00u, zero = 0x0000u;
            std::memcpy(&alpha, &one, sizeof(MKL_F16));
            std::memcpy(&beta, &zero, sizeof(MKL_F16));
        }

        cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
                    alpha, a, static_cast<MKL_INT>(K),
                    b, static_cast<MKL_INT>(K),
                    beta, c, static_cast<MKL_INT>(N));

        if (bias != nullptr) {
            OMP_PARALLEL_FOR_COLLAPSE2
            for (omp_idx_t i = 0; i < OMP_CAST(M); ++i) {
                for (omp_idx_t j = 0; j < OMP_CAST(N); ++j) {
                    // Add bias in float, write back as fp16
                    float val = utils::cast<float>(out[i * N + j]) + utils::cast<float>(bias[j]);
                    out[i * N + j] = utils::cast<T>(val);
                }
            }
        }
        return;
    }

    // ── MKL bf16 path: cblas_gemm_bf16bf16f32 (bf16 input → f32 output) ────
    if constexpr (std::is_same_v<T, bf16_t>) {
        const MKL_BF16 *a = reinterpret_cast<const MKL_BF16 *>(in);
        const MKL_BF16 *b = reinterpret_cast<const MKL_BF16 *>(weight);

        // MKL bf16 GEMM accumulates into float32 output
        std::vector<float> c_f32(M * N, 0.0f);

        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                               static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
                               1.0f, a, static_cast<MKL_INT>(K),
                               b, static_cast<MKL_INT>(K),
                               0.0f, c_f32.data(), static_cast<MKL_INT>(N));

// Convert f32 result back to bf16, adding bias
        OMP_PARALLEL_FOR_COLLAPSE2
        for (omp_idx_t i = 0; i < OMP_CAST(M); ++i) {
            for (omp_idx_t j = 0; j < OMP_CAST(N); ++j) {
                float val = c_f32[i * N + j];
                if (bias != nullptr) {
                    val += utils::cast<float>(bias[j]);
                }
                out[i * N + j] = utils::cast<T>(val);
            }
        }
        return;
    }
#endif // ENABLE_MKL

    // ── Fallback: OpenMP blocked half-precision implementation ──────────────
    linear_blocked_half_(out, in, weight, bias, M, K, N);
}
} // namespace llaisys::ops::cpu

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