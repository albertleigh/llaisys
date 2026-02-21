//
// Created by ali on 2/22/26.
//

#include "linear_cuda.cuh"

#include <array>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

namespace {

// ── persistent cuBLASLt context ─────────────────────────────────────────────
// Lazily created, never freed — bounded memory, reused across calls.
struct CublasLtContext {
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    static constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024; // 32 MB

    CublasLtContext() {
        cublasStatus_t st = cublasLtCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("linear_cuda: cublasLtCreate failed");
        }
        CUDA_CHECK(cudaMalloc(&workspace, WORKSPACE_SIZE));
    }
};

static CublasLtContext &get_cublaslt() {
    static auto *ctx = new CublasLtContext(); // intentionally leaked
    return *ctx;
}

// ── bias-add kernel (used only by the small-matrix naive path) ──────────────
template <typename T>
__global__ void bias_add_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t j = i % N;
        if constexpr (std::is_same_v<T, float>) {
            out[i] += bias[j];
        } else {
            float val = llaisys::utils::cuda::to_float(out[i])
                      + llaisys::utils::cuda::to_float(bias[j]);
            if constexpr (std::is_same_v<T, __half>) {
                out[i] = __float2half(val);
            } else { // __nv_bfloat16
                out[i] = __float2bfloat16(val);
            }
        }
    }
}

template <typename T>
void launch_bias_add(T *out, const T *bias, size_t M, size_t N) {
    if (bias == nullptr) return;

    size_t total = M * N;
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > 65535) grid = 65535;

    bias_add_kernel<<<grid, block>>>(out, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// ── small-matrix naive kernel ───────────────────────────────────────────────
// Each thread computes one output element out[i, j].
template <typename T>
__global__ void linear_naive_kernel(T *out, const T *in, const T *weight,
                                    const T *bias, size_t M, size_t K, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    size_t i = idx / N;
    size_t j = idx % N;

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        sum += llaisys::utils::cuda::to_float(in[i * K + k])
             * llaisys::utils::cuda::to_float(weight[j * K + k]);
    }
    if (bias != nullptr) {
        sum += llaisys::utils::cuda::to_float(bias[j]);
    }

    if constexpr (std::is_same_v<T, float>) {
        out[idx] = sum;
    } else if constexpr (std::is_same_v<T, __half>) {
        out[idx] = __float2half(sum);
    } else { // __nv_bfloat16
        out[idx] = __float2bfloat16(sum);
    }
}

template <typename T>
void launch_linear_naive(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t K, size_t N) {
    size_t total = M * N;
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > 65535) grid = 65535;

    linear_naive_kernel<<<grid, block>>>(out, in, weight, bias, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ── cuBLASLt GEMM with fused bias epilogue ──────────────────────────────────
// out(M×N) = in(M×K) × weight^T  [+ bias(N)]
//
// Row-major tensors are reinterpreted as column-major by cuBLAS:
//   weight (N×K) row  → (K×N) col, OP_T  → effective (N×K)  = A
//   in     (M×K) row  → (K×M) col, OP_N                     = B
//   out    (M×N) row  → (N×M) col                            = C/D
//
// cuBLASLt: D = α·op(A)·op(B) + β·C  [+ bias via epilogue]
template <cudaDataType_t DataType, cublasComputeType_t ComputeType>
void linear_cublaslt(void *out, const void *in, const void *weight, const void *bias,
                     size_t M, size_t K, size_t N) {
    auto &ctx = get_cublaslt();
    float alpha = 1.0f, beta = 0.0f;

    // ── matmul descriptor ───────────────────────────────────────────────────
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, ComputeType, CUDA_R_32F);

    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &op_n, sizeof(op_n));

    // Fuse bias addition into the GEMM epilogue (avoids a separate kernel)
    if (bias != nullptr) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epi, sizeof(epi));
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias, sizeof(bias));
    }

    // ── matrix layouts (col-major view of row-major storage) ────────────────
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    cublasLtMatrixLayoutCreate(&layout_a, DataType, K, N, K); // weight: (K×N) col
    cublasLtMatrixLayoutCreate(&layout_b, DataType, K, M, K); // in:     (K×M) col
    cublasLtMatrixLayoutCreate(&layout_c, DataType, N, M, N); // out:    (N×M) col

    // ── heuristic algorithm selection ───────────────────────────────────────
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = CublasLtContext::WORKSPACE_SIZE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t heur;
    int n_result = 0;
    cublasLtMatmulAlgoGetHeuristic(ctx.handle, matmul_desc,
                                   layout_a, layout_b, layout_c, layout_c,
                                   pref, 1, &heur, &n_result);
    if (n_result == 0) {
        throw std::runtime_error("linear_cuda: cublasLt heuristic found no algorithm");
    }

    // ── execute ─────────────────────────────────────────────────────────────
    cublasStatus_t st = cublasLtMatmul(ctx.handle, matmul_desc,
                                       &alpha,
                                       weight, layout_a,
                                       in,     layout_b,
                                       &beta,
                                       out, layout_c,   // C (unused when β=0)
                                       out, layout_c,   // D (output)
                                       &heur.algo,
                                       ctx.workspace, ws,
                                       nullptr);        // default stream
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("linear_cuda: cublasLtMatmul failed");
    }

    // ── cleanup per-call descriptors ────────────────────────────────────────
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(matmul_desc);
}

} // anonymous namespace

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, std::array<size_t, 3> dims) {
    size_t M = dims[0]; // rows of input/output
    size_t K = dims[1]; // cols of input = cols of weight (reduction dim)
    size_t N = dims[2]; // cols of output = rows of weight (output features)

    // For very small matrices, use the naive kernel to avoid cuBLASLt overhead
    constexpr size_t THRESHOLD = 64;
    if (M * N * K < THRESHOLD * THRESHOLD * THRESHOLD) {
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return launch_linear_naive(reinterpret_cast<float *>(out),
                                       reinterpret_cast<const float *>(in),
                                       reinterpret_cast<const float *>(weight),
                                       reinterpret_cast<const float *>(bias),
                                       M, K, N);
        case LLAISYS_DTYPE_F16:
            return launch_linear_naive(reinterpret_cast<__half *>(out),
                                       reinterpret_cast<const __half *>(in),
                                       reinterpret_cast<const __half *>(weight),
                                       reinterpret_cast<const __half *>(bias),
                                       M, K, N);
        case LLAISYS_DTYPE_BF16:
            return launch_linear_naive(reinterpret_cast<__nv_bfloat16 *>(out),
                                       reinterpret_cast<const __nv_bfloat16 *>(in),
                                       reinterpret_cast<const __nv_bfloat16 *>(weight),
                                       reinterpret_cast<const __nv_bfloat16 *>(bias),
                                       M, K, N);
        default:
            throw std::runtime_error("linear_cuda: unsupported dtype");
        }
    }

    // cuBLASLt path — workspace-based algorithms + fused bias epilogue
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_cublaslt<CUDA_R_32F, CUBLAS_COMPUTE_32F>(
            out, in, weight, bias, M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_cublaslt<CUDA_R_16F, CUBLAS_COMPUTE_32F>(
            out, in, weight, bias, M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_cublaslt<CUDA_R_16BF, CUBLAS_COMPUTE_32F>(
            out, in, weight, bias, M, K, N);
    default:
        throw std::runtime_error("linear_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
