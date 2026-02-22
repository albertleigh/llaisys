//
// Created by ali on 2/22/26.
//

#include "self_attention_cuda.cuh"

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"

namespace {

// ── helpers ─────────────────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, __half>)             return __float2half(v);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(v);
    else                                                  return static_cast<T>(v);
}

constexpr int MAX_HD = 256;

// ── persistent cuBLAS context ───────────────────────────────────────────────
struct CublasContext {
    cublasHandle_t handle = nullptr;
    CublasContext() {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("self_attention_cuda: cublasCreate failed");
    }
};

static CublasContext &get_cublas() {
    static auto *ctx = new CublasContext(); // intentionally leaked
    return *ctx;
}

// ── persistent workspace (grows, never shrinks) ─────────────────────────────
static void *get_workspace(size_t bytes) {
    static void *ws      = nullptr;
    static size_t ws_cap = 0;
    if (bytes > ws_cap) {
        if (ws) cudaFree(ws);
        CUDA_CHECK(cudaMalloc(&ws, bytes));
        ws_cap = bytes;
    }
    return ws;
}

// ── causal-mask + row-wise softmax kernel ───────────────────────────────────
// scores: (total_rows, sk) contiguous,  total_rows = nh * seq.
// Row ordering: head h, query position i  →  row = h * seq + i.
// Causal condition: key position j is visible iff j <= sk − seq + i.
template <typename T>
__global__ void causal_softmax_kernel(
    T *__restrict__ scores,
    size_t seq, size_t sk, size_t total_rows) {

    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row >= total_rows) return;

    size_t query_pos  = row % seq;
    size_t causal_end = sk - seq + query_pos; // last visible key position

    T *s = scores + row * sk;

    // pass 1: max over non-masked positions
    float mx = -1e30f;
    for (size_t j = 0; j <= causal_end && j < sk; ++j) {
        float val = llaisys::utils::cuda::to_float(__ldg(&s[j]));
        if (val > mx) mx = val;
    }

    // pass 2: exp + sum; write zero for masked positions
    float sum = 0.0f;
    for (size_t j = 0; j < sk; ++j) {
        if (j > causal_end) {
            s[j] = from_float<T>(0.0f);
        } else {
            float e = expf(llaisys::utils::cuda::to_float(s[j]) - mx);
            sum += e;
            s[j] = from_float<T>(e);
        }
    }

    // pass 3: normalize
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (size_t j = 0; j <= causal_end && j < sk; ++j)
        s[j] = from_float<T>(llaisys::utils::cuda::to_float(s[j]) * inv);
}

// ── online-softmax fused kernel (small-size fast path) ──────────────────────
// One thread per (query-position, head) pair.  Single pass over K using the
// online-softmax trick: running max m, sum s, output accumulator o[].
// All arithmetic is done in float; K/V vectors are pre-loaded into float
// registers per iteration to minimise half→float conversion overhead.
template <typename T>
__global__ void self_attention_kernel(
    T       *__restrict__ attn,
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    float scale,
    size_t seq, size_t nh, size_t d,
    size_t sk,  size_t nh_kv, size_t dv,
    size_t n_rep) {

    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= seq * nh) return;

    size_t i     = idx / nh;
    size_t h     = idx % nh;
    size_t h_kv  = h / n_rep;
    size_t q_abs = sk - seq + i;

    const size_t k_stride = nh_kv * d;
    const size_t v_stride = nh_kv * dv;
    const size_t k_off    = h_kv * d;
    const size_t v_off    = h_kv * dv;

    // pre-load query into float registers via __ldg (read-only cache)
    const T *q_base = q + (i * nh * d) + (h * d);
    float q_f[MAX_HD];
    for (size_t l = 0; l < d; ++l)
        q_f[l] = llaisys::utils::cuda::to_float(__ldg(&q_base[l]));

    float m = -1e30f, s_sum = 0.0f;
    float o[MAX_HD];
    for (size_t l = 0; l < dv; ++l) o[l] = 0.0f;

    float k_f[MAX_HD]; // reused per iteration

    for (size_t j = 0; j <= q_abs && j < sk; ++j) {
        // load K[j] into float registers via __ldg
        const T *k_ptr = k + j * k_stride + k_off;
        for (size_t l = 0; l < d; ++l)
            k_f[l] = llaisys::utils::cuda::to_float(__ldg(&k_ptr[l]));

        float dot = 0.0f;
        for (size_t l = 0; l < d; ++l)
            dot += q_f[l] * k_f[l];

        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float corr  = expf(m - m_new);
        float p     = expf(score - m_new);
        s_sum = s_sum * corr + p;

        // load V[j] into the same temp buffer and accumulate
        const T *v_ptr = v + j * v_stride + v_off;
        for (size_t l = 0; l < dv; ++l) {
            float vf = llaisys::utils::cuda::to_float(__ldg(&v_ptr[l]));
            o[l] = o[l] * corr + p * vf;
        }
        m = m_new;
    }

    float inv_s = (s_sum > 0.0f) ? (1.0f / s_sum) : 0.0f;
    T *out = attn + (i * nh * dv) + (h * dv);
    for (size_t l = 0; l < dv; ++l)
        out[l] = from_float<T>(o[l] * inv_s);
}

template <typename T>
void launch_self_attention(T *attn, const T *q, const T *k, const T *v,
                           float scale,
                           size_t seq, size_t nh, size_t d,
                           size_t sk,  size_t nh_kv, size_t dv) {
    size_t total = seq * nh;
    int block = 256;
    int grid  = static_cast<int>((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    size_t n_rep = nh / nh_kv;
    self_attention_kernel<<<grid, block>>>(
        attn, q, k, v, scale, seq, nh, d, sk, nh_kv, dv, n_rep);
    CUDA_CHECK(cudaGetLastError());
}

// ── cuBLAS-accelerated path ─────────────────────────────────────────────────
// Decomposes attention into three stages:
//   GEMM 1 :  S = scale · Q · Kᵀ       (batched, fused scaling via α)
//   Kernel  :  S ← causal_softmax(S)
//   GEMM 2 :  O = S · V                 (batched)
//
// GQA is handled by looping over KV-head groups; within each group the same
// K / V head is broadcast to n_rep query heads via strideA = 0.
//
// Memory layouts (row-major, reinterpreted as col-major by cuBLAS):
//   Q    (seq, nh, d)      →  per-head (d×seq) col, ldb = nh·d, stride = d
//   K    (sk,  nh_kv, d)   →  per-head (d×sk)  col, lda = nh_kv·d
//   V    (sk,  nh_kv, dv)  →  per-head (dv×sk) col, lda = nh_kv·dv
//   S    (nh,  seq, sk)    →  per-head (sk×seq) col, contiguous
//   attn (seq, nh, dv)     →  per-head (dv×seq) col, ldc = nh·dv, stride = dv
template <typename T, cudaDataType_t DataType>
void self_attention_cublas(
    T *attn, const T *q, const T *k, const T *v,
    float scale,
    size_t seq, size_t nh, size_t d,
    size_t sk,  size_t nh_kv, size_t dv) {

    auto &ctx   = get_cublas();
    size_t n_rep = nh / nh_kv;

    // workspace for scores: (nh, seq, sk) contiguous
    T *scores = static_cast<T *>(get_workspace(nh * seq * sk * sizeof(T)));

    float alpha_scale = scale, alpha_one = 1.0f, beta_zero = 0.0f;

    // ── GEMM 1: S = scale · Q · Kᵀ ─────────────────────────────────────────
    // cuBLAS col-major:  C(sk×seq) = α · K^T(sk×d) · Q(d×seq)
    int M1 = static_cast<int>(sk),  N1 = static_cast<int>(seq),  K1 = static_cast<int>(d);
    for (size_t g = 0; g < nh_kv; ++g) {
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            ctx.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,                           // transa, transb
            M1, N1, K1,                                         // M, N, K
            &alpha_scale,                                        // α = scale
            k + g * d,        DataType, static_cast<int>(nh_kv * d), 0LL,                          // A, lda, strideA (K repeated)
            q + g * n_rep * d, DataType, static_cast<int>(nh * d),    static_cast<long long>(d),    // B, ldb, strideB
            &beta_zero,                                                                             // β = 0
            scores + g * n_rep * static_cast<long long>(seq * sk), DataType, M1, static_cast<long long>(seq * sk), // C, ldc, strideC
            static_cast<int>(n_rep),                             // batchCount
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("self_attention_cuda: GEMM1 failed");
    }

    // ── causal softmax ──────────────────────────────────────────────────────
    {
        size_t total_rows = nh * seq;
        int blk  = 256;
        int grid = static_cast<int>((total_rows + blk - 1) / blk);
        if (grid > 65535) grid = 65535;
        causal_softmax_kernel<<<grid, blk>>>(scores, seq, sk, total_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── GEMM 2: O = softmax(S) · V ─────────────────────────────────────────
    // cuBLAS col-major:  C(dv×seq) = V(dv×sk) · S(sk×seq)
    int M2 = static_cast<int>(dv),  N2 = static_cast<int>(seq),  K2 = static_cast<int>(sk);
    for (size_t g = 0; g < nh_kv; ++g) {
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            ctx.handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M2, N2, K2,
            &alpha_one,
            v + g * dv,        DataType, static_cast<int>(nh_kv * dv), 0LL,
            scores + g * n_rep * static_cast<long long>(seq * sk), DataType, K2, static_cast<long long>(seq * sk),
            &beta_zero,
            attn + g * n_rep * dv, DataType, static_cast<int>(nh * dv), static_cast<long long>(dv),
            static_cast<int>(n_rep),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("self_attention_cuda: GEMM2 failed");
    }
}

} // anonymous namespace

namespace llaisys::ops::cuda {
void self_attention(
    std::byte *attn, const std::byte *q, const std::byte *k, const std::byte *v,
    float scale, llaisysDataType_t dtype,
    const std::vector<size_t> &q_shap,
    const std::vector<size_t> &k_shap,
    const std::vector<size_t> &v_shap) {

    const size_t seq   = q_shap[0];
    const size_t nh    = q_shap[1];
    const size_t d     = q_shap[2];
    const size_t sk    = k_shap[0];
    const size_t nh_kv = k_shap[1];
    const size_t dv    = v_shap[2];

    // cuBLAS batched GEMM path for larger problems; the overhead of cuBLAS
    // setup is amortised once the matrices are big enough for tensor-cores.
    bool use_cublas = (d >= 32) && (sk >= 16);

    if (use_cublas) {
        switch (dtype) {
        case LLAISYS_DTYPE_F16:
            return self_attention_cublas<__half, CUDA_R_16F>(
                reinterpret_cast<__half *>(attn),
                reinterpret_cast<const __half *>(q),
                reinterpret_cast<const __half *>(k),
                reinterpret_cast<const __half *>(v),
                scale, seq, nh, d, sk, nh_kv, dv);
        case LLAISYS_DTYPE_F32:
            return self_attention_cublas<float, CUDA_R_32F>(
                reinterpret_cast<float *>(attn),
                reinterpret_cast<const float *>(q),
                reinterpret_cast<const float *>(k),
                reinterpret_cast<const float *>(v),
                scale, seq, nh, d, sk, nh_kv, dv);
        case LLAISYS_DTYPE_BF16:
            return self_attention_cublas<__nv_bfloat16, CUDA_R_16BF>(
                reinterpret_cast<__nv_bfloat16 *>(attn),
                reinterpret_cast<const __nv_bfloat16 *>(q),
                reinterpret_cast<const __nv_bfloat16 *>(k),
                reinterpret_cast<const __nv_bfloat16 *>(v),
                scale, seq, nh, d, sk, nh_kv, dv);
        default:
            throw std::runtime_error("self_attention_cuda: unsupported dtype");
        }
    }

    // small-size fast path: fused online-softmax kernel
    if (d > MAX_HD || dv > MAX_HD)
        throw std::runtime_error("self_attention_cuda: head dim exceeds MAX_HD (256)");

    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return launch_self_attention(
            reinterpret_cast<__half *>(attn),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            scale, seq, nh, d, sk, nh_kv, dv);
    case LLAISYS_DTYPE_F32:
        return launch_self_attention(
            reinterpret_cast<float *>(attn),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, seq, nh, d, sk, nh_kv, dv);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention(
            reinterpret_cast<__nv_bfloat16 *>(attn),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            scale, seq, nh, d, sk, nh_kv, dv);
    default:
        throw std::runtime_error("self_attention_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
