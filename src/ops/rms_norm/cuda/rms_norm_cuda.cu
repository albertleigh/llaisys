//
// Created by ali on 2/22/26.
//

#include "rms_norm_cuda.cuh"

#include <array>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"

namespace {

// ── Warp-level reduce-sum ───────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ── Block-level reduce-sum (shared memory) ──────────────────────────────────
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // max 32 warps per block (1024 threads)
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// ── from_float helper ───────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float           from_float<float>(float v)           { return v; }
template <> __device__ __forceinline__ __half          from_float<__half>(float v)          { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16   from_float<__nv_bfloat16>(float v)  { return __float2bfloat16(v); }

// ── Vectorised kernel (128-bit loads / stores) ──────────────────────────────
// VEC = elements per 128-bit chunk: 4 for float, 8 for half/bf16.
// One block per row.  Each thread processes up to MAX_VECS vector chunks,
// caching the input as float in registers so global memory is read only once.
template <typename T>
__global__ void rms_norm_kernel_vec(T *__restrict__ out,
                                    const T *__restrict__ in,
                                    const T *__restrict__ weight,
                                    float eps, unsigned int N) {
    constexpr int VEC      = 16 / static_cast<int>(sizeof(T));
    constexpr int MAX_VECS = 4;                         // ≤ 16 regs (f32) / 32 regs (f16)

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int vec_n = N / VEC;                 // # 128-bit chunks per row

    const float4 *__restrict__ x_v = reinterpret_cast<const float4 *>(in + row * N);
    const float4 *__restrict__ w_v = reinterpret_cast<const float4 *>(weight);
    float4       *__restrict__ y_v = reinterpret_cast<float4 *>(out + row * N);

    // ── pass 1: vectorised load → register cache + partial sum-of-squares ───
    float cache[MAX_VECS * VEC];
    float ss = 0.0f;

#pragma unroll
    for (int v = 0; v < MAX_VECS; ++v) {
        const unsigned int vi = tid + static_cast<unsigned int>(v) * blockDim.x;
        if (vi < vec_n) {
            float4 xchunk = __ldg(&x_v[vi]);
            const T *xe = reinterpret_cast<const T *>(&xchunk);
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float val = llaisys::utils::cuda::to_float(xe[k]);
                cache[v * VEC + k] = val;
                ss += val * val;
            }
        }
    }

    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(ss / static_cast<float>(N) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // ── pass 2: normalise, scale, vectorised store ──────────────────────────
#pragma unroll
    for (int v = 0; v < MAX_VECS; ++v) {
        const unsigned int vi = tid + static_cast<unsigned int>(v) * blockDim.x;
        if (vi < vec_n) {
            float4 wchunk = __ldg(&w_v[vi]);
            const T *we = reinterpret_cast<const T *>(&wchunk);

            float4 result;
            T *re = reinterpret_cast<T *>(&result);
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float w = llaisys::utils::cuda::to_float(we[k]);
                re[k] = from_float<T>(cache[v * VEC + k] * w * inv_rms);
            }
            y_v[vi] = result;
        }
    }
}

// ── Scalar fallback (small / unaligned N) ───────────────────────────────────
constexpr int kMaxElemsPerThread = 16;

template <typename T>
__global__ void rms_norm_kernel_scalar(T *__restrict__ out,
                                       const T *__restrict__ in,
                                       const T *__restrict__ weight,
                                       float eps, unsigned int N) {
    const unsigned int row = blockIdx.x;
    const T *__restrict__ x = in  + row * N;
    T       *__restrict__ y = out + row * N;

    float cache[kMaxElemsPerThread];
    float ss = 0.0f;

#pragma unroll
    for (int e = 0; e < kMaxElemsPerThread; ++e) {
        const unsigned int j = threadIdx.x + static_cast<unsigned int>(e) * blockDim.x;
        if (j < N) {
            float v = llaisys::utils::cuda::to_float(x[j]);
            cache[e] = v;
            ss += v * v;
        }
    }

    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / static_cast<float>(N) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

#pragma unroll
    for (int e = 0; e < kMaxElemsPerThread; ++e) {
        const unsigned int j = threadIdx.x + static_cast<unsigned int>(e) * blockDim.x;
        if (j < N) {
            float w = llaisys::utils::cuda::to_float(weight[j]);
            y[j] = from_float<T>(cache[e] * w * inv_rms);
        }
    }
}

// ── Launch helper ───────────────────────────────────────────────────────────
template <typename T>
void launch_rms_norm(T *out, const T *in, const T *weight,
                     float eps, size_t M, size_t N, cudaStream_t stream) {
    constexpr int VEC = 16 / static_cast<int>(sizeof(T)); // 4 (f32) or 8 (f16/bf16)

    if (N >= static_cast<size_t>(VEC) && (N % VEC) == 0) {
        // ── vectorised path ─────────────────────────────────────────────────
        unsigned int vec_n = static_cast<unsigned int>(N / VEC);
        int block = static_cast<int>(vec_n < 1024u ? vec_n : 1024u);
        block = (block + 31) & ~31;
        if (block < 32) block = 32;

        rms_norm_kernel_vec<<<static_cast<int>(M), block, 0, stream>>>(
            out, in, weight, eps, static_cast<unsigned int>(N));
    } else {
        // ── scalar fallback ─────────────────────────────────────────────────
        int block = static_cast<int>(N < 1024 ? N : 1024);
        block = (block + 31) & ~31;
        if (block < 32) block = 32;

        rms_norm_kernel_scalar<<<static_cast<int>(M), block, 0, stream>>>(
            out, in, weight, eps, static_cast<unsigned int>(N));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, std::array<size_t, 2> dims, llaisysStream_t stream) {
    size_t M = dims[0]; // number of rows (tokens / batch)
    size_t N = dims[1]; // row width (hidden dim)
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_rms_norm(reinterpret_cast<float *>(out),
                               reinterpret_cast<const float *>(in),
                               reinterpret_cast<const float *>(weight),
                               eps, M, N, s);
    case LLAISYS_DTYPE_F16:
        return launch_rms_norm(reinterpret_cast<__half *>(out),
                               reinterpret_cast<const __half *>(in),
                               reinterpret_cast<const __half *>(weight),
                               eps, M, N, s);
    case LLAISYS_DTYPE_BF16:
        return launch_rms_norm(reinterpret_cast<__nv_bfloat16 *>(out),
                               reinterpret_cast<const __nv_bfloat16 *>(in),
                               reinterpret_cast<const __nv_bfloat16 *>(weight),
                               eps, M, N, s);
    default:
        throw std::runtime_error("rms_norm_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
