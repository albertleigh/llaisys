//
// Created by ali on 2/20/26.
//

#include "add_cuda.cuh"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

#include "../../../cuda_utils/check.cuh"
#include "../../../utils.hpp"

namespace {

// ── generic device add ──────────────────────────────────────────────────────
template <typename T>
__device__ T add_(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, __half>) {
        return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, __half2>) {
        return __hadd2(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat162>) {
        return __hadd2(a, b);
    } else if constexpr (std::is_same_v<T, float2>) {
        return make_float2(a.x + b.x, a.y + b.y);
    } else if constexpr (std::is_same_v<T, float3>) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    } else if constexpr (std::is_same_v<T, float4>) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    } else {
        return a + b;
    }
}

// ── single templated kernel ─────────────────────────────────────────────────
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = add_(a[idx], b[idx]);
    }
}

constexpr int BLOCK_SIZE = 256;

} // namespace

namespace llaisys::ops::cuda {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int grid = static_cast<int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<grid, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<grid, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            numel);
        break;
    default:
        throw std::runtime_error("add_cuda: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::cuda
