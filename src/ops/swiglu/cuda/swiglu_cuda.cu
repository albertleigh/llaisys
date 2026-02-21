//
// Created by ali on 2/22/26.
//

#include "swiglu_cuda.cuh"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"

namespace {

// ── SwiGLU kernel ───────────────────────────────────────────────────────────
// out[i] = up[i] * swish(gate[i])   where swish(x) = x / (1 + exp(-x))
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float gate_val = llaisys::utils::cuda::to_float(gate[i]);
        float up_val   = llaisys::utils::cuda::to_float(up[i]);
        float swish    = gate_val / (1.0f + expf(-gate_val));
        float res      = up_val * swish;

        if constexpr (std::is_same_v<T, float>) {
            out[i] = res;
        } else if constexpr (std::is_same_v<T, __half>) {
            out[i] = __float2half(res);
        } else { // __nv_bfloat16
            out[i] = __float2bfloat16(res);
        }
    }
}

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up, size_t numel) {
    int block = 256;
    int grid  = static_cast<int>((numel + block - 1) / block);
    if (grid > 65535) grid = 65535;

    swiglu_kernel<<<grid, block>>>(out, gate, up, numel);
    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu(reinterpret_cast<float *>(out),
                             reinterpret_cast<const float *>(gate),
                             reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu(reinterpret_cast<__half *>(out),
                             reinterpret_cast<const __half *>(gate),
                             reinterpret_cast<const __half *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu(reinterpret_cast<__nv_bfloat16 *>(out),
                             reinterpret_cast<const __nv_bfloat16 *>(gate),
                             reinterpret_cast<const __nv_bfloat16 *>(up), numel);
    default:
        throw std::runtime_error("swiglu_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
