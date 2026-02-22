//
// Created by ali on 2/22/26.
//

#include "rope_cuda.cuh"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"

namespace {

// ── Optimized RoPE CUDA kernel ──────────────────────────────────────────────
//
// Each thread handles one (s, j) pair and loops over all heads.
// This amortizes the trig computation across num_heads.
//
// Key optimisation vs the naïve one-thread-per-(s,h,j) approach:
//   trig is computed once per (s,j), reused across all heads    (num_heads×)
//
// Input layout: [seq_len, num_heads, head_dim]  (contiguous row-major)
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            float theta,
                            size_t seq_len, size_t num_heads,
                            size_t head_dim, size_t half_dim) {
    const size_t head_stride = num_heads * head_dim;
    const size_t total = seq_len * half_dim;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        const size_t j = idx % half_dim;
        const size_t s = idx / half_dim;

        // denom = θ^(2j/d)  —  matches PyTorch / CPU reference exactly
        const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
        const float denom = powf(theta, exponent);
        const float angle = static_cast<float>(pos_ids[s]) / denom;

        float cos_val, sin_val;
        sincosf(angle, &sin_val, &cos_val);

        // Apply the same rotation to every head
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t base = s * head_stride + h * head_dim;

            const float a = llaisys::utils::cuda::to_float(in[base + j]);
            const float b = llaisys::utils::cuda::to_float(in[base + j + half_dim]);

            const float a_out = a * cos_val - b * sin_val;
            const float b_out = b * cos_val + a * sin_val;

            if constexpr (std::is_same_v<T, float>) {
                out[base + j] = a_out;
                out[base + j + half_dim] = b_out;
            } else if constexpr (std::is_same_v<T, __half>) {
                out[base + j] = __float2half(a_out);
                out[base + j + half_dim] = __float2half(b_out);
            } else { // __nv_bfloat16
                out[base + j] = __float2bfloat16(a_out);
                out[base + j + half_dim] = __float2bfloat16(b_out);
            }
        }
    }
}

template <typename T>
void launch_rope(T *out, const T *in, const int64_t *pos_ids, float theta,
                 size_t seq_len, size_t num_heads, size_t head_dim, cudaStream_t stream) {
    const size_t half_dim = head_dim / 2;
    const size_t total = seq_len * half_dim;

    constexpr int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > 65535) {
        grid = 65535;
    }

    rope_kernel<<<grid, block, 0, stream>>>(out, in, pos_ids,
                                 theta,
                                 seq_len, num_heads, head_dim, half_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t dtype, const std::vector<size_t> &dims, llaisysStream_t stream) {
    const size_t seq_len = dims[0];
    const size_t num_heads = dims[1];
    const size_t head_dim = dims[2];

    const auto *pos = reinterpret_cast<const int64_t *>(pos_ids);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_rope(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(in),
                           pos, theta, seq_len, num_heads, head_dim, s);
    case LLAISYS_DTYPE_F16:
        return launch_rope(reinterpret_cast<__half *>(out),
                           reinterpret_cast<const __half *>(in),
                           pos, theta, seq_len, num_heads, head_dim, s);
    case LLAISYS_DTYPE_BF16:
        return launch_rope(reinterpret_cast<__nv_bfloat16 *>(out),
                           reinterpret_cast<const __nv_bfloat16 *>(in),
                           pos, theta, seq_len, num_heads, head_dim, s);
    default:
        throw std::runtime_error("rope_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
