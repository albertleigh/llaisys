//
// Created by ali on 2/21/26.
//

#include "embedding_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

namespace {

// ── embedding lookup kernel ─────────────────────────────────────────────────
// Each row in the output corresponds to weight[index[row]].
// Grid is 2D: blockIdx.x covers the embedding dimension, blockIdx.y covers
// the batch (number of indices).  This gives fully coalesced global memory
// accesses along the embedding dimension.
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t size, size_t embedding_dim) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size || col >= embedding_dim) {
        return;
    }

    const int64_t vocab_idx = index[row];
    out[row * embedding_dim + col] = weight[vocab_idx * embedding_dim + col];
}

// ── typed launcher ──────────────────────────────────────────────────────────
template <typename T>
void launch_embedding(T *out, const int64_t *index, const T *weight,
                      size_t size, size_t embedding_dim) {
    // Use a 2D block: 256 threads along embedding dim, 1 along batch.
    // This maximises coalesced memory access for the copy.
    constexpr unsigned int BLOCK_X = 256;
    constexpr unsigned int BLOCK_Y = 1;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((static_cast<unsigned int>(embedding_dim) + BLOCK_X - 1) / BLOCK_X,
              (static_cast<unsigned int>(size) + BLOCK_Y - 1) / BLOCK_Y);

    embedding_kernel<<<grid, block>>>(out, index, weight, size, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t size, size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return launch_embedding(reinterpret_cast<__half *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const __half *>(weight),
                                size, embedding_dim);
    case LLAISYS_DTYPE_F32:
        return launch_embedding(reinterpret_cast<float *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const float *>(weight),
                                size, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding(reinterpret_cast<__nv_bfloat16 *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const __nv_bfloat16 *>(weight),
                                size, embedding_dim);
    default:
        throw std::runtime_error("embedding_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
