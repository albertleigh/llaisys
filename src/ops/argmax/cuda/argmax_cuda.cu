//
// Created by ali on 2/20/26.
//

#include "argmax_cuda.cuh"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

#include <bits/fs_fwd.h>

namespace {
using llaisys::utils::cuda::to_float;

// ── warp-level argmax reduction via shuffle ──────────────────────────────────
// Reduces (val, idx) pairs across a warp.  The lane with the largest value
// wins; ties are broken by smaller index (stable).
template <typename T>
__device__ __forceinline__ void warp_argmax_reduce(T &out_val, size_t &out_idx, T local_val, size_t local_idx,
                                                   unsigned int warp_size) {
#pragma unroll
    for (unsigned int offset = warp_size / 2; offset > 0; offset >>= 1) {
        T other_val = __shfl_down_sync(0xFFFFFFFF, local_val, offset);
        size_t other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other_val > local_val || (other_val == local_val && other_idx < local_idx)) {
            local_val = other_val;
            local_idx = other_idx;
        }
    }

    out_val = local_val;
    out_idx = local_idx;
}

// ── first pass: per-block argmax with warp-shuffle reduction ────────────────
// Each block reduces its chunk to a single (index, value) pair.
// Uses dynamic shared memory: float smem_val[] followed by size_t smem_idx[].
template <typename T>
__global__ void argmax_shuffle_first_pass(size_t *intermediate_idx, T *intermediate_val,
                                          const T *vals, size_t numel,
                                          unsigned int warp_size) {
    extern __shared__ std::byte shared_mem[];
    const unsigned int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    size_t *sidx = reinterpret_cast<size_t *>(shared_mem);
    T *smem = reinterpret_cast<T *>(shared_mem + num_warps * sizeof(size_t));

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + tid;

    T local_max_value = vals[idx];
    size_t local_max_idx = idx;

    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        if (vals[i] > local_max_value || (vals[i] == local_max_value && i < local_max_idx)) {
            local_max_value = vals[i];
            local_max_idx = i;
        }
    }

    T reduced_max_value = vals[idx];
    size_t reduced_max_idx = idx;

    warp_argmax_reduce(reduced_max_value, reduced_max_idx, local_max_value, local_max_idx, warp_size);

    if (tid % warp_size == 0) {
        smem[tid / warp_size] = reduced_max_value;
        sidx[tid / warp_size] = reduced_max_idx;
    }

    __syncthreads();

    if (tid < warp_size) {
        bool tid_of_valid_smem_value = tid < (blockDim.x + warp_size - 1) / warp_size;
        T block_max_value = tid_of_valid_smem_value ? smem[tid] : T(-__FLT_MAX__);
        size_t block_max_idx = tid_of_valid_smem_value ? sidx[tid] : ~size_t(0);
        // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is
        // 1024. 1024 / 32 = 32, which means at most 32 warps in a block. therefore, we can directly use warp_reduce here to
        // reduce within a warp.
        warp_argmax_reduce(reduced_max_value, reduced_max_idx, block_max_value, block_max_idx, warp_size);
        if (tid == 0) {
            intermediate_idx[blockIdx.x] = reduced_max_idx;
            intermediate_val[blockIdx.x] = reduced_max_value;
        }
    }
}

// ── second pass: reduce block winners (single block) ────────────────────────
template <typename T>
__global__ void argmax_shuffle_second_pass(size_t *out_idx, T *out_val,
                                           const size_t *intermediate_idx,
                                           const T *intermediate_val,
                                           const size_t num_blocks,
                                           unsigned int warp_size) {
    extern __shared__ std::byte shared_mem[];
    const unsigned int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    size_t *sidx = reinterpret_cast<size_t *>(shared_mem);
    T *smem = reinterpret_cast<T *>(shared_mem + num_warps * sizeof(size_t));

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + tid;

    T local_max_value = -__FLT_MAX__;
    size_t local_max_idx = ~size_t(0);

    for (size_t i = idx; i < num_blocks; i += blockDim.x) {
        if (intermediate_val[i] > local_max_value || (intermediate_val[i] == local_max_value && i < local_max_idx)) {
            local_max_value = intermediate_val[i];
            local_max_idx = i;
        }
    }

    T reduced_max_value;
    size_t reduced_max_idx;

    warp_argmax_reduce(reduced_max_value, reduced_max_idx, local_max_value, local_max_idx, warp_size);

    if (tid % warp_size == 0) {
        smem[tid / warp_size] = reduced_max_value;
        sidx[tid / warp_size] = reduced_max_idx;
    }

    __syncthreads();

    if (tid < warp_size) {
        bool tid_of_valid_smem_value = tid < (blockDim.x + warp_size - 1) / warp_size;
        T block_max_value = tid_of_valid_smem_value ? smem[tid] : T(-__FLT_MAX__);
        size_t block_max_idx = tid_of_valid_smem_value ? sidx[tid] : ~size_t(0);
        // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is
        // 1024. 1024 / 32 = 32, which means at most 32 warps in a block. therefore, we can directly use warp_reduce here to
        // reduce within a warp.
        warp_argmax_reduce(reduced_max_value, reduced_max_idx, block_max_value, block_max_idx, warp_size);
        if (tid == 0) {
            *out_idx = reduced_max_idx;
            *out_val = reduced_max_value;
        }
    }
}

// ── typed launcher ──────────────────────────────────────────────────────────
template <typename T>
void launch_argmax_shuffle(size_t *out_idx, T *out_val, const T *vals, size_t numel,
                           dim3 grid, dim3 block, unsigned int warp_size) {
    T *d_intermediate_val;
    size_t *d_intermediate_idx;
    CUDA_CHECK(cudaMalloc(&d_intermediate_val, grid.x * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_intermediate_idx, grid.x * sizeof(size_t)));

    // First pass: reduce within blocks
    size_t smem_size = (block.x + warp_size - 1) / warp_size * (sizeof(T) + sizeof(size_t));
    argmax_shuffle_first_pass<<<grid, block, smem_size>>>(d_intermediate_idx, d_intermediate_val, vals, numel, warp_size);
    CUDA_CHECK(cudaGetLastError());

    // Second pass: reduce block results
    dim3 grid2(1);
    // When there were not that many reduced intermediate_val of size grid, we need not to lauch that many block.x
    dim3 block2(min(grid.x, block.x));
    size_t smem_size2 = (block2.x + warp_size - 1) / warp_size * (sizeof(T) + sizeof(size_t));
    argmax_shuffle_second_pass<<<grid2, block2, smem_size2>>>(out_idx, out_val, d_intermediate_idx, d_intermediate_val, grid.x, warp_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_intermediate_val));
    CUDA_CHECK(cudaFree(d_intermediate_idx));
}

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
template <typename T>
__global__ void launch_argmax_cooperatively(size_t *out_idx, T *out_val, const T *vals, size_t numel, unsigned int wrap_size) {
    // (block.x + wrap_size - 1) / wrap_size * (dsize(dtype) + sizeof(size_t))
    extern __shared__ std::byte shared_mem[];
    const unsigned int num_warps = (block.x + wrap_size - 1) / wrap_size;
    size_t *coop_sidx = reinterpret_cast<size_t *>(shared_mem);
    T *coop_smem = reinterpret_cast<T *>(shared_mem + num_warps * sizeof(size_t));

    cg::grid_group grid = cg::this_grid();
    // Replace with __syncthreads
    cg::thread_block block = cg::this_thread_block();
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    T local_max_value = vals[idx];
    size_t local_max_idx = idx;

    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        if (vals[i] > local_max_value || (vals[i] == local_max_value && i < local_max_idx)) {
            local_max_value = vals[i];
            local_max_idx = i;
        }
    }

    T reduced_max_value = vals[idx];
    size_t reduced_max_idx = idx;

    warp_argmax_reduce(reduced_max_value, reduced_max_idx, local_max_value, local_max_idx, warp_size);

    if (tid % warp_size == 0) {
        coop_smem[tid / warp_size] = reduced_max_value;
        coop_sidx[tid / warp_size] = reduced_max_idx;
    }

    block.sync();
    // __syncthreads();  // equivalent to block.sync()

    if (tid < warp_size) {
        bool tid_of_valid_smem_value = tid < (blockDim.x + warp_size - 1) / warp_size;
        T block_max_value = tid_of_valid_smem_value ? coop_smem[tid] : T(-__FLT_MAX__);
        size_t block_max_idx = tid_of_valid_smem_value ? coop_sidx[tid] : ~size_t(0);
        // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is
        // 1024. 1024 / 32 = 32, which means at most 32 warps in a block. therefore, we can directly use warp_reduce here to
        // reduce within a warp.
        warp_argmax_reduce(reduced_max_value, reduced_max_idx, block_max_value, block_max_idx, warp_size);
        if (tid == 0) {
            out_idx[blockIdx.x] = reduced_max_idx;
            out_val[blockIdx.x] = reduced_max_value;
        }
    }

    // Global synchronization
    grid.sync();

    if (blockIdx.x == 0) {
        T final_max_value = -__FLT_MAX__;
        size_t final_max_idx = ~size_t(0);
        for (size_t i = tid; i < gridDim.x; i += blockDim.x) {
            final_sum += output[i];
            if (out_val[i] > final_max_value || (out_val[i] == final_max_value && i < final_max_idx)) {
                local_max_value = out_val[i];
                local_max_idx = i;
            }
        }
        warp_argmax_reduce(reduced_max_value, reduced_max_idx, final_max_value, final_max_idx, warp_size);

        if (tid % wrap_size == 0) {
            // coop_sem should be larger than wrap_size * sizeof(T)
            coop_smem[tid / wrap_size] = reduced_max_value;
            coop_sidx[tid / warp_size] = reduced_max_idx;
        }

        block.sync();
        // __syncthreads();

        if (tid < wrap_size) {
            bool tid_of_valid_smem_value = tid < (blockDim.x + warp_size - 1) / warp_size;
            T block_max_value = tid_of_valid_smem_value ? coop_smem[tid] : T(-__FLT_MAX__);
            size_t block_max_idx = tid_of_valid_smem_value ? coop_sidx[tid] : ~size_t(0);
            warp_argmax_reduce(reduced_max_value, reduced_max_idx, final_max_value, final_max_idx, warp_size);
            if (tid == 0) {
                out_val[0] = reduced_max_value;
                out_idx[0] = reduced_max_idx;
            }
        }
    }
}
#endif
} // namespace

namespace llaisys::ops::cuda {
void argmax(std::byte *max_id, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    // Query warp size from device
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    unsigned int warp_size = static_cast<unsigned int>(props.warpSize);

    // Choose block/grid dimensions dynamically
    int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    // Cap grid to ~4 waves of work per SM instead of arbitrary 1024
    int max_active_blocks = props.multiProcessorCount * 4;
    if (grid_size > max_active_blocks) {
        grid_size = max_active_blocks;
    }

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
    {
        dim3 grid(grid_size);
        dim3 block(block_size);

        int grid_size_v9 = 0;
        int block_size = block_size;
        size_t smem_size = (block.x + wrap_size - 1) / wrap_size * (dsize(dtype) + sizeof(size_t));

        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size_v9, launch_argmax_cooperatively<float>, block_size, smem_size));
            break;
        case LLAISYS_DTYPE_F16:
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size_v9, launch_argmax_cooperatively<__half>, block_size, smem_size));
            break;
        case LLAISYS_DTYPE_BF16:
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size_v9, launch_argmax_cooperatively<__nv_bfloat16>, block_size, smem_size));
            break;
        default:
            throw std::runtime_error("argmax_cuda: unsupported dtype");
        }

        grid_size_v9 *= props.multiProcessorCount;
        grid_size_v9 = min(grid_size_v9, static_cast<int>((n + block_size - 1) / block_size));
        grid_size_v9 = max(grid_size_v9, 1);

        int can_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&can_launch, cudaDevAttrCooperativeLaunch, 0));

        if (can_launch) {
            size_t *d_out_idx;
            T *d_out_val;
            CUDA_CHECK(cudaMalloc(&d_out_idx, grid_size_v9 * sizeof(size_t)));
            CUDA_CHECK(cudaMalloc(&d_out_val, grid_size_v9 * sizeof(T)));

            void *kernelArgs[] = {&d_out_idx, &d_out_val, &vals, &numel, &wrap_size};

            switch (dtype) {
            case LLAISYS_DTYPE_F32:
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *)launch_argmax_cooperatively<float>, grid_size_v9, block, kernelArgs, smem_size, 0));
                break;
            case LLAISYS_DTYPE_F16:
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *)launch_argmax_cooperatively<__half>, grid_size_v9, block, kernelArgs, smem_size, 0));
                break;
            case LLAISYS_DTYPE_BF16:
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *)launch_argmax_cooperatively<__nv_bfloat16>, grid_size_v9, block, kernelArgs, smem_size, 0));
                break;
            default:
                throw std::runtime_error("argmax_cuda: unsupported dtype");
            }

            CUDA_CHECK(cudaMemcpy(max_id, d_out_idx, sizeof(size_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(max_val, d_out_val, utils::dsize(T), cudaMemcpyDeviceToHost));

            CUDA_FREE(d_out_idx)
            CUDA_FREE(d_out_val)
        } else {
            std::cerr << "CUDA device does not support cooperative launch, fall back to shuffle" << '\n';
            // don't exit, fall back to shuffle.
        }
    }
}
#endif

{
    dim3 grid(grid_size);
    dim3 block(block_size);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax_shuffle(reinterpret_cast<size_t *>(max_id),
                                     reinterpret_cast<float *>(max_val),
                                     reinterpret_cast<const float *>(vals),
                                     numel, grid, block, warp_size);
    case LLAISYS_DTYPE_F16:
        return launch_argmax_shuffle(reinterpret_cast<size_t *>(max_id),
                                     reinterpret_cast<__half *>(max_val),
                                     reinterpret_cast<const __half *>(vals),
                                     numel, grid, block, warp_size);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax_shuffle(reinterpret_cast<size_t *>(max_id),
                                     reinterpret_cast<__nv_bfloat16 *>(max_val),
                                     reinterpret_cast<const __nv_bfloat16 *>(vals),
                                     numel, grid, block, warp_size);
    default:
        throw std::runtime_error("argmax_cuda: unsupported dtype");
    }
}
}
} // namespace llaisys::ops::cuda
