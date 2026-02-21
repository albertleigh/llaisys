//
// Created by ali on 2/20/26.
//

#include "argmax_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

namespace {

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

// ── block-level argmax: warp reduce → shared memory → final warp reduce ─────
template <typename T>
__device__ __forceinline__ void block_argmax_reduce(T &out_val, size_t &out_idx,
                                                    T local_val, size_t local_idx,
                                                    unsigned int warp_size,
                                                    T *smem_val, size_t *smem_idx) {
    const unsigned int tid = threadIdx.x;
    const unsigned int num_warps = (blockDim.x + warp_size - 1) / warp_size;

    T reduced_val;
    size_t reduced_idx;
    warp_argmax_reduce(reduced_val, reduced_idx, local_val, local_idx, warp_size);

    if (tid % warp_size == 0) {
        smem_val[tid / warp_size] = reduced_val;
        smem_idx[tid / warp_size] = reduced_idx;
    }

    __syncthreads();

    if (tid < warp_size) {
        bool valid = tid < num_warps;
        T block_val = valid ? smem_val[tid] : T(-__FLT_MAX__);
        size_t block_idx = valid ? smem_idx[tid] : ~size_t(0);
        warp_argmax_reduce(out_val, out_idx, block_val, block_idx, warp_size);
    }
}

// ── single-block kernel: processes entire input, no intermediate storage ────
template <typename T>
__global__ void argmax_single_block(size_t *out_idx, T *out_val,
                                    const T *vals, size_t numel,
                                    unsigned int warp_size) {
    extern __shared__ std::byte shared_mem[];
    const unsigned int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    size_t *sidx = reinterpret_cast<size_t *>(shared_mem);
    T *sval = reinterpret_cast<T *>(shared_mem + num_warps * sizeof(size_t));

    const size_t tid = threadIdx.x;

    T local_max_value = T(-__FLT_MAX__);
    size_t local_max_idx = ~size_t(0);

    for (size_t i = tid; i < numel; i += blockDim.x) {
        T v = vals[i];
        if (v > local_max_value || (v == local_max_value && i < local_max_idx)) {
            local_max_value = v;
            local_max_idx = i;
        }
    }

    T result_val;
    size_t result_idx;
    block_argmax_reduce(result_val, result_idx, local_max_value, local_max_idx, warp_size, sval, sidx);

    if (tid == 0) {
        *out_idx = result_idx;
        *out_val = result_val;
    }
}

// ── single-pass multi-block kernel (threadfence / last-block reduction) ─────
// Each block reduces its portion, writes to intermediate arrays, then uses
// __threadfence() + atomicInc to detect the last block.  The last block
// performs the final cross-block reduction — all in ONE kernel launch.
// This avoids the second kernel launch and its associated overhead.
template <typename T>
__global__ void argmax_lastblock(size_t *out_idx, T *out_val,
                                 const T *vals, size_t numel,
                                 size_t *inter_idx, T *inter_val,
                                 unsigned int *block_counter,
                                 unsigned int warp_size) {
    extern __shared__ std::byte shared_mem[];
    const unsigned int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    size_t *sidx = reinterpret_cast<size_t *>(shared_mem);
    T *sval = reinterpret_cast<T *>(shared_mem + num_warps * sizeof(size_t));
    // last bool lives right after the smem arrays
    bool *is_last_block = reinterpret_cast<bool *>(shared_mem + num_warps * (sizeof(size_t) + sizeof(T)));

    const size_t tid = threadIdx.x;
    const size_t global_idx = blockIdx.x * blockDim.x + tid;

    // ── Phase 1: each block reduces its grid-strided chunk ──────────────────
    T local_max_value = T(-__FLT_MAX__);
    size_t local_max_idx = ~size_t(0);

    for (size_t i = global_idx; i < numel; i += blockDim.x * gridDim.x) {
        T v = vals[i];
        if (v > local_max_value || (v == local_max_value && i < local_max_idx)) {
            local_max_value = v;
            local_max_idx = i;
        }
    }

    T block_val;
    size_t block_idx;
    block_argmax_reduce(block_val, block_idx, local_max_value, local_max_idx, warp_size, sval, sidx);

    if (tid == 0) {
        inter_val[blockIdx.x] = block_val;
        inter_idx[blockIdx.x] = block_idx;

        // Ensure writes are globally visible before signaling
        __threadfence();

        // atomicInc wraps to 0 when value == gridDim.x - 1
        unsigned int ticket = atomicInc(block_counter, gridDim.x - 1);
        *is_last_block = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    // ── Phase 2: last block reduces all intermediate results ────────────────
    if (*is_last_block) {
        T final_val = T(-__FLT_MAX__);
        size_t final_idx = ~size_t(0);

        for (size_t i = tid; i < gridDim.x; i += blockDim.x) {
            T v = inter_val[i];
            size_t orig_idx = inter_idx[i];
            if (v > final_val || (v == final_val && orig_idx < final_idx)) {
                final_val = v;
                final_idx = orig_idx;
            }
        }

        // Reuse shared memory for the final reduction
        // Need a syncthreads first since smem was used in phase 1
        __syncthreads();

        T result_val;
        size_t result_idx;
        block_argmax_reduce(result_val, result_idx, final_val, final_idx, warp_size, sval, sidx);

        if (tid == 0) {
            *out_idx = result_idx;
            *out_val = result_val;
        }
    }
}

// ── typed single-block launcher ─────────────────────────────────────────────
template <typename T>
void launch_argmax_single(size_t *out_idx, T *out_val, const T *vals,
                          size_t numel, unsigned int warp_size) {
    // Query the runtime for the max block size this kernel can actually launch
    int min_grid_size, max_block_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &max_block_size, argmax_single_block<T>, 0, 0));
    // (void)min_grid_size;

    unsigned int block_size = static_cast<unsigned int>(max_block_size);
    // For small inputs, round up to warp boundary
    if (numel < block_size) {
        block_size = ((static_cast<unsigned int>(numel) + warp_size - 1) / warp_size) * warp_size;
        if (block_size > static_cast<unsigned int>(max_block_size))
            block_size = static_cast<unsigned int>(max_block_size);
    }

    size_t smem_size = (block_size + warp_size - 1) / warp_size * (sizeof(T) + sizeof(size_t));
    argmax_single_block<<<1, block_size, smem_size>>>(out_idx, out_val, vals, numel, warp_size);
    CUDA_CHECK(cudaGetLastError());
}

// ── persistent intermediate buffer for multi-block path ─────────────────────
// Bounded in size (grid_size entries, typically < 200), reused across calls.
// The atomicInc(counter, gridDim.x - 1) in the kernel wraps the counter back
// to 0 after all blocks complete, so it self-resets between invocations.
// We only need cudaMemset once at allocation time.
struct InnerBuffer {
    std::byte *buf = nullptr;
    size_t capacity = 0; // max grid_size this buffer can handle

    // Layout: [size_t inter_idx[cap]] [float inter_val[cap]] [uint counter]
    // sizeof(float) >= sizeof(__half), sizeof(__nv_bfloat16)
    // float is the largest size supported.
    static size_t bytes_for(size_t cap) {
        return cap * sizeof(size_t) + cap * sizeof(float) + sizeof(unsigned int);
    }

    void ensure(size_t grid_size) {
        if (grid_size <= capacity) {
            return;
        }
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
        capacity = grid_size * 2; // 2× headroom
        CUDA_CHECK(cudaMalloc(&buf, bytes_for(capacity)));
        // Zero the counter (and everything else for good measure)
        CUDA_CHECK(cudaMemset(buf, 0, bytes_for(capacity)));
    }

    template <typename T>
    void get_ptrs(size_t grid_size, size_t *&idx, T *&val, unsigned int *&counter) {
        idx = reinterpret_cast<size_t *>(buf);
        val = reinterpret_cast<T *>(buf + capacity * sizeof(size_t));
        counter = reinterpret_cast<unsigned int *>(
            buf + capacity * sizeof(size_t) + capacity * sizeof(float));
    }
};

static InnerBuffer &get_inner_buffer() {
    static auto *b = new InnerBuffer(); // intentionally leaked — bounded, reusable
    return *b;
}

// ── typed multi-block launcher (single-pass, last-block reduction) ──────────
template <typename T>
void launch_argmax_multiblock(size_t *out_idx, T *out_val, const T *vals,
                              size_t numel, unsigned int warp_size,
                              int sm_count) {
    // Query the runtime for the max block size this kernel can actually launch.
    // This accounts for register pressure after device linking (-rdc=true),
    // which can inflate register counts beyond what ptxas reports at compile
    // time.  Avoids "too many resources requested for launch" errors.
    int min_grid_size, max_block_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &max_block_size, argmax_lastblock<T>, 0, 0));
    (void)min_grid_size;

    int block_size = max_block_size;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    // Cap grid to ~2 waves per SM — fewer blocks means less intermediate
    // data and a cheaper final reduction in the last block
    int max_active_blocks = sm_count * 2;
    if (grid_size > max_active_blocks) {
        grid_size = max_active_blocks;
    }
    if (grid_size < 1) {
        grid_size = 1;
    }

    auto &ibuf = get_inner_buffer();
    ibuf.ensure(grid_size);

    size_t *inter_idx;
    T *inter_val;
    unsigned int *block_counter;
    ibuf.get_ptrs<T>(grid_size, inter_idx, inter_val, block_counter);

    unsigned int num_warps = (block_size + warp_size - 1) / warp_size;
    size_t smem_size = num_warps * (sizeof(T) + sizeof(size_t)) + sizeof(bool);

    argmax_lastblock<<<grid_size, block_size, smem_size>>>(out_idx, out_val, vals, numel,
                                                 inter_idx, inter_val,
                                                 block_counter, warp_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::cuda {
void argmax(std::byte *max_id, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    auto &dev = utils::cuda::get_device_info();
    dev.refresh();
    unsigned int warp_size = dev.warp_size;

    // ── Single-block path for small inputs ──────────────────────────────────
    // Each thread handles numel/block_size elements. With 1024 threads, up to
    // ~64K elements means ~64 iterations per thread — a good sweet spot before
    // multi-SM parallelism becomes worthwhile.
    constexpr size_t SINGLE_BLOCK_THRESHOLD = 1 << 16; // 64K elements
    if (numel <= SINGLE_BLOCK_THRESHOLD) {
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return launch_argmax_single(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<float *>(max_val),
                                        reinterpret_cast<const float *>(vals),
                                        numel, warp_size);
        case LLAISYS_DTYPE_F16:
            return launch_argmax_single(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<__half *>(max_val),
                                        reinterpret_cast<const __half *>(vals),
                                        numel, warp_size);
        case LLAISYS_DTYPE_BF16:
            return launch_argmax_single(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<__nv_bfloat16 *>(max_val),
                                        reinterpret_cast<const __nv_bfloat16 *>(vals),
                                        numel, warp_size);
        default:
            throw std::runtime_error("argmax_cuda: unsupported dtype");
        }
    }

    // ── Multi-block single-pass path for large inputs ───────────────────────
    // Block size is determined by cudaOccupancyMaxPotentialBlockSize so that
    // the launch respects actual register limits (which can increase beyond
    // compile-time ptxas estimates after device linking with -rdc=true).
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax_multiblock(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<float *>(max_val),
                                        reinterpret_cast<const float *>(vals),
                                        numel, warp_size, dev.sm_count);
    case LLAISYS_DTYPE_F16:
        return launch_argmax_multiblock(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<__half *>(max_val),
                                        reinterpret_cast<const __half *>(vals),
                                        numel, warp_size, dev.sm_count);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax_multiblock(reinterpret_cast<size_t *>(max_id),
                                        reinterpret_cast<__nv_bfloat16 *>(max_val),
                                        reinterpret_cast<const __nv_bfloat16 *>(vals),
                                        numel, warp_size, dev.sm_count);
    default:
        throw std::runtime_error("argmax_cuda: unsupported dtype");
    }
}
} // namespace llaisys::ops::cuda
