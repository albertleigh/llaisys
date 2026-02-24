//
// CUDA sampling: Temperature, Top-K, Top-P (nucleus), multinomial draw.
//
// Strategy
// --------
// Greedy path (temperature <= 0 or top_k == 1):
//   Single-block argmax kernel entirely on device.
//
// Sampling path:
//   1. GPU: convert logits to float + temperature scale  (element-wise)
//   2. GPU: softmax  (max-reduce → exp → sum-reduce → normalise)
//          All on-device — no host sync during softmax.
//   3. GPU: CUB radix sort (probabilities descending + indices)
//   4. D2H: copy sorted probabilities to host
//          (only top-K entries when top_k is set, full vector otherwise)
//   5. Host: Top-K cutoff → Top-P cumsum cutoff → multinomial draw
//   6. D2H: fetch the one original index of the sampled position
//

#include "sample_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#ifdef _MSC_VER
#include <cfloat>
#ifndef __FLT_MAX__
#define __FLT_MAX__ FLT_MAX
#endif
#endif

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

#include <random>
#include <vector>

namespace {

// ── Host RNG ────────────────────────────────────────────────────────────────
static std::mt19937 &host_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

// ── Cached host buffer (avoids heap alloc per call) ─────────────────────────
static std::vector<float> &host_probs_buf() {
    thread_local std::vector<float> buf;
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Greedy argmax — single-block, any vocab size
// ═══════════════════════════════════════════════════════════════════════════

template <typename T>
__device__ __forceinline__ void warp_argmax(T &val, size_t &idx, unsigned int ws) {
#pragma unroll
    for (unsigned int off = ws / 2; off > 0; off >>= 1) {
        T      ov = __shfl_down_sync(0xFFFFFFFF, val, off);
        size_t oi = __shfl_down_sync(0xFFFFFFFF, idx, off);
        if (ov > val || (ov == val && oi < idx)) { val = ov; idx = oi; }
    }
}

template <typename T>
__device__ __forceinline__ void block_argmax(T &val, size_t &idx,
                                             unsigned int ws, T *sv, size_t *si) {
    unsigned int tid = threadIdx.x;
    unsigned int nw  = (blockDim.x + ws - 1) / ws;
    warp_argmax(val, idx, ws);
    if (tid % ws == 0) { sv[tid / ws] = val; si[tid / ws] = idx; }
    __syncthreads();
    if (tid < ws) {
        bool ok = tid < nw;
        T      bv = ok ? sv[tid] : T(-__FLT_MAX__);
        size_t bi = ok ? si[tid] : ~size_t(0);
        warp_argmax(bv, bi, ws);
        val = bv; idx = bi;
    }
}

template <typename T>
__global__ void greedy_argmax_kernel(int64_t *out_idx,
                                     const T *vals, size_t numel,
                                     unsigned int ws) {
    extern __shared__ std::byte smem[];
    unsigned int nw = (blockDim.x + ws - 1) / ws;
    size_t *si = reinterpret_cast<size_t *>(smem);
    T      *sv = reinterpret_cast<T *>(smem + nw * sizeof(size_t));

    size_t tid = threadIdx.x;
    T      best_v = T(-__FLT_MAX__);
    size_t best_i = ~size_t(0);
    for (size_t i = tid; i < numel; i += blockDim.x) {
        T v = vals[i];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    block_argmax(best_v, best_i, ws, sv, si);
    if (tid == 0) *out_idx = static_cast<int64_t>(best_i);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Softmax helper kernels — read scalars from device memory (no host sync)
// ═══════════════════════════════════════════════════════════════════════════

// Convert any supported type to float + multiply by inv_temperature.
template <typename T>
__global__ void convert_and_scale(float *out, const T *in, size_t n, float inv_t) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = llaisys::utils::cuda::to_float(in[i]) * inv_t;
}

// Single-block max-reduce.
__global__ void reduce_max(float *out, const float *v, size_t n) {
    extern __shared__ float sd[];
    size_t tid = threadIdx.x;
    float mx = -__FLT_MAX__;
    for (size_t i = tid; i < n; i += blockDim.x) mx = fmaxf(mx, v[i]);
    sd[tid] = mx;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sd[tid] = fmaxf(sd[tid], sd[tid + s]);
        __syncthreads();
    }
    if (tid == 0) *out = sd[0];
}

// exp(v[i] − *d_max).  Reads max from device via shared-memory broadcast.
__global__ void exp_sub(float *v, size_t n, const float *d_max) {
    __shared__ float mx;
    if (threadIdx.x == 0) mx = *d_max;
    __syncthreads();
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = expf(v[i] - mx);
}

// Single-block sum-reduce.
__global__ void reduce_sum(float *out, const float *v, size_t n) {
    extern __shared__ float sd[];
    size_t tid = threadIdx.x;
    float s = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x) s += v[i];
    sd[tid] = s;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half > 0; half >>= 1) {
        if (tid < half) sd[tid] += sd[tid + half];
        __syncthreads();
    }
    if (tid == 0) *out = sd[0];
}

// v[i] *= (1 / *d_sum).  Reads sum from device via shared-memory broadcast.
__global__ void div_by_sum(float *v, size_t n, const float *d_sum) {
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float s = *d_sum;
        inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    }
    __syncthreads();
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv;
}

// Fill indices with [0, 1, 2, …, n-1].
__global__ void iota_kernel(int64_t *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<int64_t>(i);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Persistent workspace
// ═══════════════════════════════════════════════════════════════════════════
struct SampleWorkspace {
    float   *probs          = nullptr;  // [cap]  probabilities (sort input)
    float   *probs_sorted   = nullptr;  // [cap]  sorted probabilities (sort output)
    float   *scalar         = nullptr;  // single float (max or sum)
    int64_t *d_out          = nullptr;  // device result for greedy path
    int64_t *indices        = nullptr;  // [cap]  original-index permutation (sort input)
    int64_t *indices_sorted = nullptr;  // [cap]  sorted indices (sort output)
    void    *cub_temp       = nullptr;  // CUB temp storage
    size_t   cub_temp_bytes = 0;
    size_t   cap            = 0;

    void ensure(size_t n) {
        if (n <= cap) return;
        if (probs)          cudaFree(probs);
        if (probs_sorted)   cudaFree(probs_sorted);
        if (scalar)         cudaFree(scalar);
        if (d_out)          cudaFree(d_out);
        if (indices)        cudaFree(indices);
        if (indices_sorted) cudaFree(indices_sorted);
        if (cub_temp)       cudaFree(cub_temp);
        cap = n;
        CUDA_CHECK(cudaMalloc(&probs,          cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&probs_sorted,   cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&scalar,         sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out,          sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&indices,        cap * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&indices_sorted, cap * sizeof(int64_t)));

        // Query CUB for required temp storage size, then allocate
        cub_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, cub_temp_bytes,
            probs, probs_sorted,
            indices, indices_sorted,
            static_cast<int>(cap));
        CUDA_CHECK(cudaMalloc(&cub_temp, cub_temp_bytes));
    }
};

static SampleWorkspace &ws() {
    static auto *w = new SampleWorkspace();   // intentionally leaked
    return *w;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Typed launcher
// ═══════════════════════════════════════════════════════════════════════════
template <typename T>
int64_t launch_sample(const T *logits, size_t vocab_size,
                      float temperature, int top_k, float top_p,
                      unsigned int warp_size, cudaStream_t stream) {
    auto &w = ws();
    w.ensure(vocab_size);

    // ── Greedy fast-path ────────────────────────────────────────────────────
    if (temperature <= 0.0f || top_k == 1) {
        int mg, mb;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &mg, &mb, greedy_argmax_kernel<T>, 0, 0));
        unsigned int bs = static_cast<unsigned int>(mb);
        if (vocab_size < bs) {
            bs = ((static_cast<unsigned int>(vocab_size) + warp_size - 1)
                  / warp_size) * warp_size;
            if (bs > static_cast<unsigned int>(mb)) bs = static_cast<unsigned int>(mb);
        }
        size_t smem = (bs + warp_size - 1) / warp_size
                      * (sizeof(T) + sizeof(size_t));
        greedy_argmax_kernel<<<1, bs, smem, stream>>>(
            w.d_out, logits, vocab_size, warp_size);
        CUDA_CHECK(cudaGetLastError());

        int64_t result;
        CUDA_CHECK(cudaMemcpyAsync(&result, w.d_out, sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    // ── 1. Convert + temperature scale on device ────────────────────────────
    constexpr unsigned int BLK = 256;
    unsigned int grid = static_cast<unsigned int>((vocab_size + BLK - 1) / BLK);
    float inv_t = (temperature != 1.0f) ? 1.0f / temperature : 1.0f;

    convert_and_scale<<<grid, BLK, 0, stream>>>(
        w.probs, logits, vocab_size, inv_t);
    CUDA_CHECK(cudaGetLastError());

    // ── 2. Softmax on device (no host syncs) ────────────────────────────────
    unsigned int rbs = 1;
    while (rbs < vocab_size && rbs < 1024) rbs <<= 1;
    if (rbs > 1024) rbs = 1024;

    reduce_max<<<1, rbs, rbs * sizeof(float), stream>>>(
        w.scalar, w.probs, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    exp_sub<<<grid, BLK, 0, stream>>>(w.probs, vocab_size, w.scalar);
    CUDA_CHECK(cudaGetLastError());

    reduce_sum<<<1, rbs, rbs * sizeof(float), stream>>>(
        w.scalar, w.probs, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    div_by_sum<<<grid, BLK, 0, stream>>>(w.probs, vocab_size, w.scalar);
    CUDA_CHECK(cudaGetLastError());

    // ── 3. GPU sort: CUB radix sort descending (probs + indices) ────────────
    iota_kernel<<<grid, BLK, 0, stream>>>(w.indices, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceRadixSort::SortPairsDescending(
        w.cub_temp, w.cub_temp_bytes,
        w.probs, w.probs_sorted,
        w.indices, w.indices_sorted,
        static_cast<int>(vocab_size),
        0, 32,   // begin_bit, end_bit for float
        stream);
    CUDA_CHECK(cudaGetLastError());

    // ── 4. D2H copy sorted probabilities ────────────────────────────────────
    // When top_k is set, only the first K entries matter.
    size_t copy_n = vocab_size;
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size)
        copy_n = static_cast<size_t>(top_k);

    auto &hbuf = host_probs_buf();
    hbuf.resize(copy_n);
    CUDA_CHECK(cudaMemcpyAsync(hbuf.data(), w.probs_sorted,
                               copy_n * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 5. Host: Top-K / Top-P cutoff → multinomial draw ────────────────────
    // Probs are already sorted descending, so top-K is implicit (copy_n).
    // Top-P: scan until cumulative mass exceeds top_p.
    size_t effective = copy_n;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (size_t i = 0; i < effective; ++i) {
            cum += hbuf[i];
            if (cum > top_p) { effective = i + 1; break; }
        }
    }

    // Multinomial draw over the effective candidates
    float total = 0.0f;
    for (size_t i = 0; i < effective; ++i) total += hbuf[i];

    std::uniform_real_distribution<float> udist(0.0f, total);
    float u = udist(host_rng());

    size_t sampled_pos = effective > 0 ? effective - 1 : 0;
    float running = 0.0f;
    for (size_t i = 0; i < effective; ++i) {
        running += hbuf[i];
        if (running >= u) { sampled_pos = i; break; }
    }

    // ── 6. Fetch the original vocab index for the sampled position ──────────
    int64_t result;
    CUDA_CHECK(cudaMemcpyAsync(&result, w.indices_sorted + sampled_pos,
                               sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════
namespace llaisys::ops::cuda {

int64_t sample(const std::byte *logits, llaisysDataType_t dtype,
               size_t vocab_size, float temperature, int top_k, float top_p,
               llaisysStream_t stream) {
    auto &dev = utils::cuda::get_device_info();
    dev.refresh();
    unsigned int warp_size = dev.warp_size;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_sample(reinterpret_cast<const float *>(logits),
                             vocab_size, temperature, top_k, top_p,
                             warp_size, s);
    case LLAISYS_DTYPE_F16:
        return launch_sample(reinterpret_cast<const __half *>(logits),
                             vocab_size, temperature, top_k, top_p,
                             warp_size, s);
    case LLAISYS_DTYPE_BF16:
        return launch_sample(reinterpret_cast<const __nv_bfloat16 *>(logits),
                             vocab_size, temperature, top_k, top_p,
                             warp_size, s);
    default:
        throw std::runtime_error("sample_cuda: unsupported dtype");
    }
}

} // namespace llaisys::ops::cuda
