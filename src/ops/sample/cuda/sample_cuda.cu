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
//   3. D2H: copy probability vector to host  (~600 KB for 150K vocab)
//   4. Host: Top-K filter → Top-P filter → multinomial draw
//
// Steps 1–2 exploit GPU parallelism for the heavy lifting.
// Steps 3–4 are fast on the host because Top-K/Top-P and a single
// random draw are inherently sequential and the data fits in L2.
//

#include "sample_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef _MSC_VER
#include <cfloat>
#ifndef __FLT_MAX__
#define __FLT_MAX__ FLT_MAX
#endif
#endif

#include "../../../cuda_utils/check.cuh"
#include "../../../cuda_utils/types.cuh"
#include "../../../utils.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace {

// ── Host RNG ────────────────────────────────────────────────────────────────
static std::mt19937 &host_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
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
//  Softmax helper kernels
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

// exp(x − max_val)  element-wise.
__global__ void exp_sub(float *v, size_t n, float max_val) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = expf(v[i] - max_val);
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

// v[i] /= *sum_ptr  element-wise.
__global__ void div_by(float *v, size_t n, float inv_sum) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv_sum;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Persistent workspace
// ═══════════════════════════════════════════════════════════════════════════
struct SampleWorkspace {
    float   *probs   = nullptr;  // [vocab_size]
    float   *scalar  = nullptr;  // single float (max or sum)
    int64_t *d_out   = nullptr;  // device result for greedy path
    size_t   cap     = 0;

    void ensure(size_t n) {
        if (n <= cap) return;
        if (probs)  cudaFree(probs);
        if (scalar) cudaFree(scalar);
        if (d_out)  cudaFree(d_out);
        cap = n;
        CUDA_CHECK(cudaMalloc(&probs,  cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&scalar, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out,  sizeof(int64_t)));
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

    // ── 2. Softmax on device ────────────────────────────────────────────────
    // Reduction block size — power-of-2, ≤ 1024
    unsigned int rbs = 1;
    while (rbs < vocab_size && rbs < 1024) rbs <<= 1;
    if (rbs > 1024) rbs = 1024;

    //  a) max-reduce
    reduce_max<<<1, rbs, rbs * sizeof(float), stream>>>(
        w.scalar, w.probs, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    float h_max;
    CUDA_CHECK(cudaMemcpyAsync(&h_max, w.scalar, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //  b) exp(x − max)
    exp_sub<<<grid, BLK, 0, stream>>>(w.probs, vocab_size, h_max);
    CUDA_CHECK(cudaGetLastError());

    //  c) sum-reduce
    reduce_sum<<<1, rbs, rbs * sizeof(float), stream>>>(
        w.scalar, w.probs, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    float h_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, w.scalar, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //  d) normalise
    float inv_sum = (h_sum > 0.0f) ? 1.0f / h_sum : 0.0f;
    div_by<<<grid, BLK, 0, stream>>>(w.probs, vocab_size, inv_sum);
    CUDA_CHECK(cudaGetLastError());

    // ── 3. D2H copy probabilities ───────────────────────────────────────────
    std::vector<float> h_probs(vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(h_probs.data(), w.probs,
                               vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 4. Host-side Top-K + Top-P + draw ───────────────────────────────────
    std::vector<int64_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), int64_t(0));

    // Top-K
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<ptrdiff_t>(top_k),
            indices.end(),
            [&](int64_t a, int64_t b) { return h_probs[a] > h_probs[b]; });
        for (size_t i = static_cast<size_t>(top_k); i < vocab_size; ++i)
            h_probs[indices[i]] = 0.0f;
        float norm = 0.0f;
        for (auto p : h_probs) norm += p;
        if (norm > 0.0f) for (auto &p : h_probs) p /= norm;
    }

    // Top-P
    if (top_p > 0.0f && top_p < 1.0f) {
        std::sort(indices.begin(), indices.end(),
                  [&](int64_t a, int64_t b) { return h_probs[a] > h_probs[b]; });
        float cum = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cum += h_probs[indices[i]];
            if (cum > top_p) { cutoff = i + 1; break; }
        }
        for (size_t i = cutoff; i < vocab_size; ++i)
            h_probs[indices[i]] = 0.0f;
        float norm = 0.0f;
        for (auto p : h_probs) norm += p;
        if (norm > 0.0f) for (auto &p : h_probs) p /= norm;
    }

    // Multinomial draw
    std::discrete_distribution<int64_t> dist(h_probs.begin(), h_probs.end());
    return dist(host_rng());
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
