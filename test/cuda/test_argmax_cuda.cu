//
// GTest for CUDA argmax kernel — debuggable with cuda-gdb.
//
// Build & run:
//   xmake f -m debug --nv-gpu=y && xmake build test-cuda
//   ./build/linux/x86_64/debug/test-cuda
//
// Debug:
//   cuda-gdb ./build/linux/x86_64/debug/test-cuda
//   (cuda-gdb) break argmax_single_block
//   (cuda-gdb) run
//

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// Internal kernel header — direct call, no FFI
#include "cuda_utils/check.cuh"
#include "ops/argmax/cuda/argmax_cuda.cuh"

namespace {
// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_random_f32(std::vector<float> &v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (auto &x : v) {
        x = dist(rng);
    }
}

static size_t cpu_argmax_f32(const std::vector<float> &v) {
    return static_cast<size_t>(
        std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}
} // namespace

namespace llaisys::test::cuda::argmax {
// ─── test fixtures ──────────────────────────────────────────────────────────

class ArgmaxCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

// ─── small input  (single-block path, < 64K) ───────────────────────────────

TEST_F(ArgmaxCudaTest, SmallF32) {
    constexpr size_t N = 128;
    std::vector<float> h_vals(N);
    fill_random_f32(h_vals);

    // Plant a known maximum
    h_vals[42] = 9999.0f;

    // Allocate device memory
    float *d_vals = nullptr;
    float *d_maxval = nullptr;
    size_t *d_maxid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxval, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxid, sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Call the kernel under test
    llaisys::ops::cuda::argmax(
        reinterpret_cast<std::byte *>(d_maxid),
        reinterpret_cast<std::byte *>(d_maxval),
        reinterpret_cast<const std::byte *>(d_vals),
        LLAISYS_DTYPE_F32, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    size_t h_maxid = 0;
    float h_maxval = 0;
    CUDA_CHECK(cudaMemcpy(&h_maxid, d_maxid, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxval, d_maxval, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_maxid, 42u);
    EXPECT_FLOAT_EQ(h_maxval, 9999.0f);

    cudaFree(d_vals);
    cudaFree(d_maxval);
    cudaFree(d_maxid);
}

// ─── large input (multi-block path, > 64K) ─────────────────────────────────

TEST_F(ArgmaxCudaTest, LargeF32) {
    constexpr size_t N = 2 * 1024 * 1024; // 2M elements
    std::vector<float> h_vals(N);
    fill_random_f32(h_vals);

    size_t expected_idx = cpu_argmax_f32(h_vals);
    float expected_val = h_vals[expected_idx];

    float *d_vals = nullptr;
    float *d_maxval = nullptr;
    size_t *d_maxid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxval, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxid, sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    llaisys::ops::cuda::argmax(
        reinterpret_cast<std::byte *>(d_maxid),
        reinterpret_cast<std::byte *>(d_maxval),
        reinterpret_cast<const std::byte *>(d_vals),
        LLAISYS_DTYPE_F32, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t h_maxid = 0;
    float h_maxval = 0;
    CUDA_CHECK(cudaMemcpy(&h_maxid, d_maxid, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxval, d_maxval, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_maxid, expected_idx);
    EXPECT_FLOAT_EQ(h_maxval, expected_val);

    cudaFree(d_vals);
    cudaFree(d_maxval);
    cudaFree(d_maxid);
}

// ─── f16 ────────────────────────────────────────────────────────────────────

TEST_F(ArgmaxCudaTest, SmallF16) {
    constexpr size_t N = 256;
    std::vector<__half> h_vals(N);

    // Fill with sequential values so argmax is deterministic
    for (size_t i = 0; i < N; ++i) {
        h_vals[i] = __float2half(static_cast<float>(i));
    }

    // Plant known max
    h_vals[100] = __float2half(9999.0f);

    __half *d_vals = nullptr;
    __half *d_maxval = nullptr;
    size_t *d_maxid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_maxval, sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_maxid, sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), N * sizeof(__half),
                          cudaMemcpyHostToDevice));

    llaisys::ops::cuda::argmax(
        reinterpret_cast<std::byte *>(d_maxid),
        reinterpret_cast<std::byte *>(d_maxval),
        reinterpret_cast<const std::byte *>(d_vals),
        LLAISYS_DTYPE_F16, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t h_maxid = 0;
    __half h_maxval;
    CUDA_CHECK(cudaMemcpy(&h_maxid, d_maxid, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxval, d_maxval, sizeof(__half), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_maxid, 100u);
    EXPECT_NEAR(__half2float(h_maxval), 9999.0f, 10.0f); // f16 has limited precision

    cudaFree(d_vals);
    cudaFree(d_maxval);
    cudaFree(d_maxid);
}

// ─── edge case: single element ──────────────────────────────────────────────

TEST_F(ArgmaxCudaTest, SingleElement) {
    float val = 3.14f;

    float *d_vals = nullptr;
    float *d_maxval = nullptr;
    size_t *d_maxid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxval, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxid, sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_vals, &val, sizeof(float), cudaMemcpyHostToDevice));

    llaisys::ops::cuda::argmax(
        reinterpret_cast<std::byte *>(d_maxid),
        reinterpret_cast<std::byte *>(d_maxval),
        reinterpret_cast<const std::byte *>(d_vals),
        LLAISYS_DTYPE_F32, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t h_maxid = 999;
    float h_maxval = 0;
    CUDA_CHECK(cudaMemcpy(&h_maxid, d_maxid, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxval, d_maxval, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_maxid, 0u);
    EXPECT_FLOAT_EQ(h_maxval, 3.14f);

    cudaFree(d_vals);
    cudaFree(d_maxval);
    cudaFree(d_maxid);
}
} // namespace llaisys::test::cuda::argmax