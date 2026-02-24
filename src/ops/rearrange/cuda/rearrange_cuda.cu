//
// Created by ali on 2/22/26.
//

#include "rearrange_cuda.cuh"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../../cuda_utils/check.cuh"
#include "../../../utils.hpp"

namespace {

template <typename T>
__global__ void rearrange_kernel(T *out, const T *in,
								 const size_t *shape,
								 const size_t *stride_in,
								 const size_t *stride_out,
								 size_t ndim,
								 size_t total) {
	size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
	if (idx >= total) return;

	size_t rem = idx;
	size_t off_in = 0;
	size_t off_out = 0;

	for (size_t d = ndim; d-- > 0;) {
		size_t cur = rem % shape[d];
		rem /= shape[d];
		off_in += cur * stride_in[d];
		off_out += cur * stride_out[d];
	}

	out[off_out] = in[off_in];
}

template <typename T>
void launch_rearrange(T *out, const T *in,
					  const std::vector<size_t> &shape,
					  const std::vector<size_t> &stride_in,
					  const std::vector<size_t> &stride_out) {
	size_t total = 1;
	for (size_t i = 0; i < shape.size(); ++i) total *= shape[i];
	if (total == 0) return;

	const size_t ndim = shape.size();
	size_t *d_shape = nullptr;
	size_t *d_stride_in = nullptr;
	size_t *d_stride_out = nullptr;

	CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_stride_in, ndim * sizeof(size_t)));
	CUDA_CHECK(cudaMalloc(&d_stride_out, ndim * sizeof(size_t)));

	CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_stride_in, stride_in.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_stride_out, stride_out.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));

	int block = 256;
	int grid = static_cast<int>((total + block - 1) / block);
	if (grid > 65535) grid = 65535;

	rearrange_kernel<<<grid, block>>>(out, in, d_shape, d_stride_in, d_stride_out, ndim, total);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaFree(d_stride_out));
	CUDA_CHECK(cudaFree(d_stride_in));
	CUDA_CHECK(cudaFree(d_shape));
}

} // namespace

namespace llaisys::ops::cuda {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
			   const std::vector<size_t> &shape,
			   const std::vector<size_t> &stride_in,
			   const std::vector<size_t> &stride_out) {
	if (shape.empty()) {
		CUDA_CHECK(cudaMemcpy(out, in, llaisys::utils::dsize(dtype), cudaMemcpyDeviceToDevice));
		return;
	}

	switch (dtype) {
	case LLAISYS_DTYPE_F16:
		return launch_rearrange(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
								shape, stride_in, stride_out);
	case LLAISYS_DTYPE_BF16:
		return launch_rearrange(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
								shape, stride_in, stride_out);
	case LLAISYS_DTYPE_F32:
		return launch_rearrange(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
								shape, stride_in, stride_out);
	case LLAISYS_DTYPE_I64:
		return launch_rearrange(reinterpret_cast<int64_t *>(out), reinterpret_cast<const int64_t *>(in),
								shape, stride_in, stride_out);
	default:
		throw std::runtime_error("rearrange_cuda: unsupported dtype");
	}
}
} // namespace llaisys::ops::cuda
