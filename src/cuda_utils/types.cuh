//
// Created by ali on 2/20/26.
//


#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>


#pragma once
namespace llaisys::utils::cuda {
// ── helpers: convert any supported T to float on device ─────────────────────
template <typename T>
__device__ __forceinline__ float to_float(T v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

} // namespace llaisys::utils::cuda