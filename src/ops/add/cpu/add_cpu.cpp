#include "add_cpu.hpp"

#include "../../../utils.hpp"
#include "../../../utils/omp_compat.hpp"

#include <cmath>

namespace {
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    // For small arrays, use simple loop without parallelization overhead
    constexpr size_t PARALLEL_THRESHOLD = 1024;

    if (numel < PARALLEL_THRESHOLD) {
        // Small array - simple loop
        for (size_t i = 0; i < numel; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
            } else {
                c[i] = a[i] + b[i];
            }
        }
    } else {
        // Large array - use OpenMP parallelization with SIMD
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // Half precision: convert to float, add, convert back
            OMP_PARALLEL_FOR_SIMD_SCHED(static)
            for (omp_idx_t i = 0; i < OMP_CAST(numel); i++) {
                c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
            }
        } else {
            // Full precision: direct addition with SIMD
            OMP_PARALLEL_FOR_SIMD_SCHED(static)
            for (omp_idx_t i = 0; i < OMP_CAST(numel); i++) {
                c[i] = a[i] + b[i];
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
