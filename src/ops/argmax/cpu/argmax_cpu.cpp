//
// Created by ali on 2/10/26.
//

#include "argmax_cpu.hpp"

#include "../../../utils.hpp"
#include "../../../utils/omp_compat.hpp"

#include <limits>

namespace {
template <typename T>
void argmax_(size_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // For small arrays, use simple sequential search
    constexpr size_t PARALLEL_THRESHOLD = 10000;

    if (numel < PARALLEL_THRESHOLD) {
        // Small array - simple sequential scan
        size_t max_index = 0;
        float max_value = llaisys::utils::cast<float>(vals[0]);

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t i = 1; i < numel; ++i) {
                const float current_value = llaisys::utils::cast<float>(vals[i]);
                if (current_value > max_value) {
                    max_index = i;
                    max_value = current_value;
                }
            }
        } else {
            for (size_t i = 1; i < numel; ++i) {
                if (vals[i] > max_value) {
                    max_index = i;
                    max_value = vals[i];
                }
            }
        }
        *max_idx = max_index;
        *max_val = llaisys::utils::cast<T>(max_value);
    } else {
        // Large array - use OpenMP parallel reduction
        size_t global_max_index = 0;
        float global_max_value = llaisys::utils::cast<float>(vals[0]);

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // Half precision path
            #pragma omp parallel
            {
                size_t local_max_index = 0;
                float local_max_value = std::numeric_limits<float>::lowest();

                OMP_FOR_NOWAIT
                for (omp_idx_t i = 0; i < OMP_CAST(numel); ++i) {
                    const float current_value = llaisys::utils::cast<float>(vals[i]);
                    if (current_value > local_max_value) {
                        local_max_index = static_cast<size_t>(i);
                        local_max_value = current_value;
                    }
                }

                // Combine thread-local results
                #pragma omp critical
                {
                    if (local_max_value > global_max_value) {
                        global_max_index = local_max_index;
                        global_max_value = local_max_value;
                    }
                }
            }
        } else {
            // Full precision path
            #pragma omp parallel
            {
                size_t local_max_index = 0;
                float local_max_value = std::numeric_limits<float>::lowest();

                OMP_FOR_NOWAIT
                for (omp_idx_t i = 0; i < OMP_CAST(numel); ++i) {
                    const float current_value = static_cast<float>(vals[i]);
                    if (current_value > local_max_value) {
                        local_max_index = static_cast<size_t>(i);
                        local_max_value = current_value;
                    }
                }

                // Combine thread-local results
                #pragma omp critical
                {
                    if (local_max_value > global_max_value) {
                        global_max_index = local_max_index;
                        global_max_value = local_max_value;
                    }
                }
            }
        }

        *max_idx = global_max_index;
        *max_val = llaisys::utils::cast<T>(global_max_value);
    }
}
} // namespace

namespace llaisys::ops::cpu {

void argmax(std::byte *max_id, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel) {
    if (numel == 0) {
        EXCEPTION_INVALID_ARGUMENT("argmax on empty tensor");
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t *>(max_id), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t *>(max_id), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t *>(max_id), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu