//
// Created by ali on 2/20/26.
//

#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#ifdef ENABLE_NVIDIA_API
#define CUDA_CHECK(call)                                                                                                \
    do {                                                                                                                \
        cudaError_t error = call;                                                                                       \
        if (error != cudaSuccess) {                                                                                     \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << "\n"; \
            throw std::runtime_error(cudaGetErrorString(error));                                                        \
        }                                                                                                               \
    } while (0)

#define CUDA_FREE(ptr)                                                                                                           \
    do {                                                                                                                         \
        if (ptr) {                                                                                                               \
            cudaError_t error = cudaFree(ptr);                                                                                   \
            if (error != cudaSuccess) {                                                                                          \
                std::cerr << "CUDA free error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << "\n"; \
                throw std::runtime_error(cudaGetErrorString(error));                                                             \
            }                                                                                                                    \
            ptr = nullptr;                                                                                                       \
        }                                                                                                                        \
    } while (0)
#endif