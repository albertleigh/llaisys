//
// Created by ali on 2/20/26.
//

#pragma once
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

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
namespace llaisys::utils::cuda {
// ── cached device properties ────────────────────────────────────────────────
struct DeviceInfo {
    unsigned int warp_size;
    unsigned int max_block;
    int sm_count;
    bool valid = false;
    int device_id = -1;

    void refresh() {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        if (valid && dev == device_id) {
            return;
        }

        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
        warp_size = static_cast<unsigned int>(props.warpSize);
        max_block = static_cast<unsigned int>(props.maxThreadsPerBlock);
        if (max_block > 1024) {
            max_block = 1024;
        }
        sm_count = props.multiProcessorCount;
        device_id = dev;
        valid = true;
    }
};

inline DeviceInfo &get_device_info() {
    static DeviceInfo info;
    return info;
}
} // namespace llaisys::utils::cuda