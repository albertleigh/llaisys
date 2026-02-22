#include "caching_allocator.hpp"

namespace llaisys::core::allocators {

CachingAllocator::CachingAllocator(const LlaisysRuntimeAPI *runtime_api)
    : MemoryAllocator(runtime_api) {
}

CachingAllocator::~CachingAllocator() {
    // Actually free all pooled blocks
    for (auto &[size, ptr] : _free_blocks) {
        _api->free_device(ptr);
    }
    _free_blocks.clear();
    _allocated_sizes.clear();
}

std::byte *CachingAllocator::allocate(size_t size) {
    // Look for a free block of at least the requested size (best-fit)
    auto it = _free_blocks.lower_bound(size);
    if (it != _free_blocks.end()) {
        // Reuse a cached block — no cudaMalloc needed
        std::byte *ptr = it->second;
        size_t block_size = it->first;
        _free_blocks.erase(it);
        _allocated_sizes[ptr] = block_size;
        return ptr;
    }

    // No cached block available — must allocate from device
    std::byte *ptr = static_cast<std::byte *>(_api->malloc_device(size));
    _allocated_sizes[ptr] = size;
    return ptr;
}

void CachingAllocator::release(std::byte *memory) {
    auto it = _allocated_sizes.find(memory);
    if (it != _allocated_sizes.end()) {
        // Return to pool instead of freeing
        _free_blocks.insert({it->second, memory});
        _allocated_sizes.erase(it);
    } else {
        // Unknown pointer — free directly as fallback
        _api->free_device(memory);
    }
}

} // namespace llaisys::core::allocators
