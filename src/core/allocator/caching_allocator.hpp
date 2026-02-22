#pragma once

#include "allocator.hpp"

#include <map>
#include <unordered_map>

namespace llaisys::core::allocators {

/// A caching memory allocator that pools freed GPU allocations for reuse.
///
/// When release() is called, the memory is NOT freed but placed in a pool
/// keyed by size. Future allocate() calls check the pool first and reuse
/// a block of matching or larger size, avoiding expensive cudaMalloc/cudaFree.
///
/// This eliminates the primary bottleneck in GPU inference loops where
/// hundreds of temporary tensors are created and destroyed per step.
class CachingAllocator : public MemoryAllocator {
private:
    // Free blocks indexed by size for O(log n) best-fit lookup
    std::multimap<size_t, std::byte *> _free_blocks;

    // Track allocated size for each pointer (needed for returning to pool)
    std::unordered_map<std::byte *, size_t> _allocated_sizes;

public:
    CachingAllocator(const LlaisysRuntimeAPI *runtime_api);
    ~CachingAllocator();

    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;
};

} // namespace llaisys::core::allocators
