//
// Created by ali on 2/21/26.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t size, size_t embedding_dim, llaisysStream_t stream = core::context().runtime().stream());
} // namespace llaisys::ops::cuda
