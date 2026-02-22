//
// Created by ali on 2/22/26.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t dtype, const std::vector<size_t> &dims, llaisysStream_t stream = core::context().runtime().stream());
} // namespace llaisys::ops::cuda
