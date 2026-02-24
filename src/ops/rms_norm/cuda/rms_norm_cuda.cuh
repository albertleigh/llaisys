//
// Created by ali on 2/22/26.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <array>
#include <cstddef>

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, std::array<size_t, 2> dims, llaisysStream_t stream = core::context().runtime().stream());
} // namespace llaisys::ops::cuda
