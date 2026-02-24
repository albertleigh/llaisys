//
// Created by ali on 2/22/26.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel, llaisysStream_t stream = core::context().runtime().stream());
} // namespace llaisys::ops::cuda
