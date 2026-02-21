//
// Created by ali on 2/22/26.
//

#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);
} // namespace llaisys::ops::cuda
