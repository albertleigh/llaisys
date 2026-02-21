//
// Created by ali on 2/22/26.
//

#pragma once
#include "llaisys.h"

#include <array>
#include <cstddef>

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, std::array<size_t, 3> dims);
} // namespace llaisys::ops::cuda
