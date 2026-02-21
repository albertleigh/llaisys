//
// Created by ali on 2/22/26.
//

#pragma once
#include "llaisys.h"

#include <array>
#include <cstddef>

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, std::array<size_t, 2> dims);
} // namespace llaisys::ops::cuda
