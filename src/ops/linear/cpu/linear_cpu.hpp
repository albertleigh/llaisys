//
// Created by ali on 2/11/26.
//

#include "llaisys.h"
#include <array>
#include <cstddef>
#pragma once
namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, std::array<size_t, 3> dims);
} // namespace llaisys::ops::cpu