//
// Created by ali on 2/11/26.
//

#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t dtype, std::array<size_t, 2> dims);
} // namespace llaisys::ops::cpu
