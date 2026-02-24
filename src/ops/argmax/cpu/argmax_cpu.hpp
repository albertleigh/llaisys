//
// Created by ali on 2/10/26.
//

#include "llaisys.h"
#include <cstddef>

#pragma once

namespace llaisys::ops::cpu {

void argmax(std::byte *max_id, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t size);

} // namespace llaisys::ops::cpu
