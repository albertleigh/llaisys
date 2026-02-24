//
// Created by ali on 2/15/26.
//

#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel);
}