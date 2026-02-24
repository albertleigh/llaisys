//
// Created by ali on 2/20/26.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void argmax(std::byte *max_id, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel, llaisysStream_t stream = core::context().runtime().stream());
}
