//
// Created by ali on 2/11/26.
//

#include "llaisys.h"

#include <cstddef>
#pragma once
namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t size, size_t embedding_dim);
} // namespace llaisys::ops::cpu