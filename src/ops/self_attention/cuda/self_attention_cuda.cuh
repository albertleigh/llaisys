//
// Created by ali on 2/22/26.
//

#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cuda {
void self_attention(
    std::byte *attn, const std::byte *q, const std::byte *k, const std::byte *v,
    float scale, llaisysDataType_t dtype,
    const std::vector<size_t> &q_shap,
    const std::vector<size_t> &k_shap,
    const std::vector<size_t> &v_shap);
} // namespace llaisys::ops::cuda
