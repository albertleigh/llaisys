#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops {

/**
 * Sample a token index from a 1-D logits tensor.
 *
 * Supports CPU and NVIDIA GPU tensors.  On GPU the softmax is performed
 * on-device and only the probability vector is copied to host for the
 * Top-K / Top-P / multinomial draw.
 *
 * @param logits      1-D tensor of shape [vocab_size].
 * @param temperature Softmax temperature (<=0 → greedy argmax).
 * @param top_k       Keep only the K highest logits (<=0 → disabled).
 * @param top_p       Nucleus probability threshold in (0,1] (>=1 → disabled).
 * @return            Sampled token index.
 */
int64_t sample(tensor_t logits, float temperature, int top_k, float top_p);

} // namespace llaisys::ops
