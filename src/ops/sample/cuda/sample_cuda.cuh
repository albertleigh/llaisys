//
// CUDA sampling header — mirrors argmax_cuda.cuh pattern.
//

#pragma once
#include "../../../core/llaisys_core.hpp"
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda {

/**
 * Sample a token index from a 1-D logits buffer on device.
 *
 * Implements Temperature scaling, Top-K filtering, softmax, Top-P
 * (nucleus) filtering, and multinomial sampling — all on-device.
 *
 * @param logits      Device pointer to raw logits (vocab_size elements).
 * @param dtype       Element type (F32, F16, BF16).
 * @param vocab_size  Number of vocabulary entries.
 * @param temperature Softmax temperature (<=0 → greedy argmax).
 * @param top_k       Keep only the K highest logits (<=0 → disabled).
 * @param top_p       Nucleus probability threshold in (0,1] (>=1 → disabled).
 * @param stream      CUDA stream (defaults to the context's current stream).
 * @return            Sampled token index in [0, vocab_size).
 */
int64_t sample(const std::byte *logits, llaisysDataType_t dtype,
               size_t vocab_size, float temperature, int top_k, float top_p,
               llaisysStream_t stream = core::context().runtime().stream());

} // namespace llaisys::ops::cuda
