#pragma once

#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

/**
 * Sample a token index from a 1-D logits vector using Temperature,
 * Top-K, and Top-P (nucleus) sampling.
 *
 * @param logits     Raw logit values (1-D, vocab_size elements).
 * @param dtype      Element type of the logits buffer.
 * @param vocab_size Number of vocabulary entries.
 * @param temperature  Softmax temperature (<=0 → greedy / argmax).
 * @param top_k      Keep only the top-K logits (<=0 or >=vocab_size → no filter).
 * @param top_p      Nucleus probability threshold in (0, 1] (>=1 → no filter).
 * @return           Sampled token index in [0, vocab_size).
 */
int64_t sample(const std::byte *logits, llaisysDataType_t dtype,
               size_t vocab_size, float temperature, int top_k, float top_p);

} // namespace llaisys::ops::cpu
