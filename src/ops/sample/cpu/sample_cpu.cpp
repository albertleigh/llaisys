//
// Random sampling with Temperature, Top-K, and Top-P (nucleus) support.
//

#include "sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

// Thread-local Mersenne Twister seeded from hardware entropy.
static std::mt19937 &thread_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

template <typename T>
int64_t sample_(const T *logits_data, size_t vocab_size,
                float temperature, int top_k, float top_p) {
    // ── 1. Convert logits to float workspace ────────────────────────
    std::vector<float> logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = llaisys::utils::cast<float>(logits_data[i]);
    }

    // ── 2. Greedy fast-path ─────────────────────────────────────────
    //    temperature <= 0 or top_k == 1 → deterministic argmax.
    if (temperature <= 0.0f || top_k == 1) {
        auto it = std::max_element(logits.begin(), logits.end());
        return static_cast<int64_t>(std::distance(logits.begin(), it));
    }

    // ── 3. Temperature scaling ──────────────────────────────────────
    if (temperature != 1.0f) {
        const float inv_t = 1.0f / temperature;
        for (auto &v : logits) {
            v *= inv_t;
        }
    }

    // ── 4. Build index array (used by Top-K and Top-P) ──────────────
    std::vector<int64_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), int64_t(0));

    // ── 5. Top-K filtering ──────────────────────────────────────────
    size_t active_size = vocab_size;
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        active_size = static_cast<size_t>(top_k);
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<ptrdiff_t>(active_size),
            indices.end(),
            [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
        // Mask out everything outside the top-K
        for (size_t i = active_size; i < vocab_size; ++i) {
            logits[indices[i]] = -std::numeric_limits<float>::infinity();
        }
    }

    // ── 6. Softmax over surviving logits ────────────────────────────
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum_exp += logits[i];
    }
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] /= sum_exp;
    }

    // ── 7. Top-P (nucleus) filtering ────────────────────────────────
    if (top_p > 0.0f && top_p < 1.0f) {
        // Re-sort indices by probability (descending)
        std::sort(indices.begin(), indices.end(),
                  [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });

        float cumulative = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumulative += logits[indices[i]];
            if (cumulative > top_p) {
                cutoff = i + 1; // keep at least one token
                break;
            }
        }
        // Zero out tokens beyond the nucleus
        for (size_t i = cutoff; i < vocab_size; ++i) {
            logits[indices[i]] = 0.0f;
        }
        // Re-normalise
        float norm = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            norm += logits[i];
        }
        if (norm > 0.0f) {
            for (size_t i = 0; i < vocab_size; ++i) {
                logits[i] /= norm;
            }
        }
    }

    // ── 8. Multinomial sampling ─────────────────────────────────────
    std::discrete_distribution<int64_t> dist(logits.begin(), logits.end());
    return dist(thread_rng());
}

} // namespace

namespace llaisys::ops::cpu {

int64_t sample(const std::byte *logits_data, llaisysDataType_t dtype,
               size_t vocab_size, float temperature, int top_k, float top_p) {
    if (vocab_size == 0) {
        EXCEPTION_INVALID_ARGUMENT("sample on empty logits");
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sample_(reinterpret_cast<const float *>(logits_data),
                        vocab_size, temperature, top_k, top_p);
    case LLAISYS_DTYPE_F16:
        return sample_(reinterpret_cast<const llaisys::fp16_t *>(logits_data),
                        vocab_size, temperature, top_k, top_p);
    case LLAISYS_DTYPE_BF16:
        return sample_(reinterpret_cast<const llaisys::bf16_t *>(logits_data),
                        vocab_size, temperature, top_k, top_p);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
