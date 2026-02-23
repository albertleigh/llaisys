
//
// Created by ali on 2/16/26.
//

#include "llaisys/models/qwen2.h"

#include "../../core/context/context.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils/check.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>

using namespace llaisys;

namespace {
// --- Helper Functions ---

// Convert int64 shape to size_t shape for Tensor API
std::vector<size_t> to_dims(const std::vector<int64_t> &shape) {
    std::vector<size_t> dims;
    dims.reserve(shape.size());
    for (auto s : shape) {
        dims.push_back(static_cast<size_t>(s));
    }
    return dims;
}

// Wrapper to create tensor using int64 shape
tensor_t create_tensor(const std::vector<int64_t> &shape, llaisysDataType_t dtype, llaisysDeviceType_t device) {
    return Tensor::create(to_dims(shape), dtype, device);
}

// Helper to simulate Tensor::zeros
tensor_t zeros(const std::vector<int64_t> &shape, llaisysDataType_t dtype, llaisysDeviceType_t device) {
    tensor_t t = create_tensor(shape, dtype, device);
    if (device == LLAISYS_DEVICE_CPU) {
        std::memset(t->data(), 0, t->numel() * t->elementSize());
    }
    return t;
}
} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    // current sequence position
    size_t pos;
    llaisysDeviceType_t device_type;

    // Helper: convert opaque C handle to C++ tensor_t (shared_ptr)
    tensor_t t(llaisysTensor_t handle) {
        if (!handle) {
            return nullptr;
        }
        return *reinterpret_cast<tensor_t *>(handle);
    }

    // Helper: Check if tensor loaded
    bool has(llaisysTensor_t handle) {
        return handle != nullptr;
    }
};

extern "C" {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device_type, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();

    model->meta = *meta;
    model->device_type = device_type;
    model->pos = 0;

    // Initialize weight arrays
    size_t n = meta->nlayer;
    model->weights.attn_norm_w = new llaisysTensor_t[n]();
    model->weights.attn_q_w = new llaisysTensor_t[n]();
    model->weights.attn_q_b = new llaisysTensor_t[n]();
    model->weights.attn_k_w = new llaisysTensor_t[n]();
    model->weights.attn_k_b = new llaisysTensor_t[n]();
    model->weights.attn_v_w = new llaisysTensor_t[n]();
    model->weights.attn_v_b = new llaisysTensor_t[n]();
    model->weights.attn_o_w = new llaisysTensor_t[n]();
    model->weights.mlp_norm_w = new llaisysTensor_t[n]();
    model->weights.mlp_gate_w = new llaisysTensor_t[n]();
    model->weights.mlp_up_w = new llaisysTensor_t[n]();
    model->weights.mlp_down_w = new llaisysTensor_t[n]();

    // Allocate KV Cache
    llaisysDataType_t dtype = static_cast<llaisysDataType_t>(meta->dtype);
    for (size_t i = 0; i < n; ++i) {
        std::vector<int64_t> cache_shape = {
            static_cast<int64_t>(meta->maxseq),
            static_cast<int64_t>(meta->nkvh),
            static_cast<int64_t>(meta->dh)};

        model->k_cache.push_back(zeros(cache_shape, dtype, model->device_type));
        model->v_cache.push_back(zeros(cache_shape, dtype, model->device_type));
    }

    return model;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (ntoken == 0) {
        return -1;
    }

    // --- 1. Prepare Inputs ---
    std::vector<int64_t> seq_shape = {static_cast<int64_t>(ntoken)};

    tensor_t input = create_tensor(seq_shape, LLAISYS_DTYPE_I64, model->device_type);
    input->load(token_ids);

    std::vector<int64_t> pos_data(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        pos_data[i] = static_cast<int64_t>(model->pos + i);
    }
    tensor_t pos_ids = create_tensor(seq_shape, ::LLAISYS_DTYPE_I64, model->device_type);
    pos_ids->load(pos_data.data());

    // --- 2. Embedding ---
    tensor_t hidden_states = create_tensor(
        {static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)},
        static_cast<llaisysDataType_t>(model->meta.dtype),
        model->device_type);
    ops::embedding(hidden_states, input, model->t(model->weights.in_embed));

    // --- 3. Layers ---
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        tensor_t residual = hidden_states;

        tensor_t norm_out = Tensor::create(hidden_states->shape(), hidden_states->dtype(), hidden_states->deviceType());
        ops::rms_norm(norm_out, hidden_states, model->t(model->weights.attn_norm_w[i]), model->meta.epsilon);

        // --- Attention Block ---
        tensor_t q_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(q_proj, norm_out, model->t(model->weights.attn_q_w[i]), model->t(model->weights.attn_q_b[i]));

        tensor_t k_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh * model->meta.dh)}, hidden_states->dtype(), model->device_type);
        ops::linear(k_proj, norm_out, model->t(model->weights.attn_k_w[i]), model->t(model->weights.attn_k_b[i]));

        tensor_t v_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh * model->meta.dh)}, hidden_states->dtype(), model->device_type);
        ops::linear(v_proj, norm_out, model->t(model->weights.attn_v_w[i]), model->t(model->weights.attn_v_b[i]));

        tensor_t q = q_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nh), static_cast<int64_t>(model->meta.dh)}));
        tensor_t k = k_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh), static_cast<int64_t>(model->meta.dh)}));
        tensor_t v = v_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh), static_cast<int64_t>(model->meta.dh)}));

        ops::rope(q, q, pos_ids, model->meta.theta);
        ops::rope(k, k, pos_ids, model->meta.theta);

        // Update KV Cache
        tensor_t layer_k_cache = model->k_cache[i];
        tensor_t layer_v_cache = model->v_cache[i];

        {
            size_t row_size_bytes = model->meta.nkvh * model->meta.dh * k->elementSize();
            uint8_t *dst_k_base = reinterpret_cast<uint8_t *>(layer_k_cache->data());
            uint8_t *dst_v_base = reinterpret_cast<uint8_t *>(layer_v_cache->data());
            const uint8_t *src_k_base = reinterpret_cast<const uint8_t *>(k->data());
            const uint8_t *src_v_base = reinterpret_cast<const uint8_t *>(v->data());

            for (size_t t = 0; t < ntoken; ++t) {
                size_t cache_idx = model->pos + t;
                if (cache_idx >= model->meta.maxseq) {
                    break;
                }
                if (model->device_type == LLAISYS_DEVICE_CPU) {
                    std::memcpy(dst_k_base + cache_idx * row_size_bytes, src_k_base + t * row_size_bytes, row_size_bytes);
                    std::memcpy(dst_v_base + cache_idx * row_size_bytes, src_v_base + t * row_size_bytes, row_size_bytes);
                } else {
                    core::context().setDevice(model->device_type, 0);
                    llaisysStream_t stream = core::context().runtime().stream();
                    core::context().runtime().api()->memcpy_async(
                        dst_k_base + cache_idx * row_size_bytes, src_k_base + t * row_size_bytes, row_size_bytes, LLAISYS_MEMCPY_D2D, stream);
                    core::context().runtime().api()->memcpy_async(
                        dst_v_base + cache_idx * row_size_bytes, src_v_base + t * row_size_bytes, row_size_bytes, LLAISYS_MEMCPY_D2D, stream);
                }
            }
        }

        const std::vector<size_t> original_shape = layer_k_cache->shape();

        std::vector<size_t> &mut_k_shape = const_cast<std::vector<size_t> &>(layer_k_cache->shape());
        std::vector<size_t> &mut_v_shape = const_cast<std::vector<size_t> &>(layer_v_cache->shape());

        mut_k_shape[0] = static_cast<size_t>(model->pos + ntoken);
        mut_v_shape[0] = static_cast<size_t>(model->pos + ntoken);

        tensor_t attn_out_view = create_tensor(
            {static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nh), static_cast<int64_t>(model->meta.dh)},
            hidden_states->dtype(), model->device_type);

        float scale = 1.0f / sqrtf(static_cast<float>(model->meta.dh));

        ops::self_attention(attn_out_view, q, layer_k_cache, layer_v_cache, scale);

        mut_k_shape = original_shape;
        mut_v_shape = original_shape;

        tensor_t attn_out = attn_out_view->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}));

        tensor_t o_linear = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(o_linear, attn_out, model->t(model->weights.attn_o_w[i]), nullptr);

        ops::add(hidden_states, residual, o_linear);

        //  MLP Block
        tensor_t mlp_input = hidden_states;
        tensor_t residual_mlp = mlp_input;

        tensor_t mlp_norm = Tensor::create(mlp_input->shape(), mlp_input->dtype(), mlp_input->deviceType());
        ops::rms_norm(mlp_norm, mlp_input, model->t(model->weights.mlp_norm_w[i]), model->meta.epsilon);

        tensor_t gate = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::linear(gate, mlp_norm, model->t(model->weights.mlp_gate_w[i]), nullptr);

        tensor_t up = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::linear(up, mlp_norm, model->t(model->weights.mlp_up_w[i]), nullptr);

        tensor_t act_out = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::swiglu(act_out, gate, up);

        tensor_t down_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(down_proj, act_out, model->t(model->weights.mlp_down_w[i]), nullptr);

        ops::add(hidden_states, residual_mlp, down_proj);
    }

    // --- 4. Final Processing ---

    tensor_t final_norm_out = Tensor::create(hidden_states->shape(), hidden_states->dtype(), hidden_states->deviceType());
    ops::rms_norm(final_norm_out, hidden_states, model->t(model->weights.out_norm_w), model->meta.epsilon);

    // printf("[DEBUG] After Final Norm [0:5] = ");
    // const float* norm_data = reinterpret_cast<const float*>(final_norm_out->data());
    // for (int i = 0; i < std::min(5, (int)model->meta.hs); i++) {
    //     printf("%.4f ", norm_data[i]);
    // }
    // printf("\n");

    tensor_t last_token_emb = create_tensor({1, static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);

    size_t d_bytes = model->meta.hs * final_norm_out->elementSize();
    uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(last_token_emb->data());
    const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(final_norm_out->data());
    if (model->device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst_ptr, src_ptr + (ntoken - 1) * d_bytes, d_bytes);
    } else {
        core::context().setDevice(model->device_type, 0);
        core::context().runtime().api()->memcpy_async(
            dst_ptr, src_ptr + (ntoken - 1) * d_bytes, d_bytes, LLAISYS_MEMCPY_D2D, core::context().runtime().stream());
    }

    tensor_t logits = create_tensor({1, static_cast<int64_t>(model->meta.voc)}, hidden_states->dtype(), model->device_type);
    ops::linear(logits, last_token_emb, model->t(model->weights.out_embed), nullptr);

    tensor_t max_idx = create_tensor({1}, LLAISYS_DTYPE_I64, model->device_type);
    tensor_t max_val = create_tensor({1}, hidden_states->dtype(), model->device_type);

    ops::argmax(max_idx, max_val, logits);

    int64_t next_token;
    if (model->device_type == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<int64_t *>(max_idx->data());
    } else {
        core::context().setDevice(model->device_type, 0);
        // Synchronize the stream before reading result back to host
        core::context().runtime().synchronize();
        core::context().runtime().api()->memcpy_sync(
            &next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    }

    // Update global position
    model->pos += ntoken;

    return next_token;
}

// --- KV Cache API ---

size_t llaisysQwen2ModelGetPos(LlaisysQwen2Model *model) {
    return model->pos;
}

void llaisysQwen2ModelSetPos(LlaisysQwen2Model *model, size_t pos) {
    model->pos = pos;
}

void llaisysQwen2ModelResetKVCache(LlaisysQwen2Model *model) {
    model->pos = 0;
    llaisysDataType_t dtype = static_cast<llaisysDataType_t>(model->meta.dtype);
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        std::vector<int64_t> cache_shape = {
            static_cast<int64_t>(model->meta.maxseq),
            static_cast<int64_t>(model->meta.nkvh),
            static_cast<int64_t>(model->meta.dh)};
        model->k_cache[i] = zeros(cache_shape, dtype, model->device_type);
        model->v_cache[i] = zeros(cache_shape, dtype, model->device_type);
    }
}

static constexpr uint64_t QWEN2_KV_SNAPSHOT_MAGIC = 0x4C4C41495359534BuLL; // "LLAISYSK"

struct LlaisysQwen2KVSnapshot {
    uint64_t magic;  // must equal QWEN2_KV_SNAPSHOT_MAGIC
    size_t pos;      // number of tokens stored
    size_t nlayer;
    // CPU tensors of shape [pos, nkvh, dh] — only the used portion
    std::vector<tensor_t> k_data;
    std::vector<tensor_t> v_data;
};

LlaisysQwen2KVSnapshot_t llaisysQwen2ModelSaveKV(LlaisysQwen2Model *model) {
    auto snap = new LlaisysQwen2KVSnapshot();
    snap->magic = QWEN2_KV_SNAPSHOT_MAGIC;
    snap->pos = model->pos;
    snap->nlayer = model->meta.nlayer;

    if (model->pos == 0) {
        // Nothing to snapshot — leave vectors empty
        snap->k_data.resize(model->meta.nlayer, nullptr);
        snap->v_data.resize(model->meta.nlayer, nullptr);
        return reinterpret_cast<LlaisysQwen2KVSnapshot_t>(snap);
    }

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        // Slice the first `pos` rows out of the [maxseq, nkvh, dh] cache
        tensor_t k_slice = model->k_cache[i]->slice(0, 0, model->pos);
        tensor_t v_slice = model->v_cache[i]->slice(0, 0, model->pos);

        // Copy to CPU (contiguous + device→host copy handled by Tensor::to)
        tensor_t cpu_k = k_slice->to(LLAISYS_DEVICE_CPU);
        tensor_t cpu_v = v_slice->to(LLAISYS_DEVICE_CPU);

        snap->k_data.push_back(cpu_k);
        snap->v_data.push_back(cpu_v);
    }

    return reinterpret_cast<LlaisysQwen2KVSnapshot_t>(snap);
}

void llaisysQwen2ModelLoadKV(LlaisysQwen2Model *model, LlaisysQwen2KVSnapshot_t snapshot) {
    auto snap = reinterpret_cast<LlaisysQwen2KVSnapshot *>(snapshot);
    if (!snap) return;
    if (snap->magic != QWEN2_KV_SNAPSHOT_MAGIC) {
        fprintf(stderr, "llaisysQwen2ModelLoadKV: invalid snapshot (bad magic 0x%llx)\n",
                (unsigned long long)snap->magic);
        return;
    }

    model->pos = snap->pos;

    if (snap->pos == 0) return;

    for (size_t i = 0; i < snap->nlayer && i < model->meta.nlayer; ++i) {
        if (!snap->k_data[i] || !snap->v_data[i]) continue;

        size_t row_bytes = model->meta.nkvh * model->meta.dh
                         * model->k_cache[i]->elementSize();
        size_t copy_bytes = snap->pos * row_bytes;

        if (model->device_type == LLAISYS_DEVICE_CPU) {
            // CPU→CPU
            std::memcpy(model->k_cache[i]->data(),
                        snap->k_data[i]->data(), copy_bytes);
            std::memcpy(model->v_cache[i]->data(),
                        snap->v_data[i]->data(), copy_bytes);
        } else {
            // CPU→Device
            core::context().setDevice(model->device_type, 0);
            core::context().runtime().api()->memcpy_sync(
                model->k_cache[i]->data(),
                snap->k_data[i]->data(),
                copy_bytes, LLAISYS_MEMCPY_H2D);
            core::context().runtime().api()->memcpy_sync(
                model->v_cache[i]->data(),
                snap->v_data[i]->data(),
                copy_bytes, LLAISYS_MEMCPY_H2D);
        }
    }
}

size_t llaisysQwen2KVSnapshotGetPos(LlaisysQwen2KVSnapshot_t snapshot) {
    auto snap = reinterpret_cast<LlaisysQwen2KVSnapshot *>(snapshot);
    if (!snap) return 0;
    if (snap->magic != QWEN2_KV_SNAPSHOT_MAGIC) return 0;
    return snap->pos;
}

void llaisysQwen2KVSnapshotDestroy(LlaisysQwen2KVSnapshot_t snapshot) {
    auto snap = reinterpret_cast<LlaisysQwen2KVSnapshot *>(snapshot);
    if (snap && snap->magic != QWEN2_KV_SNAPSHOT_MAGIC) {
        fprintf(stderr, "llaisysQwen2KVSnapshotDestroy: invalid snapshot (bad magic)\n");
        return;
    }
    delete snap;  // shared_ptrs in vectors auto-release CPU memory
}
}