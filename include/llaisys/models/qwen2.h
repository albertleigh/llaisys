#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device_type, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids, size_t ntoken,
        float temperature, int top_k, float top_p);

    // --- KV Cache API ---
    // These functions copy KV cache data between device and host memory.
    // Only ONE KV cache lives on device at any time.  Snapshots reside
    // in CPU memory so that device memory can be freed/reused.

    /** Get the current sequence position (number of tokens processed so far). */
    __export size_t llaisysQwen2ModelGetPos(struct LlaisysQwen2Model *model);

    /** Set the current sequence position. */
    __export void llaisysQwen2ModelSetPos(struct LlaisysQwen2Model *model, size_t pos);

    /** Reset all KV caches to zero and reset position to 0. */
    __export void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model *model);

    /** Opaque handle for a KV cache snapshot stored in host memory. */
    typedef void *LlaisysQwen2KVSnapshot_t;

    /**
     * Copy the used portion of the model's device KV caches to CPU memory.
     * Returns an opaque snapshot handle.  Only the first ``pos`` rows are
     * copied, not the full ``maxseq`` allocation.
     */
    __export LlaisysQwen2KVSnapshot_t llaisysQwen2ModelSaveKV(struct LlaisysQwen2Model *model);

    /**
     * Restore a previously saved snapshot into the model's device KV caches.
     * Copies CPU data back to device.  The snapshot is NOT freed — call
     * llaisysQwen2KVSnapshotDestroy when no longer needed.
     */
    __export void llaisysQwen2ModelLoadKV(struct LlaisysQwen2Model *model, LlaisysQwen2KVSnapshot_t snapshot);

    /**
     * Get the stored position (number of tokens) in a snapshot.
     */
    __export size_t llaisysQwen2KVSnapshotGetPos(LlaisysQwen2KVSnapshot_t snapshot);

    /** Free a KV cache snapshot and its CPU memory. */
    __export void llaisysQwen2KVSnapshotDestroy(LlaisysQwen2KVSnapshot_t snapshot);
}
#endif // LLAISYS_MODELS_QWEN2_H
