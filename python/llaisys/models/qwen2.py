from typing import Sequence, Optional, Dict, Any, List
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, llaisysQwen2Model_t
from ..libllaisys.tensor import llaisysDataType_t, dtype_str_to_enum
from ..tensor import Tensor
from pathlib import Path
import json
import numpy as np
import safetensors.numpy
import ctypes
import torch
import time


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, max_ctx_len: int = 2048):
        self.model_path = Path(model_path)
        self.device = device

        # 1. Read Config
        with open(self.model_path / "config.json", 'r') as f:
            self.config = json.load(f)

        # 2. Populate Meta
        self.meta = LlaisysQwen2Meta()
        # Use the model's native dtype (typically bfloat16) instead of F32
        torch_dtype_str = self.config.get("torch_dtype", "float32")
        dtype_map = {"bfloat16": DataType.BF16, "float16": DataType.F16, "float32": DataType.F32}
        self.meta.dtype = dtype_map.get(torch_dtype_str, DataType.F32).value
        self.meta.nlayer = self.config["num_hidden_layers"]
        self.meta.hs = self.config["hidden_size"]
        self.meta.nh = self.config["num_attention_heads"]
        self.meta.nkvh = self.config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = self.config["intermediate_size"]
        # Cap KV cache size to avoid exceeding GPU memory
        model_maxseq = self.config["max_position_embeddings"]
        self.meta.maxseq = min(model_maxseq, max_ctx_len)
        self.meta.voc = self.config["vocab_size"]
        self.meta.epsilon = self.config["rms_norm_eps"]
        self.meta.theta = self.config.get("rope_theta", 1000000.0)
        self.meta.end_token = self.config.get("eos_token_id", 151643)

        # 3. Create model instance using C API
        self.model_handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            self.device.value,
            None,
            0
        )

        if not self.model_handle:
            raise RuntimeError("Failed to create Qwen2 model backend")

        # 4. Load weights from safetensors files into C++ memory
        self._load_weights()

    @staticmethod
    def _torch_dtype_to_llaisys(dtype: torch.dtype) -> DataType:
        if dtype == torch.float16:
            return DataType.F16
        if dtype == torch.float32:
            return DataType.F32
        if dtype == torch.float64:
            return DataType.F64
        if dtype == torch.bfloat16:
            return DataType.BF16
        if dtype == torch.int64:
            return DataType.I64
        if dtype == torch.int32:
            return DataType.I32
        if dtype == torch.int16:
            return DataType.I16
        if dtype == torch.int8:
            return DataType.I8
        if dtype == torch.uint8:
            return DataType.U8
        if dtype == torch.bool:
            return DataType.BOOL
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    @staticmethod
    def _torch_to_llaisys_tensor(tensor: torch.Tensor, device: DeviceType) -> Tensor:
        cpu_tensor = tensor.detach().cpu().contiguous()

        # Keep native dtype (typically bfloat16) — do NOT convert to float32
        # The C++ CUDA kernels support BF16 natively via tensor cores
        # if cpu_tensor.is_floating_point():
        #     cpu_tensor = cpu_tensor.to(torch.float32)

        if torch.isnan(cpu_tensor).any():
            print(f"Warning: Computed tensor contains NaN! dtype={cpu_tensor.dtype}")

        llaisys_dtype = Qwen2._torch_dtype_to_llaisys(cpu_tensor.dtype)

        # 1. create tensor handler using C API
        shape = tuple(cpu_tensor.shape)
        c_shape = (ctypes.c_size_t * len(shape))(*shape)

        handle = LIB_LLAISYS.tensorCreate(
            c_shape,
            len(shape),
            llaisys_dtype.value if hasattr(llaisys_dtype, 'value') else llaisys_dtype,
            device.value if hasattr(device, 'value') else device,
            0
        )
        if not handle:
            raise RuntimeError("Failed to allocate tensor")

        # 2. New Tensor instance
        llaisys_tensor = Tensor(shape)
        llaisys_tensor.handle = handle
        llaisys_tensor.dtype = llaisys_dtype

        # 3. Load tensor data into Tensor instance
        if cpu_tensor.dtype == torch.bfloat16:
            LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(cpu_tensor.data_ptr()))
        else:
            LIB_LLAISYS.tensorLoad(handle, cpu_tensor.numpy().ctypes.data_as(ctypes.c_void_p))

        return llaisys_tensor

    def _load_weights(self):
        weights_c = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_handle).contents

        # 构建索引 (key -> filename)
        key_to_file = {}
        st_files = sorted(list(self.model_path.glob("*.safetensors")))
        for f in st_files:
            with safetensors.numpy.safe_open(f, framework="numpy") as open_f:
                for k in open_f.keys():
                    key_to_file[k] = f

        def load(name):
            if name not in key_to_file:
                print(f"Warn: {name} missing")
                return None

            f_path = key_to_file[name]

            with safetensors.safe_open(f_path, framework="pt") as f:
                pt_tensor = f.get_tensor(name)

            t = self._torch_to_llaisys_tensor(pt_tensor, self.device)
            return t.handle

        # Assign weights
        weights_c.in_embed = load("model.embed_tokens.weight")
        weights_c.out_embed = load("lm_head.weight") if "lm_head.weight" in key_to_file else weights_c.in_embed
        weights_c.out_norm_w = load("model.norm.weight")

        for i in range(self.meta.nlayer):
            p = f"model.layers.{i}"
            weights_c.attn_norm_w[i] = load(f"{p}.input_layernorm.weight")
            weights_c.attn_q_w[i] = load(f"{p}.self_attn.q_proj.weight")
            weights_c.attn_q_b[i] = load(f"{p}.self_attn.q_proj.bias")
            weights_c.attn_k_w[i] = load(f"{p}.self_attn.k_proj.weight")
            weights_c.attn_k_b[i] = load(f"{p}.self_attn.k_proj.bias")
            weights_c.attn_v_w[i] = load(f"{p}.self_attn.v_proj.weight")
            weights_c.attn_v_b[i] = load(f"{p}.self_attn.v_proj.bias")
            weights_c.attn_o_w[i] = load(f"{p}.self_attn.o_proj.weight")

            weights_c.mlp_norm_w[i] = load(f"{p}.post_attention_layernorm.weight")
            weights_c.mlp_gate_w[i] = load(f"{p}.mlp.gate_proj.weight")
            weights_c.mlp_up_w[i] = load(f"{p}.mlp.up_proj.weight")
            weights_c.mlp_down_w[i] = load(f"{p}.mlp.down_proj.weight")

        # print("=== Verifying Weights in C++ Memory ===")
        # try:
        #     LIB_LLAISYS.tensorGetData.argtypes = [ctypes.c_void_p]
        #     LIB_LLAISYS.tensorGetData.restype = ctypes.c_void_p

        #     ptr = LIB_LLAISYS.tensorGetData(weights_c.in_embed)
        #     if not ptr:
        #         print("FATAL: tensorGetData returned NULL!")
        #     else:
        #         arr = (ctypes.c_float * 10).from_address(ptr)
        #         debug_vals = list(arr)
        #         print(f"Embed Weight [0:10]: {[f'{x:.4e}' for x in debug_vals]}")

        #         if all(x == 0 for x in debug_vals):
        #             print("!!! CRITICAL FAILURE: Weights are ALL ZERO in C++ memory !!!")
        #         else:
        #             print("Weights look OK.")
        # except Exception as e:
        #     print(f"Verification failed: {e}")

        # try:
        #     def _ptr_val(x):
        #         try: return int(x)
        #         except: return None
        #     print("weight pointers:",
        #           "in_embed=", _ptr_val(weights_c.in_embed),
        #           "out_norm_w=", _ptr_val(weights_c.out_norm_w))
        # except:
        #     pass

    def __del__(self):
        if hasattr(self, 'model_handle') and self.model_handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_handle)

    # --- KV Cache API ---

    def get_pos(self) -> int:
        """Get the current sequence position (number of tokens processed so far)."""
        return LIB_LLAISYS.llaisysQwen2ModelGetPos(self.model_handle)

    def set_pos(self, pos: int):
        """Set the current sequence position."""
        LIB_LLAISYS.llaisysQwen2ModelSetPos(self.model_handle, ctypes.c_size_t(pos))

    def reset_kv_cache(self):
        """Reset all KV caches to zero and reset position to 0."""
        LIB_LLAISYS.llaisysQwen2ModelResetKVCache(self.model_handle)

    def save_kv_state(self):
        """Snapshot the current KV cache to CPU (host) memory.

        Returns an opaque snapshot handle (``c_void_p``).  The model's
        device KV caches remain valid — call ``reset_kv_cache()``
        afterwards to free device memory.

        The snapshot must eventually be freed with ``free_kv_snapshot()``.
        """
        return LIB_LLAISYS.llaisysQwen2ModelSaveKV(self.model_handle)

    def restore_kv_state(self, snapshot):
        """Restore a snapshot (host memory) back into the model's device KV caches.

        Parameters
        ----------
        snapshot
            An opaque handle returned by ``save_kv_state()``.
        """
        LIB_LLAISYS.llaisysQwen2ModelLoadKV(self.model_handle, snapshot)

    def free_kv_snapshot(self, snapshot):
        """Free the CPU memory held by a KV cache snapshot.

        After this call the *snapshot* handle is invalid.
        """
        if snapshot:
            LIB_LLAISYS.llaisysQwen2KVSnapshotDestroy(snapshot)

    def get_snapshot_pos(self, snapshot) -> int:
        """Return the token position stored inside a snapshot."""
        if not snapshot:
            return 0
        return LIB_LLAISYS.llaisysQwen2KVSnapshotGetPos(snapshot)

    def generate(
            self,
            inputs: Sequence[int],
            max_new_tokens: int = None,
            top_k: int = 1,
            top_p: float = 0.8,
            temperature: float = 0.8,
    ):
        if not self.model_handle:
            raise RuntimeError("Model is not initialized")

        if max_new_tokens is None:
            max_new_tokens = 1

        if not isinstance(inputs, Sequence) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty sequence of token ids")

        tokens = list(int(t) for t in inputs)
        return self._infer_dialog(tokens, max_new_tokens)

    def infer_step(self, tokens: Sequence[int]) -> int:
        """Run a single inference step.

        Parameters
        ----------
        tokens:
            On the *first* call pass the full prompt token IDs.
            On subsequent calls pass a single-element list containing
            the last generated token (the KV cache retains context).

        Returns
        -------
        int
            The next predicted token ID.
        """
        in_len = len(tokens)
        c_in_buf = (ctypes.c_int64 * in_len)(*tokens)
        buf_ptr = ctypes.cast(c_in_buf, ctypes.POINTER(ctypes.c_int64))
        # print(f"[model:qwen2] input: {in_len}", flush=True)
        # t0 = time.time()
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model_handle,
            buf_ptr,
            ctypes.c_size_t(in_len),
        )
        # dt = time.time() - t0
        # print(f"[model:qwen2] generate eclipse: {dt * 1e3} ms", flush=True)
        return next_token

    def _infer_dialog(self, tokens: Sequence[int], max_steps: int) -> List[int]:
        if max_steps is None:
            max_steps = 1

        full_response = list(tokens)

        # tokens fed to the model, aka Prompt
        next_input_tokens = list(tokens)

        print(f"[infer] Starting inference. Max steps: {max_steps}", flush=True)
        print(f"[Debug] Vocab Size: {self.meta.voc}")

        for step in range(max_steps):
            next_token = self.infer_step(next_input_tokens)

            full_response.append(next_token)

            if next_token == self.meta.end_token:
                print(f"[infer] Stop token reached (token={next_token}).")
                break

            next_input_tokens = [next_token]

        return full_response
