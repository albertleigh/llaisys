"""Inference engine — wraps llaisys.models.Qwen2 + HuggingFace tokenizer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

import llaisys


# Map CLI device names to llaisys enum values
_DEVICE_MAP = {
    "cpu": llaisys.DeviceType.CPU,
}
# Add nvidia if the enum exists (compiled with --nv-gpu)
if hasattr(llaisys.DeviceType, "NVIDIA"):
    _DEVICE_MAP["nvidia"] = llaisys.DeviceType.NVIDIA


class InferenceEngine:
    """Thin async wrapper around the synchronous llaisys C backend.

    All blocking C calls are dispatched to a thread-pool via
    ``asyncio.to_thread`` so the FastAPI event loop stays responsive.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu", max_ctx_len: int = 2048):
        self.model_path = Path(model_path).resolve()
        self.device_enum = _DEVICE_MAP.get(device, llaisys.DeviceType.CPU)

        # ── Tokenizer (HuggingFace — fast, async-safe) ──────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        # ── Model (llaisys C backend) ───────────────────────────────
        self.model = llaisys.models.Qwen2(
            str(self.model_path),
            self.device_enum,
            max_ctx_len=max_ctx_len,
        )

        self._lock = asyncio.Lock()  # serialise inference (single model instance)

    # ── Tokenisation helpers ─────────────────────────────────────────

    def _apply_chat_template(self, messages: list[dict]) -> list[int]:
        """Convert chat messages → token IDs using the model's chat template."""
        text = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    # ── Synchronous generation (runs in thread) ─────────────────────

    def _generate_sync(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> list[int]:
        """Call the C model — returns full token sequence (prompt + completion)."""
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    # ── Async non-streaming ──────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> tuple[list[int], str, int]:
        """Return (completion_tokens, text, prompt_token_count)."""
        input_ids = self._apply_chat_template(messages)
        prompt_len = len(input_ids)

        async with self._lock:
            all_ids = await asyncio.to_thread(
                self._generate_sync,
                input_ids,
                max_tokens,
                top_k,
                top_p,
                temperature,
            )

        completion_ids = all_ids[prompt_len:]
        text = self.decode(completion_ids)
        return completion_ids, text, prompt_len

    # ── Async streaming (token-by-token) ─────────────────────────────

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> AsyncGenerator[tuple[str, str | None], None]:
        """Yield (token_text, finish_reason | None) one token at a time.

        For now this wraps the full-sequence generation and replays it
        token-by-token. Once the C backend exposes a step-level API we
        can swap in true streaming without changing the HTTP layer.
        """
        input_ids = self._apply_chat_template(messages)
        prompt_len = len(input_ids)

        async with self._lock:
            all_ids = await asyncio.to_thread(
                self._generate_sync,
                input_ids,
                max_tokens,
                top_k,
                top_p,
                temperature,
            )

        completion_ids = all_ids[prompt_len:]
        end_token = self.model.meta.end_token

        for i, tid in enumerate(completion_ids):
            is_last = i == len(completion_ids) - 1
            is_stop = tid == end_token

            token_text = self.decode([tid], skip_special=False)
            if is_stop:
                yield "", "stop"
                return
            elif is_last:
                yield token_text, "length"
                return
            else:
                yield token_text, None

            # Small yield to keep the event loop responsive during replay
            await asyncio.sleep(0)
