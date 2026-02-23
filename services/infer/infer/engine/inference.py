"""Inference engine — wraps llaisys.models.Qwen2 + HuggingFace tokenizer.

This module integrates with :class:`RequestPool` and
:class:`BatchScheduler` so that multiple callers can submit inference
requests concurrently.  A background ``asyncio.Task`` continuously
pulls batches from the pool, runs them through the C backend (one
inference step at a time), and delivers tokens to each caller via
per-request ``asyncio.Queue`` channels.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

import llaisys

from .pool import RequestPool
from .request import InferRequest, RequestStatus, StreamToken
from .scheduler import BatchScheduler

logger = logging.getLogger(__name__)

# Map CLI device names to llaisys enum values
_DEVICE_MAP = {
    "cpu": llaisys.DeviceType.CPU,
}
# Add nvidia if the enum exists (compiled with --nv-gpu)
if hasattr(llaisys.DeviceType, "NVIDIA"):
    _DEVICE_MAP["nvidia"] = llaisys.DeviceType.NVIDIA


class InferenceEngine:
    """Async inference engine backed by a request pool and batch scheduler.

    Lifecycle
    ---------
    1. ``__init__``  — load model + tokenizer, create pool & scheduler.
    2. ``await start()`` — launch the background inference loop.
    3. ``submit(request)`` — enqueue an :class:`InferRequest`.
       The caller then reads from ``request.output`` to get tokens.
    4. ``await stop()``  — cancel the background loop on shutdown.

    The old ``generate()`` / ``generate_stream()`` convenience methods
    are still available — they now go through the pool internally.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        max_ctx_len: int = 2048,
        max_batch_size: int = 1,
        max_pool_size: int = 128,
    ):
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

        # ── Request pool & scheduler ────────────────────────────────
        self.pool = RequestPool(max_size=max_pool_size)
        self.scheduler = BatchScheduler(
            self.pool, max_batch_size=max_batch_size
        )

        # ── Background loop state ──────────────────────────────────
        self._loop_task: asyncio.Task | None = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background inference loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(
            self._inference_loop(), name="inference-loop"
        )
        logger.info(
            "Inference loop started (max_batch=%d)", self.scheduler.max_batch_size
        )

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to finish."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("Inference loop stopped")

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

    # ── Submit a request ─────────────────────────────────────────────

    def submit(self, request: InferRequest) -> InferRequest:
        """Add a request to the pool.

        Returns the same request object whose ``output`` queue the
        caller should read from.

        Raises
        ------
        RuntimeError
            If the pool is full.
        """
        if not self.pool.add(request):
            raise RuntimeError(
                f"Request pool is full ({self.pool._max_size}); "
                "try again later."
            )
        return request

    def make_request(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
        stream: bool = False,
    ) -> InferRequest:
        """Create an :class:`InferRequest` from chat messages and submit it."""
        input_ids = self._apply_chat_template(messages)
        req = InferRequest(
            input_ids=input_ids,
            prompt_len=len(input_ids),
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stream=stream,
        )
        return self.submit(req)

    # ── Background inference loop ────────────────────────────────────

    async def _inference_loop(self) -> None:
        """Continuously pull batches from the pool and process them.

        This is the heart of the serving system.  It runs as a single
        ``asyncio.Task`` for the lifetime of the server.
        """
        logger.info("Inference loop running …")
        try:
            while self._running:
                batch = await self.scheduler.wait_and_form_batch()
                logger.info(
                    "Processing batch of %d request(s)  "
                    "[pool pending=%d, active=%d]",
                    len(batch),
                    self.pool.pending_count,
                    self.pool.active_count,
                )
                await self._process_batch(batch)
        except asyncio.CancelledError:
            logger.info("Inference loop cancelled")
            raise

    async def _process_batch(self, batch: list[InferRequest]) -> None:
        """Process a batch of requests.

        **Current behaviour** (single KV-cache in C backend):
        Each request is processed to completion sequentially.  This is
        safe and correct, and the pool / scheduler infrastructure is
        fully exercised.

        **Future upgrade path** (per-request KV caches + batched matmul):
        Replace the sequential loop with true iteration-level batching::

            while batch:
                tokens = batched_infer_step(batch)   # one step, all reqs
                for req, tok in zip(batch, tokens):
                    deliver(req, tok)
                    if req.is_done: batch.remove(req)
                # Optionally pull new requests from pool between iters
        """
        for request in batch:
            await self._process_single_request(request)

    async def _process_single_request(self, request: InferRequest) -> None:
        """Run step-by-step inference for one request.

        Each C-level inference step is offloaded to a thread so the
        event loop stays responsive.  Tokens are delivered to the
        caller's ``request.output`` queue via ``call_soon_threadsafe``
        so they arrive the instant they are generated (true streaming).
        """
        loop = asyncio.get_running_loop()
        end_token = self.model.meta.end_token
        tokenizer = self.tokenizer  # HuggingFace tokenizer (Rust, thread-safe)

        def _run_generation() -> None:
            """Blocking function executed in the thread pool."""
            next_input = list(request.input_ids)

            for step in range(request.max_tokens):
                # ── Check cancellation ───────────────────────────────
                if request.cancelled:
                    st = StreamToken(text="", finish_reason="stop")
                    loop.call_soon_threadsafe(request.output.put_nowait, st)
                    return

                # ── Single inference step (C backend) ────────────────
                token_id = self.model.infer_step(next_input)
                request.generated_ids.append(token_id)

                is_end = token_id == end_token
                is_last = step == request.max_tokens - 1

                # ── Decode & deliver ─────────────────────────────────
                if is_end:
                    token_text = ""
                    finish = "stop"
                else:
                    token_text = tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    finish = "length" if is_last else None

                st = StreamToken(text=token_text, finish_reason=finish)
                loop.call_soon_threadsafe(request.output.put_nowait, st)

                if is_end or is_last:
                    return

                # Next step feeds only the new token
                next_input = [token_id]

        try:
            await asyncio.to_thread(_run_generation)
            request.finish_reason = (
                "stop"
                if (
                    request.generated_ids
                    and request.generated_ids[-1] == end_token
                )
                else "length"
            )
        except Exception as e:
            logger.exception("Error processing request %s", request.id)
            err = StreamToken(error=str(e))
            try:
                request.output.put_nowait(err)
            except Exception:
                pass
        finally:
            self.pool.mark_completed(request)

    # ── Convenience wrappers (backward-compatible) ───────────────────
    # These go through the pool so they benefit from queuing &
    # scheduling even when called directly.

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> tuple[list[int], str, int]:
        """Non-streaming generation.  Returns ``(completion_ids, text, prompt_len)``."""
        req = self.make_request(
            messages,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stream=False,
        )

        texts: list[str] = []
        while True:
            token: StreamToken = await req.output.get()
            if token.error:
                raise RuntimeError(token.error)
            texts.append(token.text)
            if token.is_final:
                break

        completion_text = "".join(texts)
        return req.generated_ids, completion_text, req.prompt_len

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> AsyncGenerator[tuple[str, str | None], None]:
        """Yield ``(token_text, finish_reason)`` one token at a time."""
        req = self.make_request(
            messages,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stream=True,
        )

        while True:
            token: StreamToken = await req.output.get()
            if token.error:
                raise RuntimeError(token.error)
            yield token.text, token.finish_reason
            if token.is_final:
                return

    # ── Pool diagnostics ─────────────────────────────────────────────

    def pool_stats(self) -> dict:
        """Return a snapshot of pool + scheduler statistics."""
        return {
            **self.pool.stats(),
            "max_batch_size": self.scheduler.max_batch_size,
            "loop_running": self._running,
        }
