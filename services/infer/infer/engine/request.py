"""Data classes for inference requests flowing through the pool."""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field


class RequestStatus(enum.Enum):
    """Lifecycle states of an inference request."""

    QUEUED = "queued"  # Sitting in the request pool
    ACTIVE = "active"  # Currently being processed by the inference loop
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class StreamToken:
    """A single output token delivered from the inference engine.

    Protocol
    --------
    * ``finish_reason is None`` → more tokens to come.
    * ``finish_reason == "stop"`` → model hit end-of-sequence.
    * ``finish_reason == "length"`` → hit ``max_tokens`` limit.
    * ``error is not None`` → an exception occurred; ``error`` carries the
      message and the request should be considered failed.
    """

    text: str = ""
    finish_reason: str | None = None  # None ⇒ not final
    error: str | None = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None or self.error is not None


@dataclass
class InferRequest:
    """Represents a single inference request throughout its lifecycle.

    After submission, the caller reads from :pyattr:`output` (an
    ``asyncio.Queue[StreamToken]``) to receive generated tokens one by
    one.  The last token will have :pyattr:`StreamToken.is_final` set.
    """

    # ── Identity ─────────────────────────────────────────────────────
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    conversation_id: str = ""

    # ── Input ────────────────────────────────────────────────────────
    input_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0
    max_tokens: int = 256
    top_k: int = 50
    top_p: float = 0.8
    temperature: float = 0.8
    stream: bool = False

    # ── State tracking ───────────────────────────────────────────────
    generated_ids: list[int] = field(default_factory=list)
    status: RequestStatus = field(default=RequestStatus.QUEUED)
    finish_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False

    # ── KV cache slot binding ────────────────────────────────────────
    # Set by the engine when a KV cache slot is acquired for this request.
    kv_slot: object | None = None  # KVSlot (avoid circular import)
    prefix_len: int = 0  # Number of prompt tokens already in KV cache

    # ── Result delivery channel ──────────────────────────────────────
    # Consumers ``await output.get()`` to receive StreamToken objects.
    output: asyncio.Queue = field(default_factory=asyncio.Queue)

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED)

    @property
    def tokens_generated(self) -> int:
        return len(self.generated_ids)

    def cancel(self) -> None:
        """Signal the engine to stop generating for this request."""
        self.cancelled = True

    def __repr__(self) -> str:
        return (
            f"InferRequest(id={self.id!r}, status={self.status.value}, "
            f"prompt_len={self.prompt_len}, generated={self.tokens_generated})"
        )
