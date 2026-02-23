"""OpenAI-compatible Pydantic schemas for chat completion."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field
import time
import uuid


# ── Request ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r1-distill-qwen-1.5b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = False
    conversation_id: str = Field(
        ...,
        description=(
            "Conversation identifier obtained from POST /v1/conversations. "
            "All messages in the same conversation thread must use the "
            "same conversation_id so the server can reuse the KV cache."
        ),
    )


# ── Non-streaming response ──────────────────────────────────────────────────

class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    conversation_id: str = ""


# ── Streaming (SSE) response chunks ─────────────────────────────────────────

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    conversation_id: str = ""


# ── /v1/models ───────────────────────────────────────────────────────────────

class CreateConversationResponse(BaseModel):
    """Returned by POST /v1/conversations."""
    conversation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description="Unique conversation identifier to be used in subsequent chat requests.",
    )

class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llaisys"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
