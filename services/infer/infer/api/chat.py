"""POST /v1/chat/completions — OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelList,
    ModelObject,
    StreamChoice,
    Usage,
)
from ..config import settings

router = APIRouter(prefix="/v1")


def _get_engine(request: Request):
    return request.app.state.engine


# ── /v1/models ───────────────────────────────────────────────────────────────

@router.get("/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelObject(id=settings.model_name)])


# ── /v1/chat/completions ─────────────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = _get_engine(request)
    messages = [m.model_dump() for m in body.messages]

    if body.stream:
        return EventSourceResponse(
            _stream_generator(engine, body, messages),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────
    completion_ids, text, prompt_len = await engine.generate(
        messages,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
    )

    finish_reason = "stop"  # model stopped naturally or hit end token
    if len(completion_ids) >= body.max_tokens:
        finish_reason = "length"

    return ChatCompletionResponse(
        model=body.model,
        choices=[
            Choice(
                message=ChoiceMessage(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_len,
            completion_tokens=len(completion_ids),
            total_tokens=prompt_len + len(completion_ids),
        ),
    )


# ── Streaming generator ─────────────────────────────────────────────────────

async def _stream_generator(engine, body: ChatCompletionRequest, messages: list[dict]):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # First chunk: role announcement
    first = ChatCompletionChunk(
        id=chunk_id,
        model=body.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield json.dumps(first.model_dump(), ensure_ascii=False)

    # Content chunks — one per token
    async for token_text, finish_reason in engine.generate_stream(
        messages,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
    ):
        chunk = ChatCompletionChunk(
            id=chunk_id,
            model=body.model,
            choices=[
                StreamChoice(
                    delta=DeltaMessage(content=token_text) if finish_reason is None else DeltaMessage(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield json.dumps(chunk.model_dump(), ensure_ascii=False)

    # Terminal marker per OpenAI spec
    yield "[DONE]"
