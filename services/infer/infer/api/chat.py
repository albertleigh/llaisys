"""POST /v1/chat/completions — OpenAI-compatible chat endpoint.

Requests are submitted to the :class:`RequestPool` and processed by the
background inference loop.  Both streaming and non-streaming responses
read tokens from the per-request ``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..engine.request import InferRequest, StreamToken
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    CreateConversationResponse,
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


# ── /v1/pool/status ──────────────────────────────────────────────────────────

@router.get("/pool/status")
async def pool_status(request: Request):
    """Return request-pool and scheduler diagnostics."""
    engine = _get_engine(request)
    return engine.pool_stats()

# ── /v1/conversations ────────────────────────────────────────────────────

@router.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation():
    """Create a new conversation and return its id.

    The returned ``conversation_id`` must be sent with every subsequent
    ``/v1/chat/completions`` request in the same conversation thread so
    the server can match and reuse the KV cache.
    """
    return CreateConversationResponse()


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    """Delete a conversation and free its KV-cache memory."""
    engine = _get_engine(request)
    freed = engine.delete_conversation(conversation_id)
    return {"conversation_id": conversation_id, "slots_freed": freed}


# ── /v1/chat/completions ─────────────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = _get_engine(request)
    messages = [m.model_dump() for m in body.messages]
    conversation_id = body.conversation_id

    if body.stream:
        # Create the request up-front so we can cancel it if the
        # client disconnects before generation finishes.
        infer_req = engine.make_request(
            messages,
            max_tokens=body.max_tokens,
            top_k=body.top_k,
            top_p=body.top_p,
            temperature=body.temperature,
            stream=True,
            conversation_id=conversation_id,
        )
        return EventSourceResponse(
            _stream_generator(engine, body, infer_req, request),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────
    infer_req = engine.make_request(
        messages,
        max_tokens=body.max_tokens,
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        conversation_id=conversation_id,
    )

    texts: list[str] = []
    while True:
        token: StreamToken = await infer_req.output.get()
        if token.error:
            raise RuntimeError(token.error)
        texts.append(token.text)
        if token.is_final:
            break

    completion_text = "".join(texts)
    completion_ids = infer_req.generated_ids
    prompt_len = infer_req.prompt_len

    finish_reason = "stop"
    if len(completion_ids) >= body.max_tokens:
        finish_reason = "length"

    return ChatCompletionResponse(
        model=body.model,
        choices=[
            Choice(
                message=ChoiceMessage(content=completion_text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_len,
            completion_tokens=len(completion_ids),
            total_tokens=prompt_len + len(completion_ids),
        ),
        conversation_id=conversation_id,
    )


# ── Streaming generator ─────────────────────────────────────────────────────

async def _stream_generator(
    engine,
    body: ChatCompletionRequest,
    infer_req: InferRequest,
    http_request: Request,
):
    """Yield SSE chunks by reading from the request's output queue.

    If the HTTP client disconnects, the :class:`InferRequest` is
    cancelled so the inference loop can skip remaining work.
    """
    chunk_id = f"chatcmpl-{infer_req.id}"
    conv_id = infer_req.conversation_id

    # First chunk: role announcement.
    first = ChatCompletionChunk(
        id=chunk_id,
        model=body.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        conversation_id=conv_id,
    )
    yield json.dumps(first.model_dump(), ensure_ascii=False)

    # Content chunks — one per token from the pool-backed output queue
    try:
        while True:
            # Check if the HTTP client has disconnected
            if await http_request.is_disconnected():
                infer_req.cancel()
                break

            try:
                token: StreamToken = await asyncio.wait_for(
                    infer_req.output.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # No token yet — loop back and check disconnect
                continue

            if token.error:
                # Emit an error chunk (non-standard but practical)
                break

            chunk = ChatCompletionChunk(
                id=chunk_id,
                model=body.model,
                choices=[
                    StreamChoice(
                        delta=(
                            DeltaMessage(content=token.text)
                            if token.finish_reason is None
                            else DeltaMessage()
                        ),
                        finish_reason=token.finish_reason,
                    )
                ],
                conversation_id=conv_id,
            )
            yield json.dumps(chunk.model_dump(), ensure_ascii=False)

            if token.is_final:
                break
    except asyncio.CancelledError:
        infer_req.cancel()
        raise

    # Terminal marker per OpenAI spec
    yield "[DONE]"
