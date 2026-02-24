/**
 * api/chat.ts — streaming & non-streaming fetch wrappers for /v1/chat/completions.
 *
 * The Vite dev-server proxies /v1/* → http://localhost:8000 (see vite.config.ts).
 */

import type { ChatCompletionChunk, ChatCompletionRequestBody, ChatCompletionResponse } from "../types";

const ENDPOINT = "/v1/chat/completions";
const CONVERSATIONS_ENDPOINT = "/v1/conversations";
const DEFAULT_MODEL = "deepseek-r1-distill-qwen-1.5b";

// ── Conversation management ────────────────────────────────────────────────

/**
 * Ask the server to create a new conversation and return its id.
 * The returned id must be sent with every chat/completions request
 * in the same thread so the server can match and reuse the KV cache.
 */
export async function createServerConversation(): Promise<string> {
  const res = await fetch(CONVERSATIONS_ENDPOINT, { method: "POST" });
  if (!res.ok) {
    const text = await res.text().catch(() => "unknown error");
    throw new Error(`Failed to create conversation (${res.status}): ${text}`);
  }
  const data = (await res.json()) as { conversation_id: string };
  return data.conversation_id;
}

/**
 * Tell the server to delete a conversation and free its KV-cache memory.
 */
export async function deleteServerConversation(conversationId: string): Promise<void> {
  const res = await fetch(`${CONVERSATIONS_ENDPOINT}/${conversationId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "unknown error");
    console.warn(`Failed to delete server conversation (${res.status}): ${text}`);
  }
}

export interface ChatRequestOptions {
  messages: { role: string; content: string }[];
  model?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  signal?: AbortSignal;
  conversationId: string;
}

// ── Non-streaming ──────────────────────────────────────────────────────────

export async function chatCompletion(opts: ChatRequestOptions): Promise<ChatCompletionResponse> {
  const body: ChatCompletionRequestBody = {
    model: opts.model ?? DEFAULT_MODEL,
    messages: opts.messages,
    max_tokens: opts.maxTokens ?? 4096,
    temperature: opts.temperature ?? 0.8,
    top_p: opts.topP ?? 0.8,
    top_k: opts.topK ?? 50,
    stream: false,
    conversation_id: opts.conversationId,
  };

  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "unknown error");
    throw new Error(`Chat completion failed (${res.status}): ${text}`);
  }

  return (await res.json()) as ChatCompletionResponse;
}

// ── Streaming ──────────────────────────────────────────────────────────────

/**
 * Yields parsed SSE chunks from the streaming chat endpoint.
 * Automatically stops when the server sends `[DONE]`.
 */
export async function* chatCompletionStream(
  opts: ChatRequestOptions,
): AsyncGenerator<ChatCompletionChunk, void, undefined> {
  const body: ChatCompletionRequestBody = {
    model: opts.model ?? DEFAULT_MODEL,
    messages: opts.messages,
    max_tokens: opts.maxTokens ?? 4096,
    temperature: opts.temperature ?? 0.8,
    top_p: opts.topP ?? 0.8,
    top_k: opts.topK ?? 50,
    stream: true,
    conversation_id: opts.conversationId,
  };

  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "unknown error");
    throw new Error(`Chat stream failed (${res.status}): ${text}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Response body is not readable");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last (possibly incomplete) line in the buffer.
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed === "" || trimmed.startsWith(":")) continue; // SSE comment / keep-alive

        // Strip the "data: " prefix that SSE wraps around payloads.
        const payload = trimmed.startsWith("data:") ? trimmed.slice(5).trim() : trimmed;

        if (payload === "[DONE]") return;

        try {
          const chunk = JSON.parse(payload) as ChatCompletionChunk;
          yield chunk;
        } catch {
          // Malformed JSON — skip gracefully.
          console.warn("[chat stream] failed to parse chunk:", payload);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
