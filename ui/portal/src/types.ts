/** Unique branded id helpers — avoids mixing conversation / message ids. */
export type ConversationId = string & { readonly __brand: "ConversationId" };
export type MessageId = string & { readonly __brand: "MessageId" };

export function newConversationId(): ConversationId {
  return crypto.randomUUID() as ConversationId;
}

export function newMessageId(): MessageId {
  return crypto.randomUUID() as MessageId;
}

/** A single chat message (mirrors OpenAI schema). */
export interface ChatMessage {
  id: MessageId;
  role: "system" | "user" | "assistant";
  content: string;
}

/** A conversation is a named list of messages. */
export interface Conversation {
  id: ConversationId;
  title: string;
  messages: ChatMessage[];
  createdAt: number; // epoch ms
  updatedAt: number;
}

// ── OpenAI-compatible request / response shapes ────────────────────────────

export interface ChatCompletionRequestBody {
  model: string;
  messages: { role: string; content: string }[];
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  stream: boolean;
  conversation_id: string;
}

export interface ChatCompletionChoice {
  index: number;
  message: { role: string; content: string };
  finish_reason: string | null;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  conversation_id: string;
}

/** A single SSE chunk from the streaming endpoint. */
export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }[];
  conversation_id: string;
}
