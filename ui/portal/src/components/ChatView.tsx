/**
 * ChatView — main chat pane: message list + input.
 *
 * Handles streaming responses via the SSE API client.
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { useOutletContext } from "react-router-dom";

import { useConversations } from "../store";
import { chatCompletionStream } from "../api";
import { ChatMessageBubble } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { LogoIcon, SparkleIcon, SidebarExpandIcon, NewChatIcon } from "./icons";
import type { ConversationId, MessageId } from "../types";

interface OutletCtx {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

const SUGGESTIONS = [
  {
    label: "Explain a concept",
    desc: "Break down a technical topic",
    prompt: "Explain how transformer neural networks work in simple terms.",
  },
  {
    label: "Write some code",
    desc: "Generate a code snippet",
    prompt: "Write a Python function that implements binary search on a sorted list.",
  },
  {
    label: "Debug my code",
    desc: "Find and fix issues",
    prompt: "Help me debug this piece of code — I'm getting unexpected results.",
  },
  {
    label: "Brainstorm ideas",
    desc: "Creative problem solving",
    prompt: "Brainstorm 5 project ideas for learning about distributed systems.",
  },
];

export function ChatView() {
  const {
    conversations,
    activeId,
    activeConversation,
    createConversation,
    renameConversation,
    addUserMessage,
    addAssistantMessage,
    appendToken,
    deleteMessagesFrom,
  } = useConversations();

  const { sidebarOpen, toggleSidebar } = useOutletContext<OutletCtx>();

  const [streamingMsgId, setStreamingMsgId] = useState<MessageId | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll as new tokens arrive.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [activeConversation?.messages]);

  // ── Generate a streaming response ──────────────────────────────────────

  const generate = useCallback(
    async (convId: ConversationId, assistantMsgId: MessageId, messages: { role: string; content: string }[]) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setStreamingMsgId(assistantMsgId);

      try {
        for await (const chunk of chatCompletionStream({
          messages,
          signal: controller.signal,
          conversationId: convId,
        })) {
          const delta = chunk.choices[0]?.delta;
          if (delta?.content) {
            appendToken(convId, assistantMsgId, delta.content);
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // User cancelled — that's fine.
        } else {
          const text = err instanceof Error ? err.message : "Unknown error";
          appendToken(convId, assistantMsgId, `\n\n\u26a0\ufe0f Error: ${text}`);
        }
      } finally {
        setStreamingMsgId(null);
        abortRef.current = null;
      }
    },
    [appendToken],
  );

  // ── Auto-title conversation after first assistant response ─────────────

  const autoTitle = useCallback(
    (convId: ConversationId, userText: string) => {
      const conv = conversations.find((c) => c.id === convId);
      if (conv && conv.title === "New conversation") {
        const title = userText.length > 40 ? userText.slice(0, 40) + "…" : userText;
        renameConversation(convId, title);
      }
    },
    [renameConversation, conversations],
  );

  // ── Send handler ───────────────────────────────────────────────────────

  const handleSend = useCallback(
    async (text: string, overrideConvId?: ConversationId) => {
      let convId = overrideConvId ?? activeId;

      // Create conversation if none is active.
      if (convId === null) {
        convId = await createConversation();
      }

      addUserMessage(convId, text);
      autoTitle(convId, text);

      const assistantId = addAssistantMessage(convId);

      // Build the messages array from the conversation + the new user message.
      const conv = conversations.find((c) => c.id === convId);
      const history = conv ? conv.messages.map((m) => ({ role: m.role, content: m.content })) : [];
      history.push({ role: "user", content: text });

      void generate(convId, assistantId, history);
    },
    [conversations, activeId, createConversation, addUserMessage, addAssistantMessage, autoTitle, generate],
  );

  // ── Stop streaming ────────────────────────────────────────────────────

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // ── Regenerate last assistant message ─────────────────────────────────

  const handleRegenerate = useCallback(() => {
    const conv = activeConversation;
    if (!conv) return;

    // Find the last assistant message and remove it.
    const msgs = conv.messages;
    const lastAssistant = [...msgs].reverse().find((m) => m.role === "assistant");
    if (!lastAssistant) return;

    deleteMessagesFrom(conv.id, lastAssistant.id);

    // Re-generate.
    const newAssistantId = addAssistantMessage(conv.id);

    const history = msgs
      .filter((m) => m.id !== lastAssistant.id)
      .map((m) => ({ role: m.role, content: m.content }));

    void generate(conv.id, newAssistantId, history);
  }, [activeConversation, deleteMessagesFrom, addAssistantMessage, generate]);

  // ── Edit + resubmit handler ───────────────────────────────────────────

  const handleEditSubmit = useCallback(
    (msgId: MessageId, newContent: string) => {
      const conv = activeConversation;
      if (!conv) return;

      // Delete everything after the edited message (the old assistant reply, etc.).
      const idx = conv.messages.findIndex((m) => m.id === msgId);
      if (idx === -1) return;

      // Remove messages after the edited one.
      const followingMsg = conv.messages[idx + 1];
      if (followingMsg) {
        deleteMessagesFrom(conv.id, followingMsg.id);
      }

      // Start a new assistant reply.
      const assistantId = addAssistantMessage(conv.id);

      const history = conv.messages
        .slice(0, idx)
        .map((m) => ({ role: m.role, content: m.content }));
      history.push({ role: "user", content: newContent });

      void generate(conv.id, assistantId, history);
    },
    [activeConversation, deleteMessagesFrom, addAssistantMessage, generate],
  );

  const isStreaming = streamingMsgId !== null;

  // ── Empty state — DeepSeek-style welcome ───────────────────────────────

  if (!activeConversation) {
    return (
      <div className="flex-1 flex flex-col">
        {/* Top bar */}
        <header className="flex items-center gap-3 px-4 py-2.5 shrink-0">
          {!sidebarOpen && (
            <button
              onClick={toggleSidebar}
              className="p-1.5 rounded-lg hover:bg-surface-hover transition-colors text-text-dim cursor-pointer"
            >
              <SidebarExpandIcon size={18} />
            </button>
          )}
        </header>

        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-lg px-4">
            {/* Logo */}
            <div className="flex justify-center mb-5">
              <LogoIcon size={56} className="text-primary" />
            </div>
            <h2 className="text-2xl font-semibold mb-1 text-text">Hi, I'm LLAISYS</h2>
            <p className="text-text-dim text-sm mb-8">
              Your lightweight AI inference assistant. How can I help you today?
            </p>

            {/* Suggestion cards */}
            <div className="grid grid-cols-2 gap-3 mb-8">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s.label}
                  onClick={async () => {
                    const id = await createConversation();
                    // Small delay to let state settle, then send.
                    setTimeout(() => handleSend(s.prompt, id), 0);
                  }}
                  className="flex items-start gap-2.5 p-3.5 rounded-xl border border-border
                             text-left text-sm hover:bg-surface-hover/60 hover:border-text-dim/30
                             transition-all cursor-pointer group"
                >
                  <SparkleIcon size={16} className="text-primary shrink-0 mt-0.5 opacity-60
                                                     group-hover:opacity-100 transition-opacity" />
                  <div>
                    <div className="font-medium text-text text-[13px]">{s.label}</div>
                    <div className="text-text-dim text-xs mt-0.5">{s.desc}</div>
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={() => void createConversation()}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl
                         bg-primary text-surface font-medium text-sm
                         hover:bg-primary-hover transition-colors cursor-pointer"
            >
              <NewChatIcon size={16} />
              Start a conversation
            </button>
          </div>
        </div>

        {/* Input at the bottom even in welcome */}
        <ChatInput onSend={async (text) => {
          const id = await createConversation();
          setTimeout(() => handleSend(text, id), 0);
        }} />
      </div>
    );
  }

  // ── Chat view ─────────────────────────────────────────────────────────

  const messages = activeConversation.messages;

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Top bar */}
      <header className="flex items-center gap-3 px-4 py-2.5 border-b border-border/50 shrink-0 bg-surface">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg hover:bg-surface-hover transition-colors text-text-dim cursor-pointer"
          >
            <SidebarExpandIcon size={18} />
          </button>
        )}
        <LogoIcon size={20} className="text-primary opacity-60" />
        <h1 className="text-sm font-medium truncate text-text">
          {activeConversation.title}
        </h1>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-2">
        <div className="mx-auto max-w-3xl divide-y divide-border/30">
          {messages.length === 0 && (
            <p className="text-center text-text-dim text-sm py-20">
              Send a message to start the conversation.
            </p>
          )}
          {messages.map((msg, i) => (
            <ChatMessageBubble
              key={msg.id}
              message={msg}
              convId={activeConversation.id}
              isStreaming={msg.id === streamingMsgId}
              isLast={i === messages.length - 1}
              onRegenerate={handleRegenerate}
              onEditSubmit={(newContent) => handleEditSubmit(msg.id, newContent)}
            />
          ))}
        </div>
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming}
      />
    </div>
  );
}
