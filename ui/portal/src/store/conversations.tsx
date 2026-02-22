/**
 * store/conversations.tsx — Conversation state management via React Context + useState.
 *
 * Exposes plain methods instead of action/dispatch/reducer boilerplate.
 * All mutation methods are stable callbacks (safe for dependency arrays).
 *
 * Provides:
 *  - CRUD for conversations
 *  - Add / edit / delete messages
 *  - LocalStorage persistence
 *  - Active conversation tracking
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";

import type {
  Conversation,
  ConversationId,
  MessageId,
} from "../types";
import { newConversationId, newMessageId } from "../types";

// ── Persisted state shape ──────────────────────────────────────────────────

interface StoreState {
  conversations: Conversation[];
  activeId: ConversationId | null;
}

const STORAGE_KEY = "llaisys-portal-conversations";
const ACTIVE_KEY = "llaisys-portal-active-id";

function loadState(): StoreState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const activeId = (localStorage.getItem(ACTIVE_KEY) ?? null) as ConversationId | null;
    if (raw) {
      const conversations = JSON.parse(raw) as Conversation[];
      return { conversations, activeId };
    }
  } catch {
    // Corrupted storage — start fresh.
  }
  return { conversations: [], activeId: null };
}

function saveState(state: StoreState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
    if (state.activeId) {
      localStorage.setItem(ACTIVE_KEY, state.activeId);
    } else {
      localStorage.removeItem(ACTIVE_KEY);
    }
  } catch {
    // Storage full — silently ignore.
  }
}

// ── Public API exposed by useConversations() ───────────────────────────────

export interface ConversationStore {
  /** Current list of conversations (newest first). */
  conversations: Conversation[];
  /** Id of the currently-active conversation (or null). */
  activeId: ConversationId | null;
  /** Currently active conversation object (convenience derived value). */
  activeConversation: Conversation | null;

  /** Create a new empty conversation and make it active. Returns its id. */
  createConversation: () => ConversationId;
  /** Switch to an existing conversation (or null to deselect). */
  setActive: (id: ConversationId | null) => void;
  /** Delete a conversation. If it was active, activates the next one. */
  deleteConversation: (id: ConversationId) => void;
  /** Rename a conversation. */
  renameConversation: (id: ConversationId, title: string) => void;

  /** Add a user message. Returns its id. */
  addUserMessage: (convId: ConversationId, content: string) => MessageId;
  /** Add an empty assistant placeholder message. Returns its id. */
  addAssistantMessage: (convId: ConversationId) => MessageId;
  /** Replace a message's content entirely. */
  updateMessage: (convId: ConversationId, msgId: MessageId, content: string) => void;
  /** Append a streaming token to an existing message. */
  appendToken: (convId: ConversationId, msgId: MessageId, token: string) => void;
  /** Delete a message and everything after it in the conversation. */
  deleteMessagesFrom: (convId: ConversationId, msgId: MessageId) => void;
}

// ── Context ────────────────────────────────────────────────────────────────

const ConversationContext = createContext<ConversationStore | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<StoreState>(loadState);

  // Persist on every state change.
  useEffect(() => {
    saveState(state);
  }, [state]);

  // ── Helpers that update a single conversation by id ────────────────────

  /** Immutably update one conversation that matches `id`. */
  const updateConv = useCallback(
    (id: ConversationId, fn: (c: Conversation) => Conversation) => {
      setState((prev) => ({
        ...prev,
        conversations: prev.conversations.map((c) => (c.id === id ? fn(c) : c)),
      }));
    },
    [],
  );

  // ── Conversation CRUD ──────────────────────────────────────────────────

  const createConversation = useCallback((): ConversationId => {
    const id = newConversationId();
    const now = Date.now();
    const conv: Conversation = {
      id,
      title: "New conversation",
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    setState((prev) => ({
      conversations: [conv, ...prev.conversations],
      activeId: id,
    }));
    return id;
  }, []);

  const setActive = useCallback((id: ConversationId | null) => {
    setState((prev) => ({ ...prev, activeId: id }));
  }, []);

  const deleteConversation = useCallback((id: ConversationId) => {
    setState((prev) => {
      const remaining = prev.conversations.filter((c) => c.id !== id);
      const activeId = prev.activeId === id ? (remaining[0]?.id ?? null) : prev.activeId;
      return { conversations: remaining, activeId };
    });
  }, []);

  const renameConversation = useCallback(
    (id: ConversationId, title: string) => {
      updateConv(id, (c) => ({ ...c, title, updatedAt: Date.now() }));
    },
    [updateConv],
  );

  // ── Message operations ─────────────────────────────────────────────────

  const addUserMessage = useCallback(
    (convId: ConversationId, content: string): MessageId => {
      const id = newMessageId();
      updateConv(convId, (c) => ({
        ...c,
        messages: [...c.messages, { id, role: "user" as const, content }],
        updatedAt: Date.now(),
      }));
      return id;
    },
    [updateConv],
  );

  const addAssistantMessage = useCallback(
    (convId: ConversationId): MessageId => {
      const id = newMessageId();
      updateConv(convId, (c) => ({
        ...c,
        messages: [...c.messages, { id, role: "assistant" as const, content: "" }],
        updatedAt: Date.now(),
      }));
      return id;
    },
    [updateConv],
  );

  const updateMessage = useCallback(
    (convId: ConversationId, msgId: MessageId, content: string) => {
      updateConv(convId, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, content } : m)),
        updatedAt: Date.now(),
      }));
    },
    [updateConv],
  );

  const appendToken = useCallback(
    (convId: ConversationId, msgId: MessageId, token: string) => {
      updateConv(convId, (c) => ({
        ...c,
        messages: c.messages.map((m) =>
          m.id === msgId ? { ...m, content: m.content + token } : m,
        ),
        updatedAt: Date.now(),
      }));
    },
    [updateConv],
  );

  const deleteMessagesFrom = useCallback(
    (convId: ConversationId, msgId: MessageId) => {
      updateConv(convId, (c) => {
        const idx = c.messages.findIndex((m) => m.id === msgId);
        if (idx === -1) return c;
        return { ...c, messages: c.messages.slice(0, idx), updatedAt: Date.now() };
      });
    },
    [updateConv],
  );

  // ── Derived ────────────────────────────────────────────────────────────

  const activeConversation =
    state.conversations.find((c) => c.id === state.activeId) ?? null;

  // ── Provide ────────────────────────────────────────────────────────────

  const store: ConversationStore = {
    conversations: state.conversations,
    activeId: state.activeId,
    activeConversation,
    createConversation,
    setActive,
    deleteConversation,
    renameConversation,
    addUserMessage,
    addAssistantMessage,
    updateMessage,
    appendToken,
    deleteMessagesFrom,
  };

  return (
    <ConversationContext.Provider value={store}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversations(): ConversationStore {
  const ctx = useContext(ConversationContext);
  if (ctx === null) {
    throw new Error("useConversations must be used inside <ConversationProvider>");
  }
  return ctx;
}
