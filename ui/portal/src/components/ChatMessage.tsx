/**
 * ChatMessage — DeepSeek-style message row with avatar + content.
 */

import { useState, useCallback } from "react";
import type { ChatMessage as ChatMessageType, ConversationId } from "../types";
import { useConversations } from "../store";
import {
  UserAvatarIcon,
  AiAvatarIcon,
  EditIcon,
  RegenerateIcon,
  CopyIcon,
  CheckIcon,
  CloseIcon,
} from "./icons";

interface ChatMessageProps {
  message: ChatMessageType;
  convId: ConversationId;
  isStreaming?: boolean;
  onRegenerate?: () => void;
  onEditSubmit?: (newContent: string) => void;
  isLast?: boolean;
}

export function ChatMessageBubble({
  message,
  convId,
  isStreaming = false,
  onRegenerate,
  onEditSubmit,
  isLast = false,
}: ChatMessageProps) {
  const { updateMessage } = useConversations();
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const [copied, setCopied] = useState(false);

  const isUser = message.role === "user";

  const handleEditSave = () => {
    const trimmed = editContent.trim();
    if (trimmed.length === 0) return;
    updateMessage(convId, message.id, trimmed);
    setIsEditing(false);
    onEditSubmit?.(trimmed);
  };

  const handleEditCancel = () => {
    setEditContent(message.content);
    setIsEditing(false);
  };

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available — silently ignore.
    }
  }, [message.content]);

  return (
    <div className={`flex gap-3 py-5 ${isUser ? "" : ""} group`}>
      {/* Avatar */}
      <div className="shrink-0 pt-0.5">
        {isUser ? (
          <UserAvatarIcon size={30} className="text-primary" />
        ) : (
          <AiAvatarIcon size={30} className="text-accent" />
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 min-w-0">
        {/* Role label */}
        <div className="text-xs font-semibold mb-1.5 text-text">
          {isUser ? "You" : "LLAISYS"}
        </div>

        {/* Content or editor */}
        {isEditing ? (
          <div className="flex flex-col gap-2">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              rows={3}
              className="w-full bg-surface-alt border border-border rounded-xl px-4 py-3
                         text-sm text-text focus:outline-none focus:border-primary/50 resize-y
                         transition-colors"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleEditSave();
                }
                if (e.key === "Escape") handleEditCancel();
              }}
              autoFocus
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={handleEditCancel}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                           text-text-dim hover:text-text border border-border
                           hover:bg-surface-hover transition-colors cursor-pointer"
              >
                <CloseIcon size={12} /> Cancel
              </button>
              <button
                onClick={handleEditSave}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                           bg-primary text-surface font-medium hover:bg-primary-hover
                           transition-colors cursor-pointer"
              >
                <CheckIcon size={12} /> Save & Submit
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className={`text-sm leading-relaxed text-text ${isStreaming ? "cursor-blink" : ""}`}>
              <MessageContent text={message.content} />
            </div>

            {/* Action toolbar — appears on hover */}
            {!isStreaming && (
              <div className="flex items-center gap-0.5 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                {/* Copy */}
                <button
                  onClick={() => void handleCopy()}
                  className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px]
                             text-text-dim hover:text-text hover:bg-surface-hover
                             transition-colors cursor-pointer"
                  title="Copy"
                >
                  {copied ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                  <span>{copied ? "Copied" : "Copy"}</span>
                </button>

                {/* Edit (user only) */}
                {isUser && (
                  <button
                    onClick={() => {
                      setEditContent(message.content);
                      setIsEditing(true);
                    }}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px]
                               text-text-dim hover:text-text hover:bg-surface-hover
                               transition-colors cursor-pointer"
                    title="Edit"
                  >
                    <EditIcon size={12} />
                    <span>Edit</span>
                  </button>
                )}

                {/* Regenerate (last assistant only) */}
                {!isUser && isLast && onRegenerate && (
                  <button
                    onClick={onRegenerate}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px]
                               text-text-dim hover:text-text hover:bg-surface-hover
                               transition-colors cursor-pointer"
                    title="Regenerate"
                  >
                    <RegenerateIcon size={12} />
                    <span>Regenerate</span>
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** Simple content renderer. */
function MessageContent({ text }: { text: string }) {
  if (text.length === 0) {
    return (
      <span className="inline-flex items-center gap-1.5 text-text-dim">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
        Thinking…
      </span>
    );
  }
  return (
    <div className="whitespace-pre-wrap break-words">
      {text}
    </div>
  );
}
