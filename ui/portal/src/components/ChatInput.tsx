/**
 * ChatInput — DeepSeek-style message composer with floating input bar.
 */

import { useState, useRef, useEffect } from "react";
import { SendIcon, StopIcon } from "./icons";

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
}

export function ChatInput({ onSend, onStop, disabled = false, isStreaming = false }: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (trimmed.length === 0 || disabled) return;
    onSend(trimmed);
    setText("");
    // Reset height.
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isStreaming) return;
      handleSubmit();
    }
  };

  return (
    <div className="bg-surface px-4 pt-2 pb-4">
      <div className="mx-auto max-w-3xl">
        <div className="relative flex items-end rounded-2xl border border-border bg-surface-alt
                        shadow-lg shadow-black/10 focus-within:border-primary/50 transition-colors">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message LLAISYS…"
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none bg-transparent
                       px-4 py-3.5 text-sm text-text placeholder-text-dim
                       focus:outline-none disabled:opacity-50"
          />
          <div className="flex items-center pr-2 pb-1.5">
            {isStreaming ? (
              <button
                onClick={onStop}
                className="rounded-lg bg-danger/90 p-2 text-white
                           hover:bg-danger transition-colors cursor-pointer"
                title="Stop generating"
              >
                <StopIcon size={16} />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={text.trim().length === 0 || disabled}
                className="rounded-lg bg-primary p-2 text-surface
                           hover:bg-primary-hover transition-colors
                           disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                title="Send message (Enter)"
              >
                <SendIcon size={16} />
              </button>
            )}
          </div>
        </div>
        <p className="mt-2 text-center text-[10px] text-text-dim/60">
          LLAISYS may produce inaccurate information. Enter ↵ to send · Shift+Enter for new line.
        </p>
      </div>
    </div>
  );
}
