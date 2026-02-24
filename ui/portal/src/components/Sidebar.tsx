/**
 * Sidebar — DeepSeek-style conversation list + new-chat button.
 */

import { useConversations } from "../store";
import type { ConversationId } from "../types";
import {
  LogoIcon,
  NewChatIcon,
  ChatBubbleIcon,
  TrashIcon,
  SidebarCollapseIcon,
  SidebarExpandIcon,
} from "./icons";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const { conversations, activeId, createConversation, setActive, deleteConversation } = useConversations();

  const handleNew = () => {
    createConversation();
  };

  const handleSelect = (id: ConversationId) => {
    setActive(id);
  };

  const handleDelete = (e: React.MouseEvent, id: ConversationId) => {
    e.stopPropagation();
    deleteConversation(id);
  };

  return (
    <aside
      className={`
        flex flex-col bg-surface-alt
        transition-all duration-200 ease-in-out
        ${isOpen ? "w-[260px]" : "w-0"}
        overflow-hidden shrink-0
      `}
    >
      {/* Logo + brand */}
      <div className="flex items-center gap-2.5 px-4 pt-5 pb-2">
        <LogoIcon size={28} className="text-primary" />
        <span className="text-base font-semibold tracking-tight text-text">LLAISYS</span>
      </div>

      {/* New chat button */}
      <div className="px-3 py-3">
        <button
          onClick={handleNew}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
                     border border-border text-sm font-medium text-text
                     hover:bg-surface-hover active:scale-[0.98] transition-all cursor-pointer"
        >
          <NewChatIcon size={16} />
          New Chat
        </button>
      </div>

      {/* Conversation list */}
      <nav className="flex-1 overflow-y-auto px-2 pb-2">
        <div className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wider text-text-dim">
          Recent
        </div>
        {conversations.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-text-dim">
            No conversations yet
          </p>
        )}
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => handleSelect(conv.id)}
            className={`
              w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-[13px]
              transition-colors cursor-pointer group mb-0.5
              ${
                conv.id === activeId
                  ? "bg-surface-hover text-text"
                  : "text-text-dim hover:bg-surface-hover/60 hover:text-text"
              }
            `}
          >
            <ChatBubbleIcon size={14} className="shrink-0 opacity-50" />
            <span className="flex-1 truncate">{conv.title}</span>
            <span
              onClick={(e) => handleDelete(e, conv.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter") handleDelete(e as unknown as React.MouseEvent, conv.id); }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded-md
                         hover:text-danger hover:bg-surface transition-all cursor-pointer"
              title="Delete conversation"
            >
              <TrashIcon size={12} />
            </span>
          </button>
        ))}
      </nav>

      {/* Bottom bar — collapse toggle */}
      <div className="px-3 py-3 border-t border-border/50">
        <button
          onClick={onToggle}
          className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs text-text-dim
                     hover:bg-surface-hover hover:text-text transition-colors cursor-pointer w-full"
        >
          <SidebarCollapseIcon size={16} />
          <span>Collapse</span>
        </button>
      </div>
    </aside>
  );
}

/** Small floating button when sidebar is collapsed. */
export function SidebarToggleButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="fixed top-3 left-3 z-50 p-2 rounded-xl
                 bg-surface-alt border border-border
                 hover:bg-surface-hover transition-colors text-text-dim cursor-pointer
                 shadow-lg shadow-black/20"
      title="Open sidebar"
    >
      <SidebarExpandIcon size={18} />
    </button>
  );
}
