/**
 * icons.tsx — Custom SVG icon components styled after DeepSeek portal.
 *
 * All icons accept className and size props for consistency.
 */

interface IconProps {
  size?: number;
  className?: string;
}

/** LLAISYS brain/circuit logo mark. */
export function LogoIcon({ size = 24, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Outer circle */}
      <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="1.5" opacity="0.3" />
      {/* Neural network nodes */}
      <circle cx="16" cy="8" r="2.5" fill="currentColor" />
      <circle cx="9" cy="20" r="2.5" fill="currentColor" />
      <circle cx="23" cy="20" r="2.5" fill="currentColor" />
      <circle cx="16" cy="16" r="3" fill="currentColor" opacity="0.6" />
      {/* Connections */}
      <line x1="16" y1="10.5" x2="16" y2="13" stroke="currentColor" strokeWidth="1.5" />
      <line x1="13.5" y1="17.5" x2="11" y2="18.5" stroke="currentColor" strokeWidth="1.5" />
      <line x1="18.5" y1="17.5" x2="21" y2="18.5" stroke="currentColor" strokeWidth="1.5" />
      {/* Accent sparkle */}
      <path d="M24 6l1 2 2 1-2 1-1 2-1-2-2-1 2-1z" fill="currentColor" opacity="0.5" />
    </svg>
  );
}

/** New chat / compose icon — pen on paper. */
export function NewChatIcon({ size = 18, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  );
}

/** Chat bubble icon for conversation list items. */
export function ChatBubbleIcon({ size = 16, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

/** Trash / delete icon. */
export function TrashIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

/** Sidebar collapse icon (chevrons left). */
export function SidebarCollapseIcon({ size = 18, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
      <path d="M14 9l-3 3 3 3" />
    </svg>
  );
}

/** Sidebar expand icon (chevrons right). */
export function SidebarExpandIcon({ size = 18, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
      <path d="M14 9l3 3-3 3" />
    </svg>
  );
}

/** Send / arrow-up icon for submit button. */
export function SendIcon({ size = 18, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 19V5" />
      <path d="M5 12l7-7 7 7" />
    </svg>
  );
}

/** Stop / square icon for aborting stream. */
export function StopIcon({ size = 18, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
    >
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

/** Pencil / edit icon. */
export function EditIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

/** Regenerate / refresh icon. */
export function RegenerateIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M1 4v6h6" />
      <path d="M23 20v-6h-6" />
      <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10" />
      <path d="M3.51 15A9 9 0 0 0 18.36 18.36L23 14" />
    </svg>
  );
}

/** Check mark icon for confirm. */
export function CheckIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

/** X / close icon for cancel. */
export function CloseIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

/** User avatar circle. */
export function UserAvatarIcon({ size = 28, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
    >
      <circle cx="16" cy="16" r="16" fill="currentColor" opacity="0.15" />
      <circle cx="16" cy="13" r="5" fill="currentColor" opacity="0.6" />
      <path
        d="M6 27c0-5.523 4.477-10 10-10s10 4.477 10 10"
        fill="currentColor"
        opacity="0.4"
      />
    </svg>
  );
}

/** AI / bot avatar — sparkle brain icon. */
export function AiAvatarIcon({ size = 28, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
    >
      <circle cx="16" cy="16" r="16" fill="currentColor" opacity="0.1" />
      <circle cx="16" cy="16" r="7" stroke="currentColor" strokeWidth="1.5" opacity="0.5" />
      <circle cx="16" cy="13" r="1.5" fill="currentColor" />
      <circle cx="13" cy="18" r="1.5" fill="currentColor" />
      <circle cx="19" cy="18" r="1.5" fill="currentColor" />
      <line x1="16" y1="14.5" x2="14" y2="17" stroke="currentColor" strokeWidth="1" />
      <line x1="16" y1="14.5" x2="18" y2="17" stroke="currentColor" strokeWidth="1" />
      {/* Sparkles */}
      <path d="M24 7l.8 1.6 1.6.8-1.6.8-.8 1.6-.8-1.6-1.6-.8 1.6-.8z" fill="currentColor" opacity="0.6" />
      <path d="M7 9l.5 1 1 .5-1 .5-.5 1-.5-1-1-.5 1-.5z" fill="currentColor" opacity="0.4" />
    </svg>
  );
}

/** Copy icon for copying messages. */
export function CopyIcon({ size = 14, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

/** Sparkle / magic wand — used in welcome screen. */
export function SparkleIcon({ size = 20, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
    >
      <path d="M12 2l2.4 4.8 4.8 2.4-4.8 2.4L12 16.4l-2.4-4.8L4.8 9.2l4.8-2.4z" opacity="0.9" />
      <path d="M19 14l1.2 2.4 2.4 1.2-2.4 1.2L19 21.2l-1.2-2.4-2.4-1.2 2.4-1.2z" opacity="0.5" />
      <path d="M5 2l.8 1.6 1.6.8-1.6.8L5 6.8l-.8-1.6L2.6 4.4l1.6-.8z" opacity="0.4" />
    </svg>
  );
}
