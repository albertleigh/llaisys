/**
 * icons.tsx — Custom SVG icon components styled after DeepSeek portal.
 *
 * All icons accept className and size props for consistency.
 */

interface IconProps {
  size?: number;
  className?: string;
}

/** LLAISYS mascot logo mark. */
export function LogoIcon({ size = 24, className }: IconProps) {
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
      <path d="M20.725 6.166a2.42 2.42 0 0 0-1.771.786c-.216-1.023-.859-1.324-1.593-1.665c-.465-.216-.659-.696-.722-1.043c-.024-.133-.133-.24-.268-.24c-.134 0-.262.055-.326.173c-.14.25-.354.8-.373 1.81c-.029 1.503 1.21 2.662 1.834 3.053c-.064.37-.29.942-.395 1.182a4.9 4.9 0 0 1-1.87-1.234c-.958-1.043-1.738-1.781-2.756-2.503s-.337-1.583.09-1.788s.103-.415-.962-.379c-.853.029-2.067.53-2.567.777c-.51-.162-1.572-.194-2.038-.19C2.425 4.905 1 8.98 1 11c0 6.086 4.873 9 8.373 9c3.958 0 5.345-1.614 5.345-1.614c.164.101.76.316 1.838.362c1.349.057 1.851-.324 1.89-.617s-.179-.4-.37-.49c-.19-.089-.49-.26-1.055-.445c-.453-.147-.657-.308-.702-.37c2.73-2.472 3.23-5.935 3.153-7.407c2.112-.082 2.943-1.488 3.217-2.2c.28-.726.454-1.716.164-1.94c-.232-.18-.426.036-.494.167c-.372.396-.644.719-1.635.719" />
      <path d="M12 10.568s.876-.27 1.645.255c1.041.71 1.355 1.676 1.355 1.676m-1.5 4s-1.041-.507-2.604-2.539c-1.878-2.44-3.647-5.074-7.367-4.213c0 0-.029 5.25 4.971 6.752" />
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

/** AI / bot avatar — mascot icon with circular background. */
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
      <g transform="translate(4 4) scale(1)">
        <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none">
          <path d="M20.725 6.166a2.42 2.42 0 0 0-1.771.786c-.216-1.023-.859-1.324-1.593-1.665c-.465-.216-.659-.696-.722-1.043c-.024-.133-.133-.24-.268-.24c-.134 0-.262.055-.326.173c-.14.25-.354.8-.373 1.81c-.029 1.503 1.21 2.662 1.834 3.053c-.064.37-.29.942-.395 1.182a4.9 4.9 0 0 1-1.87-1.234c-.958-1.043-1.738-1.781-2.756-2.503s-.337-1.583.09-1.788s.103-.415-.962-.379c-.853.029-2.067.53-2.567.777c-.51-.162-1.572-.194-2.038-.19C2.425 4.905 1 8.98 1 11c0 6.086 4.873 9 8.373 9c3.958 0 5.345-1.614 5.345-1.614c.164.101.76.316 1.838.362c1.349.057 1.851-.324 1.89-.617s-.179-.4-.37-.49c-.19-.089-.49-.26-1.055-.445c-.453-.147-.657-.308-.702-.37c2.73-2.472 3.23-5.935 3.153-7.407c2.112-.082 2.943-1.488 3.217-2.2c.28-.726.454-1.716.164-1.94c-.232-.18-.426.036-.494.167c-.372.396-.644.719-1.635.719" />
          <path d="M12 10.568s.876-.27 1.645.255c1.041.71 1.355 1.676 1.355 1.676m-1.5 4s-1.041-.507-2.604-2.539c-1.878-2.44-3.647-5.074-7.367-4.213c0 0-.029 5.25 4.971 6.752" />
        </g>
      </g>
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
