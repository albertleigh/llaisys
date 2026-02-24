/**
 * AppLayout — Main shell with sidebar + content area.
 *
 * ┌────────────┬──────────────────────────────────┐
 * │  Sidebar   │         Content (Outlet)         │
 * │            │                                  │
 * │            │                                  │
 * └────────────┴──────────────────────────────────┘
 */

import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { ConversationProvider } from "../store";

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <ConversationProvider>
      <div className="flex h-full overflow-hidden">
        {/* Sidebar */}
        <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen((o) => !o)} />

        {/* Main content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <Outlet context={{ sidebarOpen, toggleSidebar: () => setSidebarOpen((o) => !o) }} />
        </main>
      </div>
    </ConversationProvider>
  );
}
