/**
 * src/App.tsx
 * ────────────
 * Root component. Renders sidebar + active panel.
 */

import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { RAGPanel } from "./components/RAGPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import { AuthScreen } from "./components/AuthScreen";

export type Panel = "chat" | "rag" | "settings";

export default function App() {
  const [activePanel, setActivePanel] = useState<Panel>("chat");
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(
    !!localStorage.getItem("access_token")
  );

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <AuthScreen onAuthSuccess={() => setIsAuthenticated(true)} />;
  }

  return (
    <div className="app-shell">
      <Sidebar active={activePanel} onChange={setActivePanel} onLogout={handleLogout} />
      <main className="main-content">
        {activePanel === "chat" && <ChatPanel />}
        {activePanel === "rag" && <RAGPanel />}
        {activePanel === "settings" && <SettingsPanel />}
      </main>
      <StatusBar />
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

const NAV_ITEMS: { id: Panel; icon: string; label: string }[] = [
  { id: "chat", icon: "◈", label: "Chat" },
  { id: "rag", icon: "⬡", label: "Knowledge" },
  { id: "settings", icon: "◎", label: "Settings" },
];

function Sidebar({
  active,
  onChange,
  onLogout,
}: {
  active: Panel;
  onChange: (p: Panel) => void;
  onLogout: () => void;
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-icon">◈</span>
        <span className="logo-text">NEXUS</span>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${active === item.id ? "nav-item--active" : ""}`}
            onClick={() => onChange(item.id)}
            title={item.label}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <button
          className="w-full mb-4 flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-error-container/20 text-error hover:bg-error-container/40 transition-colors text-sm"
          onClick={onLogout}
        >
          <span className="material-symbols-outlined text-[18px]">logout</span>
          Logout
        </button>
        <span className="version-badge">v1.0</span>
      </div>
    </aside>
  );
}
