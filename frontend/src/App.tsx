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

export type Panel = "chat" | "rag" | "settings";

export default function App() {
  const [activePanel, setActivePanel] = useState<Panel>("chat");

  return (
    <div className="app-shell">
      <Sidebar active={activePanel} onChange={setActivePanel} />
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
}: {
  active: Panel;
  onChange: (p: Panel) => void;
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
        <span className="version-badge">v1.0</span>
      </div>
    </aside>
  );
}
