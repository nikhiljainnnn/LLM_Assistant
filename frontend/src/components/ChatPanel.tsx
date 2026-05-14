/**
 * src/components/ChatPanel.tsx
 * ─────────────────────────────
 * Main chat interface: message list, input bar, settings drawer.
 */

import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { useChat } from "../hooks/useChat";
import type { ChatMessage } from "../hooks/useChat";
import type { SourceChunk } from "../lib/api";

export function ChatPanel() {
  const { messages, isLoading, settings, setSettings, sendMessage, clearChat, stopStreaming } =
    useChat();
  const [input, setInput] = useState("");
  const [showSources, setShowSources] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-left">
          <h1 className="chat-title">AI Assistant</h1>
          <div className="chat-badges">
            <span className={`badge ${settings.useRAG ? "badge--on" : "badge--off"}`}>
              {settings.useRAG ? "RAG ●" : "RAG ○"}
            </span>
            <span className="badge badge--model">{settings.provider}</span>
          </div>
        </div>
        <div className="chat-header-actions">
          <button
            className="icon-btn"
            onClick={() => setSettings(s => ({ ...s, useRAG: !s.useRAG }))}
            title="Toggle RAG"
          >
            ⬡
          </button>
          <button
            className="icon-btn"
            onClick={() => setSettings(s => ({ ...s, stream: !s.stream }))}
            title="Toggle streaming"
          >
            {settings.stream ? "⟿" : "→"}
          </button>
          <button className="icon-btn icon-btn--danger" onClick={clearChat} title="Clear conversation">
            ⊘
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 && <EmptyState />}
        {messages.map(msg => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onShowSources={() => setShowSources(showSources === msg.id ? null : msg.id)}
            sourcesOpen={showSources === msg.id}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-row">
          <textarea
            className="chat-input"
            placeholder="Ask anything… (Shift+Enter for newline)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          {isLoading ? (
            <button className="send-btn send-btn--stop" onClick={stopStreaming}>
              ⏹
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!input.trim()}
            >
              ▶
            </button>
          )}
        </div>
        <div className="input-meta">
          <span>Temp {settings.temperature}</span>
          <span>·</span>
          <span>Max {settings.maxTokens} tokens</span>
          <span>·</span>
          <span>{settings.stream ? "Streaming" : "Standard"}</span>
        </div>
      </div>
    </div>
  );
}

// ── Message Bubble ────────────────────────────────────────────────────────────

function MessageBubble({
  message,
  onShowSources,
  sourcesOpen,
}: {
  message: ChatMessage;
  onShowSources: () => void;
  sourcesOpen: boolean;
}) {
  return (
    <div className={`message message--${message.role} ${message.error ? "message--error" : ""}`}>
      <div className="message-avatar">
        {message.role === "user" ? "U" : "AI"}
      </div>
      <div className="message-body">
        <div className="message-content">
          {message.role === "assistant" ? (
            <MarkdownContent content={message.content} />
          ) : (
            <p>{message.content}</p>
          )}
          {message.isStreaming && <span className="cursor-blink">▌</span>}
        </div>

        {/* Meta */}
        <div className="message-meta">
          {message.latency_ms != null && (
            <span className="meta-chip">{Math.round(message.latency_ms)}ms</span>
          )}
          {message.sources && message.sources.length > 0 && (
            <button className="meta-chip meta-chip--sources" onClick={onShowSources}>
              {sourcesOpen ? "▲" : "▼"} {message.sources.length} source
              {message.sources.length > 1 ? "s" : ""}
            </button>
          )}
        </div>

        {/* Sources */}
        {sourcesOpen && message.sources && (
          <SourcesPanel sources={message.sources} />
        )}
      </div>
    </div>
  );
}

// ── Sources Panel ─────────────────────────────────────────────────────────────

function SourcesPanel({ sources }: { sources: SourceChunk[] }) {
  return (
    <div className="sources-panel">
      <h4 className="sources-heading">Retrieved Context</h4>
      {sources.map((s, i) => (
        <div key={i} className="source-item">
          <div className="source-header">
            <span className="source-name">{s.source}</span>
            <span className="source-score">{(s.score * 100).toFixed(1)}%</span>
          </div>
          <p className="source-text">{s.text.slice(0, 280)}…</p>
        </div>
      ))}
    </div>
  );
}

// ── Markdown renderer (simple, no heavy deps) ─────────────────────────────────

function MarkdownContent({ content }: { content: string }) {
  // Simple renderer: bold, code blocks, inline code, line breaks
  const lines = content.split("\n");
  const elements: JSX.Element[] = [];
  let inCode = false;
  let codeLines: string[] = [];
  let codeLang = "";

  lines.forEach((line, i) => {
    if (line.startsWith("```")) {
      if (!inCode) {
        inCode = true;
        codeLang = line.slice(3).trim();
        codeLines = [];
      } else {
        elements.push(
          <pre key={i} className="code-block">
            {codeLang && <span className="code-lang">{codeLang}</span>}
            <code>{codeLines.join("\n")}</code>
          </pre>
        );
        inCode = false;
        codeLines = [];
      }
      return;
    }

    if (inCode) {
      codeLines.push(line);
      return;
    }

    if (line.startsWith("### ")) {
      elements.push(<h3 key={i} className="md-h3">{line.slice(4)}</h3>);
    } else if (line.startsWith("## ")) {
      elements.push(<h2 key={i} className="md-h2">{line.slice(3)}</h2>);
    } else if (line.startsWith("# ")) {
      elements.push(<h1 key={i} className="md-h1">{line.slice(2)}</h1>);
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      elements.push(<li key={i} className="md-li">{renderInline(line.slice(2))}</li>);
    } else if (line.trim() === "") {
      elements.push(<br key={i} />);
    } else {
      elements.push(<p key={i} className="md-p">{renderInline(line)}</p>);
    }
  });

  return <div className="markdown">{elements}</div>;
}

function renderInline(text: string): JSX.Element {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("`") && part.endsWith("`"))
          return <code key={i} className="inline-code">{part.slice(1, -1)}</code>;
        if (part.startsWith("**") && part.endsWith("**"))
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "Explain how Retrieval-Augmented Generation works",
  "What is LoRA and how does it reduce fine-tuning costs?",
  "Compare GPT-4 and open-source alternatives",
  "How does FAISS perform similarity search?",
];

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon">◈</div>
      <h2 className="empty-title">NEXUS AI Assistant</h2>
      <p className="empty-subtitle">RAG-powered · Streaming · Multi-model</p>
      <div className="suggestions">
        {SUGGESTIONS.map(s => (
          <div key={s} className="suggestion-chip">{s}</div>
        ))}
      </div>
    </div>
  );
}
