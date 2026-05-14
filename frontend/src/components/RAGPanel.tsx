/**
 * src/components/RAGPanel.tsx
 * ────────────────────────────
 * Upload documents, view vector store stats, semantic search.
 */

import { useRef, useState } from "react";
import { getRAGStats, ingestFile, ingestText, semanticSearch } from "../lib/api";
import type { IngestResponse, RAGStats, SearchResult } from "../lib/api";

export function RAGPanel() {
  const [activeTab, setActiveTab] = useState<"upload" | "text" | "search">("upload");

  return (
    <div className="rag-panel">
      <div className="panel-header">
        <h1 className="panel-title">Knowledge Base</h1>
        <p className="panel-subtitle">Manage documents ingested into the FAISS vector store</p>
      </div>

      <StatsCard />

      <div className="tab-bar">
        {(["upload", "text", "search"] as const).map(tab => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? "tab-btn--active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === "upload" ? "⬆ Upload File" : tab === "text" ? "⌨ Paste Text" : "⊛ Search"}
          </button>
        ))}
      </div>

      {activeTab === "upload" && <FileUploadTab />}
      {activeTab === "text" && <TextIngestTab />}
      {activeTab === "search" && <SemanticSearchTab />}
    </div>
  );
}

// ── Stats ─────────────────────────────────────────────────────────────────────

function StatsCard() {
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try { setStats(await getRAGStats()); } catch {}
    setLoading(false);
  };

  return (
    <div className="stats-card">
      <div className="stats-row">
        <div className="stat-item">
          <span className="stat-value">{stats?.total_vectors ?? "—"}</span>
          <span className="stat-label">Vectors</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{stats?.embedding_dim ?? "—"}</span>
          <span className="stat-label">Dimensions</span>
        </div>
        <div className="stat-item">
          <span className="stat-value stat-value--sm">{stats?.store_path ?? "—"}</span>
          <span className="stat-label">Store Path</span>
        </div>
      </div>
      <button className="refresh-btn" onClick={refresh} disabled={loading}>
        {loading ? "Loading…" : "↻ Refresh Stats"}
      </button>
    </div>
  );
}

// ── File Upload ───────────────────────────────────────────────────────────────

function FileUploadTab() {
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await ingestFile(file));
    } catch (e: unknown) {
      setError((e as Error).message);
    }
    setLoading(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="tab-content">
      <div
        className={`drop-zone ${dragging ? "drop-zone--active" : ""} ${loading ? "drop-zone--loading" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          style={{ display: "none" }}
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
        {loading ? (
          <div className="drop-loading">
            <span className="spinner">◌</span>
            <span>Processing…</span>
          </div>
        ) : (
          <>
            <div className="drop-icon">⬡</div>
            <p className="drop-text">Drop a file here or click to browse</p>
            <p className="drop-hint">PDF · DOCX · TXT · Markdown</p>
          </>
        )}
      </div>

      {result && (
        <div className="ingest-result ingest-result--success">
          <span className="result-icon">✓</span>
          <div>
            <strong>{result.source_name}</strong> ingested successfully
            <br />
            <span className="result-meta">
              {result.chunks_added} chunks added · {result.total_vectors} total vectors
            </span>
          </div>
        </div>
      )}
      {error && (
        <div className="ingest-result ingest-result--error">
          <span className="result-icon">✗</span>
          {error}
        </div>
      )}
    </div>
  );
}

// ── Text Ingest ───────────────────────────────────────────────────────────────

function TextIngestTab() {
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await ingestText(text, source || "manual-input"));
    } catch (e: unknown) {
      setError((e as Error).message);
    }
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <input
        className="text-input"
        placeholder="Source name (optional)"
        value={source}
        onChange={e => setSource(e.target.value)}
      />
      <textarea
        className="text-area"
        placeholder="Paste document text here…"
        value={text}
        onChange={e => setText(e.target.value)}
        rows={10}
      />
      <button className="primary-btn" onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? "Ingesting…" : "⬡ Ingest Text"}
      </button>
      {result && (
        <div className="ingest-result ingest-result--success">
          ✓ {result.chunks_added} chunks · {result.total_vectors} total vectors
        </div>
      )}
      {error && <div className="ingest-result ingest-result--error">✗ {error}</div>}
    </div>
  );
}

// ── Semantic Search ───────────────────────────────────────────────────────────

function SemanticSearchTab() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await semanticSearch(query, 5));
    } catch (e: unknown) {
      setError((e as Error).message);
    }
    setLoading(false);
  };

  return (
    <div className="tab-content">
      <div className="search-row">
        <input
          className="text-input search-input"
          placeholder="Search your knowledge base…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSearch()}
        />
        <button className="primary-btn" onClick={handleSearch} disabled={loading || !query.trim()}>
          {loading ? "…" : "⊛"}
        </button>
      </div>

      {error && <div className="ingest-result ingest-result--error">✗ {error}</div>}

      {results && (
        <div className="search-results">
          <p className="results-count">{results.results.length} results for "{results.query}"</p>
          {results.results.map((r, i) => (
            <div key={i} className="search-result-item">
              <div className="result-header">
                <span className="result-source">{r.source}</span>
                <span className="result-score">{(r.score * 100).toFixed(1)}%</span>
              </div>
              <p className="result-text">{r.text.slice(0, 300)}{r.text.length > 300 ? "…" : ""}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
