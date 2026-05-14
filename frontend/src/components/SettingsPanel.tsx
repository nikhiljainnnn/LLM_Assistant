/**
 * src/components/SettingsPanel.tsx
 */

import { useEffect, useState } from "react";
import { getHealth, listFineTuneJobs } from "../lib/api";
import type { FineTuneJob, HealthResponse } from "../lib/api";

export function SettingsPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<FineTuneJob[]>([]);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [loadingJobs, setLoadingJobs] = useState(false);

  const fetchHealth = async () => {
    setLoadingHealth(true);
    try { setHealth(await getHealth()); } catch {}
    setLoadingHealth(false);
  };

  const fetchJobs = async () => {
    setLoadingJobs(true);
    try { setJobs(await listFineTuneJobs()); } catch {}
    setLoadingJobs(false);
  };

  useEffect(() => { fetchHealth(); fetchJobs(); }, []);

  return (
    <div className="settings-panel">
      <div className="panel-header">
        <h1 className="panel-title">System</h1>
        <p className="panel-subtitle">Health, providers, and fine-tuning jobs</p>
      </div>

      {/* Health */}
      <section className="settings-section">
        <div className="section-header">
          <h2 className="section-title">Backend Health</h2>
          <button className="refresh-btn" onClick={fetchHealth} disabled={loadingHealth}>
            {loadingHealth ? "…" : "↻"}
          </button>
        </div>
        {health ? (
          <div className="health-grid">
            <HealthRow label="Status" value={health.status} ok={health.status === "ok"} />
            <HealthRow label="Version" value={health.version} />
            <HealthRow label="Vector Store" value={`${health.vector_store_size} vectors`} />
            {Object.entries(health.providers).map(([k, v]) => (
              <HealthRow key={k} label={`Provider: ${k}`} value={v ? "available" : "not configured"} ok={v} />
            ))}
          </div>
        ) : (
          <div className="empty-section">
            {loadingHealth ? "Checking…" : "Could not reach backend"}
          </div>
        )}
      </section>

      {/* Fine-tune Jobs */}
      <section className="settings-section">
        <div className="section-header">
          <h2 className="section-title">Fine-Tune Jobs</h2>
          <button className="refresh-btn" onClick={fetchJobs} disabled={loadingJobs}>
            {loadingJobs ? "…" : "↻"}
          </button>
        </div>
        {jobs.length === 0 ? (
          <div className="empty-section">
            No jobs yet. Submit via <code>POST /api/v1/finetune</code>.
          </div>
        ) : (
          <div className="jobs-list">
            {jobs.map(job => (
              <JobCard key={job.job_id} job={job} />
            ))}
          </div>
        )}
      </section>

      {/* API Reference */}
      <section className="settings-section">
        <h2 className="section-title">API Reference</h2>
        <div className="api-routes">
          {API_ROUTES.map(r => (
            <div key={r.path} className="api-route">
              <span className={`method method--${r.method.toLowerCase()}`}>{r.method}</span>
              <code className="route-path">{r.path}</code>
              <span className="route-desc">{r.desc}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function HealthRow({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="health-row">
      <span className="health-label">{label}</span>
      <span className={`health-value ${ok === true ? "health-value--ok" : ok === false ? "health-value--err" : ""}`}>
        {value}
      </span>
    </div>
  );
}

function JobCard({ job }: { job: FineTuneJob }) {
  const statusColor = {
    pending: "job-status--pending",
    running: "job-status--running",
    completed: "job-status--ok",
    failed: "job-status--err",
  }[job.status] ?? "";

  return (
    <div className="job-card">
      <div className="job-header">
        <code className="job-id">{job.job_id.slice(0, 8)}…</code>
        <span className={`job-status ${statusColor}`}>{job.status}</span>
      </div>
      {job.error && <p className="job-error">{job.error}</p>}
      {Object.keys(job.metrics).length > 0 && (
        <div className="job-metrics">
          {Object.entries(job.metrics).map(([k, v]) => (
            <span key={k} className="metric-chip">{k}: {String(v)}</span>
          ))}
        </div>
      )}
    </div>
  );
}

const API_ROUTES = [
  { method: "POST", path: "/api/v1/chat", desc: "Chat completion" },
  { method: "POST", path: "/api/v1/chat/stream", desc: "SSE streaming chat" },
  { method: "DELETE", path: "/api/v1/chat/{id}", desc: "Clear conversation" },
  { method: "POST", path: "/api/v1/rag/ingest", desc: "Ingest raw text" },
  { method: "POST", path: "/api/v1/rag/ingest/file", desc: "Upload document" },
  { method: "POST", path: "/api/v1/rag/search", desc: "Semantic search" },
  { method: "GET", path: "/api/v1/rag/stats", desc: "Vector store stats" },
  { method: "POST", path: "/api/v1/finetune", desc: "Submit fine-tune job" },
  { method: "GET", path: "/api/v1/finetune", desc: "List jobs" },
  { method: "GET", path: "/health", desc: "Health check" },
];
