/**
 * src/components/SettingsPanel.tsx
 * ──────────────────────────────────
 * System status + Fine-tune job monitor with:
 *   • Submit form for new LoRA fine-tuning jobs
 *   • Auto-polling every 5s when any job is running/pending
 *   • Rich job cards with status badges, elapsed time, and metrics
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getHealth,
  listFineTuneJobs,
  submitFineTuneJob,
  type FineTuneJob,
  type FineTuneRequest,
  type HealthResponse,
} from "../lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

const ACTIVE_STATUSES = new Set(["pending", "running"]);

function elapsedTime(startedAt: string | null): string {
  if (!startedAt) return "—";
  const elapsed = Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000);
  if (elapsed < 60) return `${elapsed}s`;
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`;
  return `${Math.floor(elapsed / 3600)}h ${Math.floor((elapsed % 3600) / 60)}m`;
}

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { cls: string; dot: string; label: string }> = {
    pending:   { cls: "badge badge--pending",   dot: "◷", label: "Pending" },
    running:   { cls: "badge badge--running",   dot: "⟳", label: "Running" },
    completed: { cls: "badge badge--completed", dot: "✓", label: "Completed" },
    failed:    { cls: "badge badge--failed",    dot: "✕", label: "Failed" },
  };
  const { cls, dot, label } = cfg[status] ?? { cls: "badge", dot: "?", label: status };
  return (
    <span className={cls}>
      <span className={status === "running" ? "spin-icon" : ""}>{dot}</span>
      {label}
    </span>
  );
}

// ── Submit Form ───────────────────────────────────────────────────────────────

const DEFAULT_FORM: FineTuneRequest = {
  dataset_path: "",
  base_model: "",
  epochs: 3,
  batch_size: 4,
  learning_rate: 2e-4,
};

function SubmitJobForm({ onSubmitted }: { onSubmitted: () => void }) {
  const [form, setForm] = useState<FineTuneRequest>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleChange = (key: keyof FineTuneRequest, value: string | number) =>
    setForm(f => ({ ...f, [key]: value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    if (!form.dataset_path.trim()) {
      setError("Dataset path is required.");
      return;
    }
    setSubmitting(true);
    try {
      const resp = await submitFineTuneJob(form);
      setSuccess(`Job submitted! ID: ${resp.job_id.slice(0, 12)}…`);
      setForm(DEFAULT_FORM);
      onSubmitted();
    } catch (err: any) {
      setError(err.message ?? "Submission failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="submit-job-container">
      <button
        className="expand-btn"
        onClick={() => setExpanded(x => !x)}
        aria-expanded={expanded}
      >
        <span className="expand-icon">{expanded ? "▲" : "▼"}</span>
        Submit New Fine-Tune Job
      </button>

      {expanded && (
        <form className="submit-job-form" onSubmit={handleSubmit}>
          <div className="form-row">
            <label className="form-label" htmlFor="dataset_path">
              Dataset Path <span className="required">*</span>
            </label>
            <input
              id="dataset_path"
              className="form-input"
              type="text"
              placeholder="./data/finetune_dataset.jsonl"
              value={form.dataset_path}
              onChange={e => handleChange("dataset_path", e.target.value)}
              required
            />
          </div>

          <div className="form-row">
            <label className="form-label" htmlFor="base_model">Base Model</label>
            <input
              id="base_model"
              className="form-input"
              type="text"
              placeholder="meta-llama/Llama-2-7b-hf (uses default if blank)"
              value={form.base_model ?? ""}
              onChange={e => handleChange("base_model", e.target.value)}
            />
          </div>

          <div className="form-row form-row--3col">
            <div>
              <label className="form-label" htmlFor="epochs">Epochs</label>
              <input
                id="epochs"
                className="form-input form-input--sm"
                type="number"
                min={1}
                max={20}
                value={form.epochs}
                onChange={e => handleChange("epochs", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="form-label" htmlFor="batch_size">Batch Size</label>
              <input
                id="batch_size"
                className="form-input form-input--sm"
                type="number"
                min={1}
                max={64}
                value={form.batch_size}
                onChange={e => handleChange("batch_size", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="form-label" htmlFor="learning_rate">Learning Rate</label>
              <input
                id="learning_rate"
                className="form-input form-input--sm"
                type="number"
                step={1e-5}
                min={1e-6}
                max={0.01}
                value={form.learning_rate}
                onChange={e => handleChange("learning_rate", parseFloat(e.target.value))}
              />
            </div>
          </div>

          {error && <p className="form-error">⚠ {error}</p>}
          {success && <p className="form-success">✓ {success}</p>}

          <button
            className="submit-btn"
            type="submit"
            disabled={submitting}
          >
            {submitting ? "Submitting…" : "🚀 Start Fine-Tuning"}
          </button>
        </form>
      )}
    </div>
  );
}

// ── Job Card ──────────────────────────────────────────────────────────────────

function JobCard({ job }: { job: FineTuneJob }) {
  const hasMetrics = Object.keys(job.metrics ?? {}).length > 0;
  return (
    <div className={`job-card job-card--${job.status}`}>
      <div className="job-card-header">
        <div className="job-id-group">
          <code className="job-id" title={job.job_id}>
            {job.job_id.slice(0, 12)}…
          </code>
        </div>
        <StatusBadge status={job.status} />
      </div>

      <div className="job-card-meta">
        {job.started_at && (
          <span className="meta-item">
            <span className="meta-icon">⏱</span>
            Elapsed: <strong>{elapsedTime(job.started_at)}</strong>
          </span>
        )}
        {job.finished_at && (
          <span className="meta-item">
            <span className="meta-icon">✓</span>
            Finished: <strong>{new Date(job.finished_at).toLocaleTimeString()}</strong>
          </span>
        )}
      </div>

      {job.error && (
        <div className="job-error-box">
          <span className="error-icon">✕</span>
          {job.error}
        </div>
      )}

      {hasMetrics && (
        <div className="job-metrics">
          {Object.entries(job.metrics).map(([k, v]) => (
            <div key={k} className="metric-chip">
              <span className="metric-key">{k.replace(/_/g, " ")}</span>
              <span className="metric-val">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
            </div>
          ))}
        </div>
      )}

      {job.status === "running" && (
        <div className="progress-bar-container">
          <div className="progress-bar-track">
            <div className="progress-bar-fill" />
          </div>
          <span className="progress-label">Training in progress…</span>
        </div>
      )}
    </div>
  );
}

// ── Health Row ────────────────────────────────────────────────────────────────

function HealthRow({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="health-row">
      <span className="health-label">{label}</span>
      <span
        className={`health-value ${
          ok === true ? "health-value--ok" : ok === false ? "health-value--err" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}

// ── Main Panel ────────────────────────────────────────────────────────────────

export function SettingsPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<FineTuneJob[]>([]);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchHealth = async () => {
    setLoadingHealth(true);
    try { setHealth(await getHealth()); } catch {}
    setLoadingHealth(false);
  };

  const fetchJobs = useCallback(async () => {
    setLoadingJobs(true);
    try { setJobs(await listFineTuneJobs()); } catch {}
    setLoadingJobs(false);
  }, []);

  // Auto-poll every 5s when there are active jobs
  useEffect(() => {
    const hasActive = jobs.some(j => ACTIVE_STATUSES.has(j.status));
    if (hasActive && !pollRef.current) {
      pollRef.current = setInterval(fetchJobs, 5000);
    } else if (!hasActive && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobs, fetchJobs]);

  useEffect(() => {
    fetchHealth();
    fetchJobs();
  }, [fetchJobs]);

  const activeJobs = jobs.filter(j => ACTIVE_STATUSES.has(j.status));
  const doneJobs   = jobs.filter(j => !ACTIVE_STATUSES.has(j.status));

  return (
    <div className="settings-panel">
      <div className="panel-header">
        <h1 className="panel-title">System</h1>
        <p className="panel-subtitle">Health, providers, and fine-tuning job monitor</p>
      </div>

      {/* ── Backend Health ─────────────────────────────────── */}
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
              <HealthRow
                key={k}
                label={`Provider: ${k}`}
                value={v ? "available" : "not configured"}
                ok={v}
              />
            ))}
          </div>
        ) : (
          <div className="empty-section">
            {loadingHealth ? "Checking…" : "Could not reach backend"}
          </div>
        )}
      </section>

      {/* ── Fine-tune Jobs ─────────────────────────────────── */}
      <section className="settings-section">
        <div className="section-header">
          <h2 className="section-title">
            Fine-Tune Jobs
            {activeJobs.length > 0 && (
              <span className="active-pill">{activeJobs.length} active</span>
            )}
          </h2>
          <button className="refresh-btn" onClick={fetchJobs} disabled={loadingJobs}>
            {loadingJobs ? "…" : "↻"}
          </button>
        </div>

        {/* Submit form */}
        <SubmitJobForm onSubmitted={fetchJobs} />

        {/* Active jobs */}
        {activeJobs.length > 0 && (
          <div className="jobs-group">
            <p className="jobs-group-label">🔄 In Progress</p>
            <div className="jobs-list">
              {activeJobs.map(job => <JobCard key={job.job_id} job={job} />)}
            </div>
          </div>
        )}

        {/* Completed / failed jobs */}
        {doneJobs.length > 0 && (
          <div className="jobs-group">
            <p className="jobs-group-label">📋 History</p>
            <div className="jobs-list">
              {doneJobs.map(job => <JobCard key={job.job_id} job={job} />)}
            </div>
          </div>
        )}

        {jobs.length === 0 && !loadingJobs && (
          <div className="empty-section">
            No jobs yet. Use the form above to submit your first fine-tuning run.
          </div>
        )}
      </section>

      {/* ── API Reference ──────────────────────────────────── */}
      <section className="settings-section">
        <h2 className="section-title">API Reference</h2>
        <div className="api-routes">
          {API_ROUTES.map(r => (
            <div key={r.path + r.method} className="api-route">
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

// ── API Routes reference data ────────────────────────────────────────────────

const API_ROUTES = [
  { method: "POST",   path: "/api/v1/auth/register",   desc: "Register a new user" },
  { method: "POST",   path: "/api/v1/auth/token",      desc: "Login — get JWT token" },
  { method: "POST",   path: "/api/v1/chat",            desc: "Chat completion" },
  { method: "POST",   path: "/api/v1/chat/stream",     desc: "SSE streaming chat" },
  { method: "DELETE", path: "/api/v1/chat/{id}",       desc: "Clear conversation" },
  { method: "POST",   path: "/api/v1/rag/ingest",      desc: "Ingest raw text" },
  { method: "POST",   path: "/api/v1/rag/ingest/file", desc: "Upload document" },
  { method: "POST",   path: "/api/v1/rag/search",      desc: "Semantic search" },
  { method: "GET",    path: "/api/v1/rag/stats",       desc: "Vector store stats" },
  { method: "POST",   path: "/api/v1/finetune",        desc: "Submit fine-tune job" },
  { method: "GET",    path: "/api/v1/finetune",        desc: "List jobs" },
  { method: "GET",    path: "/api/v1/finetune/{id}",   desc: "Get job status" },
  { method: "GET",    path: "/health",                 desc: "Health check" },
  { method: "GET",    path: "/metrics",                desc: "Prometheus metrics" },
];
