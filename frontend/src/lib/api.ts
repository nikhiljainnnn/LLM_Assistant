/**
 * src/lib/api.ts — Typed API client for the LLM Assistant backend.
 */

// Force relative paths so it goes through the NGINX or Vite proxy
const BASE_URL = "";
// @ts-ignore
const API_KEY = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_KEY) ?? "dev-api-key";

const defaultHeaders = { "Content-Type": "application/json", "X-API-Key": API_KEY };

export interface Message { role: "user" | "assistant" | "system"; content: string; }
export interface SourceChunk { text: string; source: string; score: number; chunk_index: number; }
export interface TokenUsage { prompt_tokens: number; completion_tokens: number; total_tokens: number; }
export interface ChatResponse { conversation_id: string; message: Message; sources: SourceChunk[]; usage: TokenUsage | null; latency_ms: number; }
export interface ChatRequest { message: string; conversation_id: string; use_rag?: boolean; provider?: "openai" | "huggingface"; model?: string; system_prompt?: string; temperature?: number; max_tokens?: number; }
export interface IngestResponse { source_name: string; chunks_added: number; total_vectors: number; }
export interface SearchResult { query: string; results: SourceChunk[]; }
export interface RAGStats { total_vectors: number; embedding_dim: number; store_path: string; }
export interface HealthResponse { status: string; version: string; providers: Record<string, boolean>; vector_store_size: number; }
export interface FineTuneJob { job_id: string; status: string; started_at: string | null; finished_at: string | null; metrics: Record<string, unknown>; error: string | null; }

export interface TokenResponse { access_token: string; token_type: string; }

export class APIError extends Error {
  constructor(public status: number, public detail: string) { super(`API Error ${status}: ${detail}`); }
}

const getHeaders = (): Record<string, string> => {
  const token = localStorage.getItem("access_token");
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    headers["X-API-Key"] = API_KEY;
  }
  return headers;
};

const getFormHeaders = (): Record<string, string> => {
  const token = localStorage.getItem("access_token");
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    headers["X-API-Key"] = API_KEY;
  }
  return headers;
};

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try { const b = await res.json(); detail = b.detail ?? detail; } catch {}
    throw new APIError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function login(username: string, password: string):Promise<TokenResponse> {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);
  const res = await fetch(`${BASE_URL}/api/v1/auth/token`, { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form.toString() });
  return handleResponse<TokenResponse>(res);
}

export async function register(username: string, email: string, password: string):Promise<any> {
  const res = await fetch(`${BASE_URL}/api/v1/auth/register`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, email, password, full_name: username }) });
  return handleResponse<any>(res);
}

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/chat`, { method: "POST", headers: getHeaders(), body: JSON.stringify(req) });
  return handleResponse<ChatResponse>(res);
}

export async function streamChat(req: ChatRequest, onToken: (t: string) => void, onDone: () => void, onError: (e: Error) => void): Promise<void> {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/chat/stream`, { method: "POST", headers: getHeaders(), body: JSON.stringify({ ...req, stream: true }) });
    if (!res.ok) throw new APIError(res.status, res.statusText);
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value, { stream: true }).split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") { onDone(); return; }
          onToken(payload);
        }
      }
    }
    onDone();
  } catch (err) { onError(err instanceof Error ? err : new Error(String(err))); }
}

export async function clearConversation(id: string): Promise<void> {
  await fetch(`${BASE_URL}/api/v1/chat/${id}`, { method: "DELETE", headers: getHeaders() });
}

export async function ingestText(text: string, sourceName: string): Promise<IngestResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/rag/ingest`, { method: "POST", headers: getHeaders(), body: JSON.stringify({ text, source_name: sourceName }) });
  return handleResponse<IngestResponse>(res);
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("source_name", file.name);
  const res = await fetch(`${BASE_URL}/api/v1/rag/ingest/file`, { method: "POST", headers: getFormHeaders(), body: form });
  return handleResponse<IngestResponse>(res);
}

export async function semanticSearch(query: string, topK = 5): Promise<SearchResult> {
  const res = await fetch(`${BASE_URL}/api/v1/rag/search`, { method: "POST", headers: getHeaders(), body: JSON.stringify({ query, top_k: topK }) });
  return handleResponse<SearchResult>(res);
}

export async function getRAGStats(): Promise<RAGStats> {
  const res = await fetch(`${BASE_URL}/api/v1/rag/stats`, { headers: getHeaders() });
  return handleResponse<RAGStats>(res);
}

export async function listFineTuneJobs(): Promise<FineTuneJob[]> {
  const res = await fetch(`${BASE_URL}/api/v1/finetune`, { headers: getHeaders() });
  return handleResponse<FineTuneJob[]>(res);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`, { headers: getHeaders() });
  return handleResponse<HealthResponse>(res);
}
