# 🤖 NEXUS — LLM-Powered AI Assistant

A **production-ready**, end-to-end conversational AI assistant with RAG, LoRA fine-tuning, JWT authentication, and a distributed backend.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     React Frontend (NEXUS)                        │
│         Auth UI │ Chat │ Knowledge Base (RAG) │ Settings          │
│                      Nginx (Reverse Proxy)                        │
└─────────────────────────┬────────────────────────────────────────┘
                           │ /api/* → backend:8000   (no CORS)
┌─────────────────────────▼────────────────────────────────────────┐
│                      FastAPI Backend                              │
│  ┌───────────┐  ┌─────────────┐  ┌────────────┐  ┌──────────┐  │
│  │  /auth    │  │  /api/chat  │  │  /api/rag  │  │/finetune │  │
│  └─────┬─────┘  └──────┬──────┘  └─────┬──────┘  └────┬─────┘  │
│        │               │                │               │         │
│  ┌─────▼───────────────▼────────────────▼───────────────▼──────┐ │
│  │                    Service Layer                              │ │
│  │  LLM Service │ RAG Service │ Embedding │ Memory │ Fine-tune  │ │
│  └─────┬────────────────┬─────────────────────────────┬─────────┘ │
│        │                │                             │            │
│  ┌─────▼──────┐  ┌──────▼────────────┐  ┌───────────▼──────────┐ │
│  │ vLLM /     │  │ PostgreSQL        │  │  Celery Worker        │ │
│  │ OpenAI API │  │ + pgvector        │  │  (LoRA Fine-tuning)   │ │
│  └────────────┘  └───────────────────┘  └──────────────────────┘ │
│                           │                       │               │
│                    ┌──────▼───────────────────────▼──────┐       │
│                    │            Redis                      │       │
│                    │  (Celery broker · Chat history)       │       │
│                    └──────────────────────────────────────┘       │
│                                                                    │
│  Prometheus (/metrics) ◄─── prometheus-fastapi-instrumentator     │
└────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- (Optional) An OpenAI API key **or** a vLLM server with a HuggingFace model

### 1. Configure Environment
```bash
cp .env.example .env
# Fill in APP_SECRET_KEY, OPENAI_API_KEY, HF_TOKEN, etc.
```

### 2. Start the Full Stack
```bash
docker-compose up --build -d
```

Services started:
| Service | URL |
|---|---|
| Frontend (NEXUS) | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### 3. Run Database Migrations
```bash
docker exec -it llm-assistant-backend alembic upgrade head
```
This creates the `users` and `document_chunks` (pgvector) tables.

### 4. Register & Login
Navigate to http://localhost:3000. You will be greeted by the authentication screen — register a new account and log in. The JWT token is stored in `localStorage` and sent automatically with every request.

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `APP_SECRET_KEY` | *(insecure)* | Secret key for JWT signing — **change in production** |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL connection string (pgvector image required) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL used by Celery & memory service |
| `OPENAI_API_KEY` | *(empty)* | Set to use GPT-4o / text-embedding-3-small |
| `OPENAI_DEFAULT_MODEL` | `gpt-4o-mini` | Default OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `HF_TOKEN` | *(empty)* | HuggingFace token for private/gated models |
| `HF_DEFAULT_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Default HF model (used without vLLM) |
| `VLLM_BASE_URL` | *(empty)* | Point to a running vLLM server (OpenAI-compatible API) |
| `CHUNK_SIZE` | `512` | RAG document chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of chunks retrieved per query |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` for local embeddings |
| `FINETUNE_BASE_MODEL` | `meta-llama/Llama-2-7b-hf` | Base model for LoRA fine-tuning |
| `MAX_HISTORY_TURNS` | `10` | Sliding window for Redis-backed chat memory |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per minute per IP |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_FORMAT` | `json` | `json` (production) \| `text` (development) |

---

## 📁 Project Structure

```
llm-assistant/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routers (auth, chat, rag, finetune)
│   │   ├── core/          # Config, DB engine, security (JWT), middleware, logging
│   │   ├── models/        # SQLAlchemy domain models (User, DocumentChunk) + Pydantic schemas
│   │   ├── services/      # Business logic (LLM, RAG, Embedding, Memory, Finetune)
│   │   ├── tasks/         # Celery tasks (finetune_tasks.py)
│   │   ├── utils/         # Helpers
│   │   └── worker.py      # Celery app factory
│   ├── migrations/        # Alembic async migrations
│   ├── tests/             # Pytest test suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/    # AuthScreen, ChatPanel, RAGPanel, SettingsPanel, StatusBar
│       ├── hooks/         # Custom React hooks
│       └── lib/           # api.ts (JWT-authenticated fetch client)
├── configs/               # Model & RAG config files
├── scripts/               # Fine-tuning & ingestion helper scripts
├── prometheus.yml         # Prometheus scrape config
├── docker-compose.yml
└── .env.example
```

---

## 🔧 Features

| Feature | Implementation |
|---|---|
| **Multi-model LLM** | OpenAI GPT-4o/3.5 · HuggingFace models via vLLM (async OpenAI-compatible client) |
| **RAG Pipeline** | PostgreSQL + **pgvector** with cosine similarity (`<=>`) — no FAISS |
| **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| **Persistent Memory** | Redis-backed conversation history with sliding window |
| **LoRA Fine-tuning** | Async Celery tasks (SFTTrainer + PEFT) — no HTTP timeouts |
| **Document Ingestion** | PDF, DOCX, TXT, Markdown — chunked and embedded into pgvector |
| **Authentication** | JWT (OAuth2 password flow) via `/auth/register` + `/auth/token` |
| **Rate Limiting** | Per-IP request throttling with SlowAPI |
| **Observability** | Prometheus metrics via `prometheus-fastapi-instrumentator` |
| **Structured Logging** | JSON/text structured logs with `structlog` |
| **Reverse Proxy** | Nginx proxies `/api/*` to the backend — eliminates CORS entirely |

---

## 🛠️ Local Development (without Docker)

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp ../.env.example .env         # configure DATABASE_URL, REDIS_URL, etc.
uvicorn app.main:app --reload --port 8000
```

> **Note:** A running PostgreSQL (with pgvector extension) and Redis instance are required.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
The dev server proxies `/api` to `http://localhost:8000` via Vite's proxy config.

### Celery Worker
```bash
cd backend
celery -A app.worker.celery_app worker --loglevel=info
```

---

## 🔌 Connecting a vLLM Server (GPU Inference)

Start a vLLM instance exposing the OpenAI-compatible API on your GPU server:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000
```

Then update `.env`:
```env
VLLM_BASE_URL=http://your-gpu-server:8000/v1
HF_DEFAULT_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

The backend's `llm_service.py` will automatically route HuggingFace model requests through the vLLM server using the async OpenAI client.

---

## 📊 Monitoring

Prometheus scrapes the backend's `/metrics` endpoint every 15 seconds (see [`prometheus.yml`](prometheus.yml)). Access the Prometheus UI at http://localhost:9090.

Key metrics exposed:
- `http_requests_total` — request count by route and status code
- `http_request_duration_seconds` — latency histograms
- `http_requests_in_progress` — in-flight requests

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```
