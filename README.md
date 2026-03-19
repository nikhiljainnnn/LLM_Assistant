# 🤖 LLM-Powered AI Assistant

A production-ready, end-to-end conversational AI assistant with RAG, LoRA fine-tuning support, and a modern frontend.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│           Chat UI │ Document Upload │ Settings              │
└─────────────────────────┬───────────────────────────────────┘
                           │ REST / SSE (streaming)
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Chat API  │  │  RAG API    │  │  Fine-tune API      │  │
│  └─────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│        │                │                      │             │
│  ┌─────▼──────────────────▼──────────────────▼──────────┐  │
│  │              Service Layer                             │  │
│  │  LLM Service │ RAG Service │ Embedding │ Memory       │  │
│  └─────┬──────────────────┬──────────────────────────────┘  │
│        │                  │                                   │
│  ┌─────▼──────┐  ┌────────▼──────┐  ┌────────────────────┐ │
│  │OpenAI/HF   │  │ FAISS Vector  │  │  LoRA Fine-tuner   │ │
│  │  Models    │  │     Store     │  │  (Transformers)    │ │
│  └────────────┘  └───────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key (optional: HuggingFace token for private models)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env   # Fill in your API keys
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

### Docker (Full Stack)
```bash
docker-compose up --build
```

## 📁 Project Structure

```
llm-assistant/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routers
│   │   ├── core/          # Config, security, middleware
│   │   ├── models/        # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── utils/         # Helpers
│   ├── tests/             # Pytest test suite
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/    # React components
│       ├── hooks/         # Custom hooks
│       └── lib/           # API client, utils
├── scripts/               # Fine-tuning & ingestion scripts
├── configs/               # Model & RAG configs
├── docker-compose.yml
└── .env.example
```

## 🔧 Features

- **Multi-model support**: OpenAI GPT-4/3.5 + HuggingFace Transformers
- **RAG Pipeline**: FAISS vector store with semantic chunking
- **Streaming**: Server-Sent Events for real-time token streaming
- **Memory**: Conversation history with sliding window
- **Fine-tuning**: LoRA/QLoRA with PEFT for domain adaptation
- **Document Ingestion**: PDF, DOCX, TXT, Markdown support
- **Observability**: Structured logging, token tracking, latency metrics
- **Auth**: API key-based authentication
- **Rate Limiting**: Per-user request throttling
