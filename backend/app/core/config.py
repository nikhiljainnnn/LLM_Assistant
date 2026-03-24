"""
app/core/config.py
──────────────────
Centralised settings loaded from environment / .env file.
All downstream modules import `settings` from here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "insecure-dev-secret"
    api_key: str = "dev-api-key"
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── OpenAI ─────────────────────────────────────────────
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── HuggingFace ────────────────────────────────────────
    hf_token: str = ""
    hf_default_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # ── RAG ────────────────────────────────────────────────
    vector_store_path: Path = Path("./data/faiss_index")
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    embedding_device: str = "cpu"

    # ── Fine-tuning ────────────────────────────────────────
    finetune_base_model: str = "meta-llama/Llama-2-7b-hf"
    finetune_output_dir: Path = Path("./models/lora_adapter")
    finetune_epochs: int = 3
    finetune_batch_size: int = 4
    finetune_lr: float = 2e-4
    finetune_max_seq_len: int = 512

    # ── Memory ─────────────────────────────────────────────
    max_history_turns: int = 10

    # ── Rate Limiting ──────────────────────────────────────
    rate_limit_requests: int = 60
    rate_limit_tokens: int = 50_000

    # ── Logging ────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
