"""
app/services/finetune_service.py
──────────────────────────────────
LoRA / QLoRA fine-tuning using HuggingFace PEFT + TRL.

Flow:
  1. Load base model in 4-bit (QLoRA) or full precision
  2. Apply LoRA adapter via PEFT
  3. Load dataset from HuggingFace Hub or local JSONL
  4. Run SFTTrainer (supervised fine-tuning)
  5. Save adapter weights

This runs in a background thread to avoid blocking the API server.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import FineTuneRequest

logger = get_logger(__name__)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class FineTuneJob:
    def __init__(self, job_id: str, request: FineTuneRequest) -> None:
        self.job_id = job_id
        self.request = request
        self.status = JobStatus.pending
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.metrics: dict[str, Any] = {}


class FineTuneService:
    def __init__(self) -> None:
        self._jobs: dict[str, FineTuneJob] = {}

    def submit(self, request: FineTuneRequest) -> FineTuneJob:
        job_id = str(uuid.uuid4())
        job = FineTuneJob(job_id=job_id, request=request)
        self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._run_job, args=(job,), daemon=True, name=f"finetune-{job_id[:8]}"
        )
        thread.start()
        logger.info("finetune_job_submitted", job_id=job_id)
        return job

    def get_job(self, job_id: str) -> FineTuneJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[FineTuneJob]:
        return list(self._jobs.values())

    # ── Core training logic (runs in background thread) ──────────────────────

    def _run_job(self, job: FineTuneJob) -> None:
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        logger.info("finetune_job_started", job_id=job.job_id)

        try:
            self._train(job)
            job.status = JobStatus.completed
            logger.info("finetune_job_completed", job_id=job.job_id, metrics=job.metrics)
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            logger.error("finetune_job_failed", job_id=job.job_id, error=str(exc))
        finally:
            job.finished_at = datetime.utcnow()

    def _train(self, job: FineTuneJob) -> None:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer

        req = job.request
        base_model = req.base_model or settings.finetune_base_model
        output_dir = str(settings.finetune_output_dir / job.job_id)

        # ── Quantisation config (QLoRA: 4-bit) ──────────────────────────────
        use_4bit = torch.cuda.is_available()
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # ── Load base model ──────────────────────────────────────────────────
        logger.info("loading_base_model", model=base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, token=settings.hf_token or None
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            token=settings.hf_token or None,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # ── LoRA config ──────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=req.lora_r,
            lora_alpha=req.lora_alpha,
            lora_dropout=req.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=req.target_modules,
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "lora_applied",
            trainable_params=trainable,
            total_params=total,
            pct=round(100 * trainable / total, 2),
        )

        # ── Dataset ─────────────────────────────────────────────────────────
        if req.dataset_name.startswith("local:"):
            from datasets import Dataset
            import json
            path = req.dataset_name[len("local:"):]
            with open(path) as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list(data)
        else:
            dataset = load_dataset(req.dataset_name, split="train")

        # ── Training args ────────────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=req.epochs,
            per_device_train_batch_size=req.batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
            learning_rate=req.learning_rate,
            weight_decay=0.001,
            fp16=torch.cuda.is_available() and not use_4bit,
            bf16=use_4bit and torch.cuda.is_bf16_supported(),
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=25,
            save_strategy="epoch",
            report_to="none",
        )

        # ── SFT Trainer ──────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=settings.finetune_max_seq_len,
        )
        train_result = trainer.train()
        trainer.save_model(output_dir)

        job.metrics = {
            "train_loss": round(train_result.training_loss, 4),
            "train_runtime_s": round(train_result.metrics.get("train_runtime", 0)),
            "adapter_path": output_dir,
        }


finetune_service = FineTuneService()
