import json
from datetime import datetime
from typing import Any

from celery import shared_task

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

@shared_task(bind=True, name="finetune.train")
def run_finetune_task(self, req_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Celery task that performs LoRA fine-tuning.
    We pass req_dict instead of FineTuneRequest because Pydantic models aren't JSON serialisable out of the box for Celery.
    """
    job_id = self.request.id
    logger.info("finetune_job_started", job_id=job_id)

    try:
        metrics = _train(job_id, req_dict)
        logger.info("finetune_job_completed", job_id=job_id, metrics=metrics)
        return {"status": "completed", "metrics": metrics, "error": None}
    except Exception as exc:
        logger.error("finetune_job_failed", job_id=job_id, error=str(exc))
        return {"status": "failed", "metrics": {}, "error": str(exc)}

def _train(job_id: str, req: dict[str, Any]) -> dict[str, Any]:
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

    base_model = req.get("base_model") or settings.finetune_base_model
    output_dir = str(settings.finetune_output_dir / job_id)

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
        r=req.get("lora_r", 16),
        lora_alpha=req.get("lora_alpha", 32),
        lora_dropout=req.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=req.get("target_modules", ["q_proj", "v_proj"]),
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
    dataset_name = req.get("dataset_name", "")
    if dataset_name.startswith("local:"):
        from datasets import Dataset
        path = dataset_name[len("local:"):]
        with open(path) as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(dataset_name, split="train")

    # ── Training args ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=req.get("epochs", 3),
        per_device_train_batch_size=req.get("batch_size", 4),
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
        learning_rate=req.get("learning_rate", 2e-4),
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

    return {
        "train_loss": round(train_result.training_loss, 4),
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0)),
        "adapter_path": output_dir,
    }
