"""
app/services/finetune_service.py
──────────────────────────────────
Service layer to interface with Celery tasks for fine-tuning.
"""

from __future__ import annotations

from typing import Any

from celery.result import AsyncResult

from app.core.logging import get_logger
from app.models.schemas import FineTuneRequest
from app.worker import celery_app

logger = get_logger(__name__)


class JobStatus:
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class FineTuneService:
    def submit(self, request: FineTuneRequest) -> dict[str, Any]:
        req_dict = request.model_dump()
        task = celery_app.send_task("finetune.train", args=[req_dict])
        logger.info("finetune_job_submitted", job_id=task.id)
        return {
            "job_id": task.id,
            "status": JobStatus.pending,
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        result = AsyncResult(job_id, app=celery_app)
        
        # Mapping celery states to our app states
        status_map = {
            "PENDING": JobStatus.pending,
            "STARTED": JobStatus.running,
            "SUCCESS": JobStatus.completed,
            "FAILURE": JobStatus.failed,
            "REVOKED": JobStatus.failed,
        }
        
        state = status_map.get(result.state, JobStatus.pending)
        
        metrics = {}
        error = None
        
        if state == JobStatus.completed:
            res_dict = result.result
            if isinstance(res_dict, dict):
                # We returned a dict from run_finetune_task
                if res_dict.get("status") == "failed":
                    state = JobStatus.failed
                    error = res_dict.get("error")
                metrics = res_dict.get("metrics", {})
        elif state == JobStatus.failed:
            error = str(result.result)

        return {
            "job_id": job_id,
            "status": state,
            "metrics": metrics,
            "error": error
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        # Without a database of job IDs, listing all jobs is non-trivial in pure Redis-backed Celery.
        # In a real app we'd track job IDs in Postgres.
        # We will return an empty list or mock it for now since we removed the in-memory dictionary.
        # A full production system should insert a row into `finetune_jobs` in PostgreSQL on submit.
        return []

finetune_service = FineTuneService()
