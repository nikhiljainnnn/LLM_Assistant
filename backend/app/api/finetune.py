"""
app/api/finetune.py
────────────────────
POST /api/v1/finetune        — submit a LoRA fine-tuning job
GET  /api/v1/finetune        — list all jobs
GET  /api/v1/finetune/{id}   — get job status + metrics
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_current_user
from app.models.domain import User
from app.models.schemas import FineTuneRequest, FineTuneResponse
from app.services.finetune_service import JobStatus, finetune_service

router = APIRouter(prefix="/api/v1/finetune", tags=["finetune"])


@router.post("", response_model=FineTuneResponse)
async def submit_finetune(
    req: FineTuneRequest,
    current_user: User = Depends(get_current_user),
):
    job = finetune_service.submit(req)
    return FineTuneResponse(
        job_id=job["job_id"],
        status=job["status"],
        message="Fine-tuning job submitted. Poll /api/v1/finetune/{job_id} for status.",
    )


@router.get("")
async def list_jobs(current_user: User = Depends(get_current_user)):
    jobs = finetune_service.list_jobs()
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "started_at": j.get("started_at"),
            "finished_at": j.get("finished_at"),
            "metrics": j.get("metrics", {}),
            "error": j.get("error"),
        }
        for j in jobs
    ]


@router.get("/{job_id}")
async def get_job(job_id: str, current_user: User = Depends(get_current_user)):
    job = finetune_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "metrics": job.get("metrics", {}),
        "error": job.get("error"),
    }
