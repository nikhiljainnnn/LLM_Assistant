from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "llm_assistant_tasks",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Automatically discover tasks in these packages
    imports=["app.tasks.finetune_tasks"],
)
