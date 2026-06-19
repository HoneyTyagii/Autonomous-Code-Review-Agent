"""Celery application configuration.

Sets up the Celery app with Redis as broker and result backend,
configures serialization, task routes, and retry policies.
"""

from celery import Celery

from code_review_agent.config import get_settings

settings = get_settings()

celery_app = Celery(
    "code_review_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result expiration (24 hours)
    result_expires=86400,

    # Task time limits
    task_soft_time_limit=600,  # 10 minutes soft limit
    task_time_limit=900,  # 15 minutes hard limit

    # Retry policy
    task_default_retry_delay=60,
    task_max_retries=3,

    # Task routing
    task_routes={
        "code_review_agent.tasks.review_tasks.review_pull_request": {
            "queue": "reviews",
        },
        "code_review_agent.tasks.review_tasks.index_repository": {
            "queue": "indexing",
        },
    },

    # Worker concurrency (per worker instance)
    worker_concurrency=4,

    # Task discovery
    imports=["code_review_agent.tasks.review_tasks"],
)
