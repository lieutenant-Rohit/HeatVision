"""
Celery application configuration.

Connects to Redis as both the message broker and result backend.

Start the Celery worker (from project root):
    celery -A app.celery_app worker --loglevel=info

Start the Celery beat scheduler (for periodic nightly refresh):
    celery -A app.celery_app beat --loglevel=info

Start both in one process (dev convenience):
    celery -A app.celery_app worker --beat --loglevel=info
"""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "heatmapper",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# ── Default configuration ──
celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    result_expires=3600 * 24 * 7,       # keep task results for 7 days
    task_track_started=True,
    task_acks_late=True,                 # re-deliver if worker crashes
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=1,     # fail fast if Redis is down
    broker_connection_timeout=3,         # seconds
    redis_socket_connect_timeout=3,
    redis_socket_timeout=5,
)

# ── Periodic schedule (nightly refresh at 2 AM) ──
celery_app.conf.beat_schedule = {
    "nightly-data-refresh": {
        "task": "app.tasks.refresh.refresh_pipeline",
        "schedule": crontab(
            hour=settings.REFRESH_SCHEDULE_HOUR,
            minute=settings.REFRESH_SCHEDULE_MINUTE,
        ),
        "kwargs": {"auto": True},
    },
}

# ── Auto-discover tasks ──
celery_app.autodiscover_tasks(["app.tasks"])
