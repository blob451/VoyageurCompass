"""
Celery configuration for VoyageurCompass project.
"""
import os
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')

# Create the Celery app instance
app = Celery('VoyageurCompass')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Configure periodic tasks
app.conf.beat_schedule = {
    'sync-market-data-daily': {
        'task': 'Data.services.tasks.sync_market_data',
        'schedule': crontab(hour=0, minute=0),  # Run daily at midnight
        'options': {
            'expires': 3600,  # Task expires after 1 hour if not executed
        }
    },
    'cleanup-old-cache-weekly': {
        'task': 'Data.services.tasks.cleanup_old_cache',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Run weekly on Sunday at 3 AM
    },
    'generate-analytics-report': {
        'task': 'Data.services.tasks.generate_analytics_report',
        'schedule': crontab(hour=6, minute=0),  # Run daily at 6 AM
    },
}


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """A simple debug task to test Celery is working."""
    logger.info(f'Request: {self.request!r}')
