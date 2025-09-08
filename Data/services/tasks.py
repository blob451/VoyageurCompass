"""
Celery tasks for the Data app.
Handles asynchronous processing and scheduled jobs.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.utils import timezone
from datetime import datetime, timedelta
import json
import time

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3)
def sync_market_data(self):
    """
    Synchronize market data from external sources.
    This task refactors the logic from synchronizer.py.
    """
    try:
        logger.info("Starting market data synchronization...")
        
        # Import here to avoid circular imports
        from Data.services.synchronizer import DataSynchronizer
        
        synchronizer = DataSynchronizer()
        
        # Cache key for tracking sync status
        cache_key = 'market_data_sync_status'
        
        # Set sync status in cache
        cache.set(cache_key, {
            'status': 'running',
            'started_at': timezone.now().isoformat(),
            'task_id': self.request.id
        }, timeout=3600)
        
        # Perform the actual synchronization
        result = synchronizer.sync_all_data()
        
        # Update cache with completion status
        cache.set(cache_key, {
            'status': 'completed',
            'completed_at': timezone.now().isoformat(),
            'result': result,
            'task_id': self.request.id
        }, timeout=86400)  # Keep for 24 hours
        
        # Invalidate relevant caches after sync
        cache.delete_pattern('voyageur:market:*')
        
        logger.info(f"Market data synchronization completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Market data sync failed: {exc}")
        
        # Update cache with error status
        cache.set(cache_key, {
            'status': 'failed',
            'failed_at': timezone.now().isoformat(),
            'error': str(exc),
            'task_id': self.request.id
        }, timeout=3600)
        
        # Retry the task with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@shared_task
def cleanup_old_cache():
    """
    Clean up old cache entries and expired data.
    Runs weekly to maintain cache hygiene.
    """
    try:
        logger.info("Starting cache cleanup...")
        
        # Define patterns for different cache categories
        cleanup_patterns = [
            ('voyageur:temp:*', 0),  # Immediate cleanup for temp data
            ('voyageur:session:*', 86400),  # 24 hours for session data
            ('voyageur:analytics:*', 604800),  # 7 days for analytics
        ]
        
        cleanup_count = 0
        
        for pattern, max_age in cleanup_patterns:
            # Note: This requires django-redis backend
            keys = cache.keys(pattern)
            
            for key in keys:
                if max_age == 0:
                    cache.delete(key)
                    cleanup_count += 1
                else:
                    # Check age of cache entry
                    ttl = cache.ttl(key)
                    if ttl is not None and ttl < max_age:
                        cache.delete(key)
                        cleanup_count += 1
        
        logger.info(f"Cache cleanup completed. Removed {cleanup_count} entries.")
        
        # Store cleanup stats
        cache.set('voyageur:maintenance:last_cleanup', {
            'timestamp': timezone.now().isoformat(),
            'entries_removed': cleanup_count
        }, timeout=604800)  # Keep for 7 days
        
        return {'entries_removed': cleanup_count}
        
    except Exception as exc:
        logger.error(f"Cache cleanup failed: {exc}")
        raise


@shared_task
def generate_analytics_report():
    """
    Generate daily analytics report.
    Aggregates data and prepares reports for dashboard viewing.
    """
    try:
        logger.info("Generating analytics report...")
        
        # Import models and services
        from django.db.models import Count, Avg, Sum, Q
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        
        # Get date range for report
        end_date = timezone.now()
        start_date = end_date - timedelta(days=1)
        
        # Aggregate user metrics
        user_metrics = {
            'total_users': User.objects.count(),
            'active_users_24h': User.objects.filter(
                last_login__gte=start_date
            ).count(),
            'new_users_24h': User.objects.filter(
                date_joined__gte=start_date
            ).count(),
        }
        
        # Aggregate market data metrics (example)
        market_metrics = {
            'data_points_collected': 0,  # Placeholder
            'avg_processing_time': 0,  # Placeholder
        }
        
        # Try to import and use MarketData if it exists
        try:
            from Data.models import MarketData
            market_metrics['data_points_collected'] = MarketData.objects.filter(
                created_at__gte=start_date
            ).count()
        except ImportError:
            logger.info("MarketData model not available")
        
        # System performance metrics
        system_metrics = {
            'cache_hit_rate': calculate_cache_hit_rate(),
            'task_success_rate': calculate_task_success_rate(),
            'api_response_time': calculate_avg_response_time(),
        }
        
        # Compile full report
        report = {
            'report_date': end_date.isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            },
            'user_metrics': user_metrics,
            'market_metrics': market_metrics,
            'system_metrics': system_metrics,
            'generated_at': timezone.now().isoformat(),
        }
        
        # Cache the report for quick access - fix cache key to avoid "/"
        date_str = end_date.date().isoformat().replace('/', '-')
        cache_key = f'voyageur:analytics:daily_report:{date_str}'
        cache.set(cache_key, report, timeout=604800)  # Keep for 7 days
        
        # Also store in a list of recent reports
        recent_reports_key = 'voyageur:analytics:recent_reports'
        recent_reports = cache.get(recent_reports_key, [])
        recent_reports.insert(0, {
            'date': date_str,
            'cache_key': cache_key
        })
        # Keep only last 30 reports
        recent_reports = recent_reports[:30]
        cache.set(recent_reports_key, recent_reports, timeout=2592000)  # 30 days
        
        logger.info(f"Analytics report generated: {report}")
        return report
        
    except Exception as exc:
        logger.error(f"Analytics report generation failed: {exc}")
        raise


@shared_task
def process_data_upload(file_path, user_id):
    """
    Process uploaded data files asynchronously.
    Used for handling large file uploads without blocking the request.
    """
    try:
        logger.info(f"Processing uploaded file: {file_path} for user {user_id}")
        
        # Simulate file processing
        # In real implementation, this would parse and store data
        time.sleep(2)  # Simulate processing time
        
        # Update processing status in cache
        status_key = f'voyageur:upload:status:{user_id}:{file_path}'
        cache.set(status_key, {
            'status': 'completed',
            'processed_at': timezone.now().isoformat(),
            'file': file_path
        }, timeout=3600)
        
        logger.info(f"File processing completed: {file_path}")
        return {'status': 'success', 'file': file_path}
        
    except Exception as exc:
        logger.error(f"File processing failed: {exc}")
        
        # Update status with error
        status_key = f'voyageur:upload:status:{user_id}:{file_path}'
        cache.set(status_key, {
            'status': 'failed',
            'error': str(exc),
            'failed_at': timezone.now().isoformat()
        }, timeout=3600)
        
        raise


# Helper functions for analytics
def calculate_cache_hit_rate():
    """Calculate cache hit rate from Redis stats."""
    try:
        # This is a simplified example
        # In production, you'd query Redis INFO stats
        return 0.85  # 85% hit rate placeholder
    except:
        return 0


def calculate_task_success_rate():
    """Calculate Celery task success rate."""
    try:
        # Query Celery result backend for task stats
        # This is a placeholder implementation
        return 0.95  # 95% success rate placeholder
    except:
        return 0


def calculate_avg_response_time():
    """Calculate average API response time."""
    try:
        # Would query your monitoring/logging system
        return 0.150  # 150ms placeholder
    except:
        return 0
