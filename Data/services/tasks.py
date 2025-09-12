"""
Celery tasks implementing asynchronous processing capabilities for Data module.
Provides scheduled job execution and background processing functionality.
"""

import time
from datetime import timedelta

from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.utils import timezone

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3)
def sync_market_data(self):
    """
    Execute comprehensive market data synchronisation from external sources.
    Implements refactored synchroniser logic with enhanced error handling.
    """
    try:
        logger.info("Starting market data synchronization...")

        # Import here to avoid circular imports
        from Data.services.synchronizer import DataSynchronizer

        synchronizer = DataSynchronizer()

        # Cache key for tracking sync status
        cache_key = "market_data_sync_status"

        # Set sync status in cache
        cache.set(
            cache_key,
            {"status": "running", "started_at": timezone.now().isoformat(), "task_id": self.request.id},
            timeout=3600,
        )

        # Perform the actual synchronization
        result = synchronizer.sync_all_data()

        # Update cache with completion status
        cache.set(
            cache_key,
            {
                "status": "completed",
                "completed_at": timezone.now().isoformat(),
                "result": result,
                "task_id": self.request.id,
            },
            timeout=86400,
        )  # Keep for 24 hours

        # Invalidate relevant caches after sync
        cache.delete_pattern("voyageur:market:*")

        logger.info(f"Market data synchronization completed: {result}")
        return result

    except Exception as exc:
        logger.error(f"Market data sync failed: {exc}")

        # Update cache with error status
        cache.set(
            cache_key,
            {
                "status": "failed",
                "failed_at": timezone.now().isoformat(),
                "error": str(exc),
                "task_id": self.request.id,
            },
            timeout=3600,
        )

        # Retry the task with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))


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
            ("voyageur:temp:*", 0),  # Immediate cleanup for temp data
            ("voyageur:session:*", 86400),  # 24 hours for session data
            ("voyageur:analytics:*", 604800),  # 7 days for analytics
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
        cache.set(
            "voyageur:maintenance:last_cleanup",
            {"timestamp": timezone.now().isoformat(), "entries_removed": cleanup_count},
            timeout=604800,
        )  # Keep for 7 days

        return {"entries_removed": cleanup_count}

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
        from django.contrib.auth import get_user_model

        User = get_user_model()

        # Get date range for report
        end_date = timezone.now()
        start_date = end_date - timedelta(days=1)

        # Aggregate user metrics
        user_metrics = {
            "total_users": User.objects.count(),
            "active_users_24h": User.objects.filter(last_login__gte=start_date).count(),
            "new_users_24h": User.objects.filter(date_joined__gte=start_date).count(),
        }

        # Aggregate market data metrics (example)
        market_metrics = {
            "data_points_collected": 0,  # Placeholder
            "avg_processing_time": 0,  # Placeholder
        }

        # Try to import and use MarketData if it exists
        try:
            from Data.models import MarketData

            market_metrics["data_points_collected"] = MarketData.objects.filter(created_at__gte=start_date).count()
        except ImportError:
            logger.info("MarketData model not available")

        # System performance metrics
        system_metrics = {
            "cache_hit_rate": calculate_cache_hit_rate(),
            "task_success_rate": calculate_task_success_rate(),
            "api_response_time": calculate_avg_response_time(),
        }

        # Compile full report
        report = {
            "report_date": end_date.isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "user_metrics": user_metrics,
            "market_metrics": market_metrics,
            "system_metrics": system_metrics,
            "generated_at": timezone.now().isoformat(),
        }

        # Cache the report for quick access - fix cache key to avoid "/"
        date_str = end_date.date().isoformat().replace("/", "-")
        cache_key = f"voyageur:analytics:daily_report:{date_str}"
        cache.set(cache_key, report, timeout=604800)  # Keep for 7 days

        # Also store in a list of recent reports
        recent_reports_key = "voyageur:analytics:recent_reports"
        recent_reports = cache.get(recent_reports_key, [])
        recent_reports.insert(0, {"date": date_str, "cache_key": cache_key})
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
        status_key = f"voyageur:upload:status:{user_id}:{file_path}"
        cache.set(
            status_key,
            {"status": "completed", "processed_at": timezone.now().isoformat(), "file": file_path},
            timeout=3600,
        )

        logger.info(f"File processing completed: {file_path}")
        return {"status": "success", "file": file_path}

    except Exception as exc:
        logger.error(f"File processing failed: {exc}")

        # Update status with error
        status_key = f"voyageur:upload:status:{user_id}:{file_path}"
        cache.set(
            status_key, {"status": "failed", "error": str(exc), "failed_at": timezone.now().isoformat()}, timeout=3600
        )

        raise


# Helper functions for analytics
def calculate_cache_hit_rate():
    """Calculate cache hit rate from Redis stats."""
    try:
        # This is a simplified example
        # In production, you'd query Redis INFO stats
        return 0.85  # 85% hit rate placeholder
    except Exception:
        return 0


def calculate_task_success_rate():
    """Calculate Celery task success rate."""
    try:
        # Query Celery result backend for task stats
        # This is a placeholder implementation
        return 0.95  # 95% success rate placeholder
    except Exception:
        return 0


def calculate_avg_response_time():
    """Calculate average API response time."""
    try:
        # Would query your monitoring/logging system
        return 0.150  # 150ms placeholder
    except Exception:
        return 0


# Yahoo Finance Cache Warming Tasks
@shared_task(bind=True, max_retries=3)
def warm_yahoo_cache_single_stock(self, symbol: str):
    """
    Execute cache warming operations for individual stock symbol.
    
    Args:
        symbol: Target stock symbol for cache population
    """
    try:
        logger.info(f"Warming cache for stock: {symbol}")
        
        from Data.services.yahoo_cache import yahoo_cache
        
        # Warm cache with different data types and periods
        cache_operations = [
            ("info", ""),
            ("history", "1d"),
            ("history", "5d"),
            ("history", "1mo"),
            ("history", "3mo"),
            ("history", "1y"),
            ("financials", ""),
        ]
        
        successful_operations = 0
        failed_operations = []
        
        for data_type, period in cache_operations:
            try:
                if data_type == "info":
                    result = yahoo_cache.get_stock_info(symbol)
                elif data_type == "history":
                    result = yahoo_cache.get_stock_history(symbol, period)
                elif data_type == "financials":
                    result = yahoo_cache.get_stock_financials(symbol)
                
                if result:
                    successful_operations += 1
                    logger.debug(f"Successfully cached {symbol} {data_type} {period}")
                else:
                    failed_operations.append(f"{data_type}:{period}")
                    
            except Exception as cache_error:
                logger.warning(f"Failed to cache {symbol} {data_type} {period}: {cache_error}")
                failed_operations.append(f"{data_type}:{period}")
        
        result = {
            "symbol": symbol,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "total_operations": len(cache_operations)
        }
        
        logger.info(f"Cache warming completed for {symbol}: {successful_operations}/{len(cache_operations)} successful")
        return result
        
    except Exception as exc:
        logger.error(f"Cache warming failed for {symbol}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))


@shared_task(bind=True, max_retries=2)
def warm_yahoo_cache_batch(self, symbols: list, batch_size: int = 10):
    """
    Execute batch cache warming operations for multiple stock symbols.
    
    Args:
        symbols: Stock symbol collection for cache population
        batch_size: Parallel processing batch size configuration
    """
    try:
        logger.info(f"Starting cache warming for {len(symbols)} stocks in batches of {batch_size}")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from Data.services.yahoo_cache import yahoo_cache
        
        batch_results = []
        
        # Process stocks in batches to avoid overwhelming the Yahoo Finance API
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
                # Submit cache warming tasks for this batch
                future_to_symbol = {
                    executor.submit(warm_single_stock_sync, symbol): symbol 
                    for symbol in batch
                }
                
                batch_successful = 0
                batch_failed = []
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=30)
                        if result.get("successful_operations", 0) > 0:
                            batch_successful += 1
                        else:
                            batch_failed.append(symbol)
                    except Exception as exc:
                        logger.warning(f"Batch warming failed for {symbol}: {exc}")
                        batch_failed.append(symbol)
                
                batch_results.append({
                    "batch_number": i//batch_size + 1,
                    "batch_symbols": batch,
                    "successful": batch_successful,
                    "failed": batch_failed
                })
                
                # Brief pause between batches to be respectful to Yahoo Finance
                if i + batch_size < len(symbols):
                    time.sleep(5)
        
        # Aggregate results
        total_successful = sum(r["successful"] for r in batch_results)
        total_failed = sum(len(r["failed"]) for r in batch_results)
        
        final_result = {
            "total_symbols": len(symbols),
            "successful": total_successful,
            "failed": total_failed,
            "batch_results": batch_results,
            "completed_at": timezone.now().isoformat()
        }
        
        logger.info(f"Batch cache warming completed: {total_successful}/{len(symbols)} successful")
        return final_result
        
    except Exception as exc:
        logger.error(f"Batch cache warming failed: {exc}")
        raise self.retry(exc=exc, countdown=300)  # Retry after 5 minutes


def warm_single_stock_sync(symbol: str):
    """Synchronous version of single stock cache warming for use in batch processing."""
    from Data.services.yahoo_cache import yahoo_cache
    
    cache_operations = [
        ("info", ""),
        ("history", "1d"),
        ("history", "1mo"),
        ("history", "1y"),
    ]
    
    successful_operations = 0
    failed_operations = []
    
    for data_type, period in cache_operations:
        try:
            if data_type == "info":
                result = yahoo_cache.get_stock_info(symbol)
            elif data_type == "history":
                result = yahoo_cache.get_stock_history(symbol, period)
            
            if result:
                successful_operations += 1
            else:
                failed_operations.append(f"{data_type}:{period}")
                
        except Exception as cache_error:
            failed_operations.append(f"{data_type}:{period}")
    
    return {
        "symbol": symbol,
        "successful_operations": successful_operations,
        "failed_operations": failed_operations,
        "total_operations": len(cache_operations)
    }


@shared_task(bind=True)
def warm_sp500_cache(self):
    """
    Warm cache for S&P 500 stocks.
    Scheduled to run daily to ensure commonly requested stocks are cached.
    """
    try:
        logger.info("Starting S&P 500 cache warming")
        
        # S&P 500 symbols (subset for demonstration - in production would fetch dynamically)
        sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "UNH", "JNJ",
            "V", "PG", "JPM", "HD", "MA", "PFE", "CVX", "ABBV", "BAC", "KO",
            "AVGO", "PEP", "TMO", "COST", "WMT", "DIS", "ABT", "CRM", "ACN", "NFLX",
            "ADBE", "CSCO", "XOM", "VZ", "INTC", "CMCSA", "DHR", "QCOM", "TXN", "PM",
            "NEE", "RTX", "ORCL", "IBM", "AMD", "HON", "COP", "MO", "AMGN", "UNP"
        ]
        
        # Trigger batch cache warming
        result = warm_yahoo_cache_batch.delay(sp500_symbols, batch_size=10)
        
        logger.info(f"S&P 500 cache warming initiated for {len(sp500_symbols)} stocks")
        return {
            "status": "initiated",
            "symbols_count": len(sp500_symbols),
            "batch_task_id": result.id,
            "initiated_at": timezone.now().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"S&P 500 cache warming failed: {exc}")
        raise


@shared_task
def cleanup_expired_yahoo_cache():
    """
    Clean up expired Yahoo Finance cache entries from database.
    Scheduled to run daily to maintain database hygiene.
    """
    try:
        logger.info("Starting Yahoo Finance cache cleanup")
        
        from Data.models import YahooFinanceCache
        
        # Delete expired entries
        expired_count = YahooFinanceCache.objects.filter(
            expires_at__lt=timezone.now()
        ).delete()[0]
        
        # Delete failed entries older than 1 day
        failed_count = YahooFinanceCache.objects.filter(
            fetch_success=False,
            created_at__lt=timezone.now() - timedelta(days=1)
        ).delete()[0]
        
        result = {
            "expired_entries_deleted": expired_count,
            "failed_entries_deleted": failed_count,
            "total_deleted": expired_count + failed_count,
            "cleanup_at": timezone.now().isoformat()
        }
        
        logger.info(f"Yahoo Finance cache cleanup completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Yahoo Finance cache cleanup failed: {exc}")
        raise


@shared_task(bind=True, max_retries=3)
def async_stock_backfill(self, symbol: str, required_years: int = 2, user_id: int = None):
    """
    Perform async stock data backfill and trigger analysis when complete.
    
    Args:
        symbol: Stock ticker symbol to backfill
        required_years: Years of historical data required 
        user_id: User ID who requested the analysis
        
    Returns:
        Dict with backfill results and analysis trigger status
    """
    try:
        logger.info(f"Starting async backfill for {symbol} ({required_years} years)")
        
        # Set status in cache for UI tracking
        status_key = f"backfill_status_{symbol}_{user_id or 'system'}"
        cache.set(status_key, {
            "status": "running", 
            "symbol": symbol,
            "started_at": timezone.now().isoformat(),
            "task_id": self.request.id
        }, timeout=3600)
        
        from Data.services.yahoo_finance import yahoo_finance_service
        
        # Perform the backfill
        backfill_result = yahoo_finance_service.backfill_eod_gaps_concurrent(
            symbol=symbol,
            required_years=required_years,
            max_attempts=3
        )
        
        if backfill_result.get('success'):
            logger.info(f"Async backfill successful for {symbol}: {backfill_result['stock_backfilled']} records")
            
            # Update status - backfill complete
            cache.set(status_key, {
                "status": "backfill_complete",
                "symbol": symbol,
                "backfill_result": backfill_result,
                "completed_at": timezone.now().isoformat(),
                "task_id": self.request.id
            }, timeout=3600)
            
            # Trigger analysis if user requested it
            if user_id:
                try:
                    # Trigger async analysis
                    analysis_task = async_stock_analysis.delay(symbol, user_id)
                    logger.info(f"Triggered async analysis for {symbol}, task_id: {analysis_task.id}")
                    
                    # Update status - analysis triggered
                    cache.set(status_key, {
                        "status": "analysis_triggered",
                        "symbol": symbol,
                        "backfill_result": backfill_result,
                        "analysis_task_id": analysis_task.id,
                        "analysis_triggered_at": timezone.now().isoformat(),
                        "task_id": self.request.id
                    }, timeout=3600)
                    
                except Exception as analysis_error:
                    logger.error(f"Failed to trigger analysis for {symbol}: {analysis_error}")
                    # Still return success for backfill, but note analysis failure
                    cache.set(status_key, {
                        "status": "backfill_complete_analysis_failed",
                        "symbol": symbol,
                        "backfill_result": backfill_result,
                        "analysis_error": str(analysis_error),
                        "task_id": self.request.id
                    }, timeout=3600)
            
            return {
                "success": True,
                "symbol": symbol,
                "backfill_result": backfill_result,
                "analysis_triggered": bool(user_id)
            }
        else:
            # Backfill failed
            error_msg = f"Backfill failed for {symbol}: {backfill_result.get('errors', ['Unknown error'])}"
            logger.error(error_msg)
            
            cache.set(status_key, {
                "status": "failed",
                "symbol": symbol,
                "error": error_msg,
                "backfill_result": backfill_result,
                "failed_at": timezone.now().isoformat(),
                "task_id": self.request.id
            }, timeout=3600)
            
            # Retry with exponential backoff
            raise self.retry(exc=Exception(error_msg), countdown=60 * (2**self.request.retries))
            
    except Exception as exc:
        logger.error(f"Async backfill failed for {symbol}: {exc}")
        
        # Update cache with error status
        status_key = f"backfill_status_{symbol}_{user_id or 'system'}"
        cache.set(status_key, {
            "status": "failed",
            "symbol": symbol,
            "error": str(exc),
            "failed_at": timezone.now().isoformat(),
            "task_id": self.request.id
        }, timeout=3600)
        
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))


@shared_task(bind=True, max_retries=2)
def async_stock_analysis(self, symbol: str, user_id: int):
    """
    Perform async stock analysis after data is available.
    
    Args:
        symbol: Stock ticker symbol to analyze
        user_id: User ID who requested the analysis
        
    Returns:
        Dict with analysis results
    """
    try:
        logger.info(f"Starting async analysis for {symbol} (user: {user_id})")
        
        # Set status in cache for UI tracking  
        status_key = f"analysis_status_{symbol}_{user_id}"
        cache.set(status_key, {
            "status": "running",
            "symbol": symbol, 
            "user_id": user_id,
            "started_at": timezone.now().isoformat(),
            "task_id": self.request.id
        }, timeout=3600)
        
        from Analytics.engine.ta_engine import TechnicalAnalysisEngine
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        user = User.objects.get(id=user_id)
        
        # Run the analysis
        engine = TechnicalAnalysisEngine()
        analysis_result = engine.analyze_stock(symbol, user=user, fast_mode=False)
        
        logger.info(f"Async analysis complete for {symbol}: {analysis_result['score_0_10']}/10")
        
        # Update status with results
        cache.set(status_key, {
            "status": "completed",
            "symbol": symbol,
            "user_id": user_id, 
            "analysis_result": analysis_result,
            "completed_at": timezone.now().isoformat(),
            "task_id": self.request.id
        }, timeout=3600)
        
        return {
            "success": True,
            "symbol": symbol,
            "user_id": user_id,
            "analysis_result": analysis_result
        }
        
    except Exception as exc:
        logger.error(f"Async analysis failed for {symbol}: {exc}")
        
        # Update cache with error status
        status_key = f"analysis_status_{symbol}_{user_id}"
        cache.set(status_key, {
            "status": "failed",
            "symbol": symbol,
            "user_id": user_id,
            "error": str(exc),
            "failed_at": timezone.now().isoformat(),
            "task_id": self.request.id
        }, timeout=3600)
        
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries))


@shared_task(bind=True)
def monitor_data_quality_task(self):
    """
    Automated data quality monitoring task.
    
    Runs comprehensive data quality checks and stores results for dashboard.
    Scheduled to run daily to monitor system health.
    """
    try:
        logger.info("Starting automated data quality monitoring")
        
        from Data.services.data_quality_monitor import data_quality_monitor
        
        # Run comprehensive quality check
        result = data_quality_monitor.run_comprehensive_check()
        
        if 'error' not in result:
            overall_score = result.get('overall_quality_score', 0)
            issues_count = len(result.get('recommendations', []))
            
            # Log key metrics
            logger.info(f"Data quality check completed - Score: {overall_score:.1f}/10, Issues: {issues_count}")
            
            # Store historical metrics for trending
            cache.set(
                f"data_quality_history_{timezone.now().date().isoformat()}",
                {
                    'date': timezone.now().date().isoformat(),
                    'overall_score': overall_score,
                    'issues_count': issues_count,
                    'timestamp': timezone.now().isoformat()
                },
                timeout=604800  # Keep for 7 days
            )
            
            # Alert on critical issues
            if overall_score < 4.0:
                logger.error(f"CRITICAL: Data quality score is critically low: {overall_score:.1f}/10")
            elif overall_score < 6.0:
                logger.warning(f"WARNING: Data quality score is concerning: {overall_score:.1f}/10")
            
            return {
                'success': True,
                'overall_score': overall_score,
                'issues_count': issues_count,
                'timestamp': timezone.now().isoformat()
            }
        else:
            logger.error(f"Data quality monitoring failed: {result['error']}")
            return {
                'success': False,
                'error': result['error'],
                'timestamp': timezone.now().isoformat()
            }
            
    except Exception as exc:
        logger.error(f"Data quality monitoring task failed: {exc}")
        
        # Cache error status
        cache.set(
            "data_quality_monitor_error", 
            {
                'error': str(exc),
                'timestamp': timezone.now().isoformat(),
                'task_id': self.request.id
            },
            timeout=3600
        )
        
        raise
