"""
MultilingualOptimizer service for VoyageurCompass.

Handles parallel processing of multilingual explanation requests with intelligent
caching, resource management, and optimization strategies.
"""

import hashlib
import logging
import os
import psutil
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from django.conf import settings
from django.core.cache import cache

from Analytics.services.explanation_service import get_explanation_service
from Analytics.services.translation_service import get_translation_service
from Analytics.services.multilingual_exceptions import (
    MultilingualBaseException,
    LanguageNotSupportedException,
    ParallelProcessingException,
    ResourceExhaustedException,
    handle_multilingual_exception
)
from Analytics.services.unified_cache_manager import (
    get_cache_key_generator,
    generate_multilingual_cache_key
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRequest:
    """Request for multilingual optimization."""
    analysis_id: int
    symbol: str
    analysis_data: Dict[str, Any]
    target_languages: List[str]
    detail_level: str = "standard"
    user_id: Optional[int] = None
    force_regenerate: bool = False
    priority: str = "normal"


@dataclass
class OptimizationResult:
    """Result of multilingual optimization."""
    analysis_id: int
    symbol: str
    explanations: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    processing_time: float
    success: bool
    errors: List[str] = None


class MultilingualOptimizer:
    """Optimizer for parallel multilingual explanation generation."""

    def __init__(self):
        """Initialize the multilingual optimizer."""
        self.explanation_service = get_explanation_service()
        self.translation_service = get_translation_service()

        # Configuration
        self.max_workers = getattr(settings, 'MULTILINGUAL_MAX_WORKERS', 4)
        self.min_workers = getattr(settings, 'MULTILINGUAL_MIN_WORKERS', 1)
        self.cache_ttl = getattr(settings, 'MULTILINGUAL_CACHE_TTL', 3600)
        self.supported_languages = getattr(settings, 'MULTILINGUAL_SUPPORTED_LANGUAGES', ['en', 'fr', 'es'])
        self.memory_limit_mb = getattr(settings, 'MULTILINGUAL_MEMORY_LIMIT_MB', 512)

        # Dynamic thread pool sizing
        self.enable_dynamic_sizing = getattr(settings, 'MULTILINGUAL_DYNAMIC_POOL_SIZING', True)
        self.cpu_threshold_high = getattr(settings, 'MULTILINGUAL_CPU_THRESHOLD_HIGH', 80.0)
        self.cpu_threshold_low = getattr(settings, 'MULTILINGUAL_CPU_THRESHOLD_LOW', 50.0)
        self.memory_threshold_high = getattr(settings, 'MULTILINGUAL_MEMORY_THRESHOLD_HIGH', 80.0)
        self.pool_adjustment_interval = getattr(settings, 'MULTILINGUAL_POOL_ADJUSTMENT_INTERVAL', 30)  # seconds

        # Thread pool management
        self._current_pool_size = self.max_workers
        self._last_pool_adjustment = time.time()
        self._pool_performance_history = []

        # Performance tracking
        self._metrics_lock = threading.Lock()
        self._performance_metrics = {
            'requests_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_efficiency': [],
            'memory_usage_peak': 0.0,
            'errors': []
        }

        # Request deduplication
        self._active_requests: Set[str] = set()
        self._request_lock = threading.Lock()

        # Resource management
        self._resource_monitor = ResourceMonitor()

    def process_multilingual_request(
        self,
        request: OptimizationRequest
    ) -> OptimizationResult:
        """
        Process a multilingual explanation request with optimization.

        Args:
            request: Optimization request with target languages and settings

        Returns:
            OptimizationResult with explanations and performance metrics
        """
        start_time = time.time()

        try:
            # Validate request
            if not self._validate_request(request):
                return OptimizationResult(
                    analysis_id=request.analysis_id,
                    symbol=request.symbol,
                    explanations={},
                    performance_metrics={},
                    cache_stats={},
                    processing_time=0.0,
                    success=False,
                    errors=["Invalid request parameters"]
                )

            # Check for duplicate requests
            request_hash = self._generate_request_hash(request)
            if self._is_duplicate_request(request_hash):
                logger.info(f"Duplicate request detected for {request.symbol}")
                return self._wait_for_duplicate_completion(request_hash, request)

            try:
                self._mark_request_active(request_hash)

                # Generate explanations in parallel
                explanations, cache_stats = self._generate_parallel_explanations(request)

                # Calculate performance metrics
                processing_time = time.time() - start_time
                performance_metrics = self._calculate_performance_metrics(
                    request, explanations, processing_time
                )

                # Update global metrics
                self._update_global_metrics(processing_time, cache_stats, explanations)

                return OptimizationResult(
                    analysis_id=request.analysis_id,
                    symbol=request.symbol,
                    explanations=explanations,
                    performance_metrics=performance_metrics,
                    cache_stats=cache_stats,
                    processing_time=processing_time,
                    success=True
                )

            finally:
                self._mark_request_complete(request_hash)

        except MultilingualBaseException as e:
            # Handle known multilingual exceptions
            error_dict = handle_multilingual_exception(e, {
                "analysis_id": request.analysis_id,
                "symbol": request.symbol,
                "target_languages": request.target_languages
            }, logger)

            return OptimizationResult(
                analysis_id=request.analysis_id,
                symbol=request.symbol,
                explanations={},
                performance_metrics={},
                cache_stats={},
                processing_time=time.time() - start_time,
                success=False,
                errors=[error_dict]
            )

        except Exception as e:
            # Handle unexpected exceptions
            error_dict = handle_multilingual_exception(e, {
                "analysis_id": request.analysis_id,
                "symbol": request.symbol,
                "target_languages": request.target_languages,
                "error_type": "unexpected"
            }, logger)

            return OptimizationResult(
                analysis_id=request.analysis_id,
                symbol=request.symbol,
                explanations={},
                performance_metrics={},
                cache_stats={},
                processing_time=time.time() - start_time,
                success=False,
                errors=[error_dict]
            )

    def _generate_parallel_explanations(
        self,
        request: OptimizationRequest
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate explanations for multiple languages in parallel."""
        explanations = {}
        cache_stats = {
            'hits': 0,
            'misses': 0,
            'languages_cached': [],
            'languages_generated': []
        }

        # Check cache first for all languages
        cached_explanations = self._check_bulk_cache(request)
        explanations.update(cached_explanations)

        # Identify languages that need generation
        languages_to_generate = []
        for lang in request.target_languages:
            if lang not in explanations:
                languages_to_generate.append(lang)
                cache_stats['misses'] += 1
            else:
                cache_stats['hits'] += 1
                cache_stats['languages_cached'].append(lang)

        # Generate missing explanations in parallel
        if languages_to_generate:
            generated_explanations = self._parallel_generation(
                request, languages_to_generate
            )
            explanations.update(generated_explanations)
            cache_stats['languages_generated'] = list(generated_explanations.keys())

            # Cache the new explanations
            self._cache_bulk_explanations(request, generated_explanations)

        return explanations, cache_stats

    def _parallel_generation(
        self,
        request: OptimizationRequest,
        target_languages: List[str]
    ) -> Dict[str, Any]:
        """Generate explanations for multiple languages using parallel processing."""
        explanations = {}

        # Monitor memory usage
        memory_start = self._resource_monitor.get_memory_usage()

        # Get optimal pool size for this request
        optimal_pool_size = self._get_dynamic_pool_size(request)
        actual_pool_size = min(optimal_pool_size, len(target_languages))

        logger.debug(f"Using thread pool size: {actual_pool_size} for {len(target_languages)} languages")

        with ThreadPoolExecutor(max_workers=actual_pool_size) as executor:
            # Submit tasks for each language
            future_to_language = {}
            for language in target_languages:
                future = executor.submit(
                    self._generate_single_language,
                    request,
                    language
                )
                future_to_language[future] = language

            # Collect results
            for future in as_completed(future_to_language):
                language = future_to_language[future]
                try:
                    timeout = getattr(settings, 'MULTILINGUAL_TIMEOUT', 60)
                    explanation = future.result(timeout=timeout)
                    if explanation:
                        explanations[language] = explanation
                        logger.info(f"Generated explanation for {language}: {request.symbol}")
                    else:
                        logger.warning(f"Failed to generate explanation for {language}: {request.symbol}")
                except Exception as e:
                    logger.error(f"Error generating {language} explanation: {str(e)}")

        # Monitor memory after generation
        memory_end = self._resource_monitor.get_memory_usage()
        memory_used = memory_end - memory_start

        # Calculate success rate for pool performance tracking
        successful_generations = sum(1 for exp in explanations.values() if exp is not None)
        success_rate = successful_generations / len(target_languages) if target_languages else 0

        # Record pool performance for dynamic sizing optimization
        generation_time = time.time() - memory_start  # Approximate generation time
        self._record_pool_performance(request, generation_time, success_rate)

        with self._metrics_lock:
            self._performance_metrics['memory_usage_peak'] = max(
                self._performance_metrics['memory_usage_peak'],
                memory_used
            )

        return explanations

    def _generate_single_language(
        self,
        request: OptimizationRequest,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation for a single language."""
        try:
            # Use explanation service for generation
            explanation = self.explanation_service.explain_prediction_single(
                request.analysis_data,
                detail_level=request.detail_level,
                language=language,
                force_regenerate=request.force_regenerate
            )

            if explanation:
                # Add metadata
                explanation['generated_at'] = datetime.now().isoformat()
                explanation['optimizer_version'] = 'v4.0'
                explanation['parallel_generated'] = True

            return explanation

        except Exception as e:
            logger.error(f"Error generating single language explanation ({language}): {str(e)}")
            return None

    def _check_bulk_cache(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Check cache for all requested languages at once."""
        cached_explanations = {}

        for language in request.target_languages:
            cache_key = self._generate_cache_key(request, language)
            cached_result = cache.get(cache_key)

            if cached_result:
                cached_explanations[language] = cached_result
                logger.debug(f"Cache hit for {language}: {request.symbol}")

        return cached_explanations

    def _cache_bulk_explanations(
        self,
        request: OptimizationRequest,
        explanations: Dict[str, Any]
    ) -> None:
        """Cache multiple explanations efficiently."""
        cache_operations = {}

        for language, explanation in explanations.items():
            cache_key = self._generate_cache_key(request, language)
            cache_operations[cache_key] = explanation

        # Bulk cache set
        cache.set_many(cache_operations, timeout=self.cache_ttl)
        logger.debug(f"Cached {len(cache_operations)} explanations for {request.symbol}")

    def _generate_cache_key(self, request: OptimizationRequest, language: str) -> str:
        """Generate cache key for a specific request and language using unified system."""
        try:
            cache_key_generator = get_cache_key_generator()
            return cache_key_generator.generate_explanation_key(
                analysis_data=request.analysis_data,
                detail_level=request.detail_level,
                language=language,
                explanation_type="multilingual_optimization",
                user_id=request.user_id
            )
        except Exception as e:
            logger.warning(f"Error generating unified cache key, falling back: {str(e)}")
            # Fallback to original method
            key_components = [
                'multilingual_optimizer',
                str(request.analysis_id),
                request.symbol,
                request.detail_level,
                language,
                'v4.0'
            ]
            key_string = ':'.join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_request_hash(self, request: OptimizationRequest) -> str:
        """Generate hash for request deduplication."""
        hash_components = [
            str(request.analysis_id),
            request.symbol,
            request.detail_level,
            ':'.join(sorted(request.target_languages)),
            str(request.force_regenerate)
        ]

        hash_string = ':'.join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _is_duplicate_request(self, request_hash: str) -> bool:
        """Check if this request is already being processed."""
        with self._request_lock:
            return request_hash in self._active_requests

    def _mark_request_active(self, request_hash: str) -> None:
        """Mark request as actively being processed."""
        with self._request_lock:
            self._active_requests.add(request_hash)

    def _mark_request_complete(self, request_hash: str) -> None:
        """Mark request as completed."""
        with self._request_lock:
            self._active_requests.discard(request_hash)

    def _wait_for_duplicate_completion(
        self,
        request_hash: str,
        request: OptimizationRequest
    ) -> OptimizationResult:
        """Wait for duplicate request to complete and return cached result."""
        max_wait_time = 30  # 30 seconds max wait
        wait_interval = 0.5  # Check every 500ms
        waited_time = 0

        while waited_time < max_wait_time:
            if not self._is_duplicate_request(request_hash):
                # Request completed, try to get from cache
                cached_explanations = self._check_bulk_cache(request)
                if cached_explanations:
                    return OptimizationResult(
                        analysis_id=request.analysis_id,
                        symbol=request.symbol,
                        explanations=cached_explanations,
                        performance_metrics={'from_duplicate_cache': True},
                        cache_stats={'all_cached': True},
                        processing_time=waited_time,
                        success=True
                    )

            time.sleep(wait_interval)
            waited_time += wait_interval

        # Timeout waiting for duplicate
        logger.warning(f"Timeout waiting for duplicate request: {request.symbol}")
        return OptimizationResult(
            analysis_id=request.analysis_id,
            symbol=request.symbol,
            explanations={},
            performance_metrics={},
            cache_stats={},
            processing_time=waited_time,
            success=False,
            errors=["Timeout waiting for duplicate request completion"]
        )

    def _validate_request(self, request: OptimizationRequest) -> bool:
        """Validate optimization request parameters."""
        try:
            if not request.target_languages:
                raise LanguageNotSupportedException(
                    "",
                    self.supported_languages,
                    {"validation_error": "No target languages specified"}
                )

            # Check supported languages
            unsupported_languages = []
            for lang in request.target_languages:
                if lang not in self.supported_languages:
                    unsupported_languages.append(lang)

            if unsupported_languages:
                raise LanguageNotSupportedException(
                    unsupported_languages[0],  # First unsupported language
                    self.supported_languages,
                    {"unsupported_languages": unsupported_languages}
                )

            # Check detail level
            if request.detail_level not in ['summary', 'standard', 'detailed']:
                raise MultilingualBaseException(
                    f"Invalid detail level: {request.detail_level}",
                    error_code="INVALID_DETAIL_LEVEL",
                    details={
                        "requested_detail_level": request.detail_level,
                        "valid_detail_levels": ['summary', 'standard', 'detailed']
                    },
                    recovery_hint="Use one of the valid detail levels: summary, standard, or detailed"
                )

            # Check analysis data
            if not request.analysis_data:
                raise MultilingualBaseException(
                    "Analysis data is required but not provided",
                    error_code="MISSING_ANALYSIS_DATA",
                    recovery_hint="Ensure analysis data is included in the request"
                )

            return True

        except MultilingualBaseException:
            # Re-raise multilingual exceptions
            raise
        except Exception as e:
            # Convert unexpected validation errors
            raise MultilingualBaseException(
                f"Request validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"original_error": str(e)},
                recovery_hint="Check request parameters and try again"
            )

    def _calculate_performance_metrics(
        self,
        request: OptimizationRequest,
        explanations: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """Calculate performance metrics for the request."""
        num_languages = len(request.target_languages)
        num_generated = len(explanations)

        # Calculate parallel efficiency
        sequential_estimate = num_languages * 2.5  # Assume 2.5s per language sequentially
        parallel_efficiency = (sequential_estimate / processing_time) if processing_time > 0 else 0

        return {
            'languages_requested': num_languages,
            'languages_generated': num_generated,
            'success_rate': (num_generated / num_languages) if num_languages > 0 else 0,
            'processing_time': processing_time,
            'parallel_efficiency': parallel_efficiency,
            'avg_time_per_language': processing_time / num_languages if num_languages > 0 else 0,
            'memory_efficient': processing_time < 10.0,  # Under 10 seconds is efficient
            'cache_effective': any('cached' in str(exp) for exp in explanations.values())
        }

    def _update_global_metrics(
        self,
        processing_time: float,
        cache_stats: Dict[str, Any],
        explanations: Dict[str, Any]
    ) -> None:
        """Update global performance metrics."""
        with self._metrics_lock:
            self._performance_metrics['requests_processed'] += 1
            self._performance_metrics['total_processing_time'] += processing_time
            self._performance_metrics['cache_hits'] += cache_stats.get('hits', 0)
            self._performance_metrics['cache_misses'] += cache_stats.get('misses', 0)

            # Calculate parallel efficiency for this request
            num_languages = len(explanations)
            if num_languages > 1:
                sequential_estimate = num_languages * 2.5
                efficiency = (sequential_estimate / processing_time) if processing_time > 0 else 0
                self._performance_metrics['parallel_efficiency'].append(efficiency)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._metrics_lock:
            metrics = self._performance_metrics.copy()

            # Calculate derived metrics
            if metrics['requests_processed'] > 0:
                metrics['avg_processing_time'] = (
                    metrics['total_processing_time'] / metrics['requests_processed']
                )

                total_cache_requests = metrics['cache_hits'] + metrics['cache_misses']
                if total_cache_requests > 0:
                    metrics['cache_hit_rate'] = metrics['cache_hits'] / total_cache_requests
                else:
                    metrics['cache_hit_rate'] = 0.0

                if metrics['parallel_efficiency']:
                    metrics['avg_parallel_efficiency'] = statistics.mean(metrics['parallel_efficiency'])
                else:
                    metrics['avg_parallel_efficiency'] = 0.0

            return metrics

    def warm_cache(
        self,
        symbols: List[str],
        languages: List[str] = None,
        detail_levels: List[str] = None
    ) -> Dict[str, Any]:
        """Warm cache for common symbols and languages."""
        languages = languages or self.supported_languages
        detail_levels = detail_levels or ['summary', 'standard']

        start_time = time.time()
        warmed_count = 0

        logger.info(f"Starting cache warming for {len(symbols)} symbols")

        for symbol in symbols:
            try:
                # Mock analysis data for cache warming
                analysis_data = {
                    'symbol': symbol,
                    'score_0_10': 7.0,
                    'weighted_scores': {'sma50vs200': 0.7},
                    'indicators': {'sma50': 180.0, 'sma200': 175.0}
                }

                for detail_level in detail_levels:
                    request = OptimizationRequest(
                        analysis_id=0,  # Cache warming doesn't need real ID
                        symbol=symbol,
                        analysis_data=analysis_data,
                        target_languages=languages,
                        detail_level=detail_level
                    )

                    # Process in smaller batches to avoid memory issues
                    for i in range(0, len(languages), 2):
                        batch_languages = languages[i:i+2]
                        batch_request = OptimizationRequest(
                            analysis_id=0,
                            symbol=symbol,
                            analysis_data=analysis_data,
                            target_languages=batch_languages,
                            detail_level=detail_level
                        )

                        result = self.process_multilingual_request(batch_request)
                        if result.success:
                            warmed_count += len(result.explanations)

                        # Small delay to prevent overwhelming the system
                        time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error warming cache for {symbol}: {str(e)}")

        warming_time = time.time() - start_time

        logger.info(f"Cache warming completed: {warmed_count} explanations in {warming_time:.2f}s")

        return {
            'warmed_explanations': warmed_count,
            'warming_time': warming_time,
            'symbols_processed': len(symbols),
            'languages_processed': len(languages),
            'detail_levels_processed': len(detail_levels)
        }

    def _calculate_optimal_pool_size(self) -> int:
        """Calculate optimal thread pool size based on system resources and load."""
        if not self.enable_dynamic_sizing:
            return self._current_pool_size

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            # Calculate base pool size based on CPU cores
            cpu_count = os.cpu_count() or 4
            base_pool_size = min(cpu_count, self.max_workers)

            # Adjust based on CPU load
            if cpu_percent > self.cpu_threshold_high:
                # High CPU load - reduce pool size
                adjustment_factor = 0.7
            elif cpu_percent < self.cpu_threshold_low:
                # Low CPU load - potentially increase pool size
                adjustment_factor = 1.3
            else:
                # Normal CPU load - keep current size
                adjustment_factor = 1.0

            # Adjust based on memory usage
            if memory_percent > self.memory_threshold_high:
                # High memory usage - be more conservative
                adjustment_factor *= 0.8

            # Consider recent performance history
            if len(self._pool_performance_history) >= 3:
                # Calculate average efficiency of recent requests
                recent_efficiency = statistics.mean(self._pool_performance_history[-3:])
                if recent_efficiency < 0.5:  # Poor efficiency
                    adjustment_factor *= 0.9
                elif recent_efficiency > 0.8:  # Good efficiency
                    adjustment_factor *= 1.1

            # Calculate new pool size
            new_pool_size = int(base_pool_size * adjustment_factor)

            # Apply constraints
            new_pool_size = max(self.min_workers, min(new_pool_size, self.max_workers))

            # Avoid frequent small adjustments
            if abs(new_pool_size - self._current_pool_size) < 1:
                new_pool_size = self._current_pool_size

            return new_pool_size

        except Exception as e:
            logger.warning(f"Error calculating optimal pool size: {str(e)}")
            return self._current_pool_size

    def _should_adjust_pool_size(self) -> bool:
        """Check if enough time has passed to adjust pool size."""
        return (time.time() - self._last_pool_adjustment) >= self.pool_adjustment_interval

    def _get_dynamic_pool_size(self, request: OptimizationRequest) -> int:
        """Get the appropriate pool size for the current request."""
        if not self.enable_dynamic_sizing:
            return self._current_pool_size

        # Check if it's time to recalculate pool size
        if self._should_adjust_pool_size():
            optimal_size = self._calculate_optimal_pool_size()

            if optimal_size != self._current_pool_size:
                logger.info(f"Adjusting thread pool size: {self._current_pool_size} -> {optimal_size}")
                self._current_pool_size = optimal_size
                self._last_pool_adjustment = time.time()

        # For this specific request, consider the number of target languages
        languages_count = len(request.target_languages)
        request_optimal_size = min(languages_count, self._current_pool_size)

        return max(1, request_optimal_size)

    def _record_pool_performance(self, request: OptimizationRequest, processing_time: float, success_rate: float):
        """Record performance metrics for pool size optimization."""
        try:
            # Calculate efficiency metric (lower time and higher success rate = better)
            if processing_time > 0:
                efficiency = success_rate / processing_time
            else:
                efficiency = success_rate

            # Keep only recent history (last 10 requests)
            self._pool_performance_history.append(efficiency)
            if len(self._pool_performance_history) > 10:
                self._pool_performance_history.pop(0)

            # Log performance for debugging
            logger.debug(f"Pool performance recorded: efficiency={efficiency:.3f}, "
                        f"pool_size={self._current_pool_size}, "
                        f"languages={len(request.target_languages)}")

        except Exception as e:
            logger.warning(f"Error recording pool performance: {str(e)}")

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get current thread pool statistics for monitoring."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()

            return {
                'current_pool_size': self._current_pool_size,
                'max_workers': self.max_workers,
                'min_workers': self.min_workers,
                'dynamic_sizing_enabled': self.enable_dynamic_sizing,
                'system_metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'available_memory_mb': memory_info.available / (1024 * 1024)
                },
                'thresholds': {
                    'cpu_high': self.cpu_threshold_high,
                    'cpu_low': self.cpu_threshold_low,
                    'memory_high': self.memory_threshold_high
                },
                'performance_history': {
                    'recent_efficiency': (
                        statistics.mean(self._pool_performance_history[-5:])
                        if len(self._pool_performance_history) >= 5 else 0
                    ),
                    'samples_count': len(self._pool_performance_history)
                },
                'last_adjustment': self._last_pool_adjustment,
                'next_adjustment_eligible': self._should_adjust_pool_size()
            }
        except Exception as e:
            logger.warning(f"Error getting pool statistics: {str(e)}")
            return {
                'current_pool_size': self._current_pool_size,
                'error': str(e)
            }


class ResourceMonitor:
    """Enhanced resource monitor with proactive memory monitoring and adaptive batch sizing."""

    def __init__(self):
        """Initialize resource monitor."""
        self.memory_samples = []
        self.max_samples = 100

        # Memory thresholds and adaptive sizing
        self.memory_warning_threshold = getattr(settings, 'MULTILINGUAL_MEMORY_WARNING_MB', 800)
        self.memory_critical_threshold = getattr(settings, 'MULTILINGUAL_MEMORY_CRITICAL_MB', 1200)
        self.adaptive_batch_sizing = getattr(settings, 'MULTILINGUAL_ADAPTIVE_BATCH_SIZING', True)

        # Batch size management
        self.default_batch_size = 3
        self.min_batch_size = 1
        self.max_batch_size = 6
        self.current_batch_size = self.default_batch_size

        # Memory pressure tracking
        self.memory_pressure_history = []
        self.max_pressure_samples = 20

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Keep rolling average
            self.memory_samples.append(memory_mb)
            if len(self.memory_samples) > self.max_samples:
                self.memory_samples.pop(0)

            # Track memory pressure
            self._track_memory_pressure(memory_mb)

            return memory_mb
        except ImportError:
            # psutil not available, return mock value
            return 50.0
        except Exception:
            return 0.0

    def get_average_memory_usage(self) -> float:
        """Get average memory usage."""
        if self.memory_samples:
            return sum(self.memory_samples) / len(self.memory_samples)
        return 0.0

    def get_system_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive system memory information."""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                'system_total_mb': memory_info.total / (1024 * 1024),
                'system_available_mb': memory_info.available / (1024 * 1024),
                'system_used_percent': memory_info.percent,
                'process_rss_mb': process_memory.rss / (1024 * 1024),
                'process_vms_mb': process_memory.vms / (1024 * 1024),
                'memory_pressure_level': self.get_memory_pressure_level(),
                'recommended_batch_size': self.get_adaptive_batch_size()
            }
        except Exception as e:
            logger.warning(f"Error getting system memory info: {str(e)}")
            return {
                'error': str(e),
                'memory_pressure_level': 'unknown',
                'recommended_batch_size': self.default_batch_size
            }

    def _track_memory_pressure(self, current_memory_mb: float):
        """Track memory pressure for adaptive sizing."""
        try:
            # Calculate pressure level (0.0 = no pressure, 1.0 = critical pressure)
            if current_memory_mb < self.memory_warning_threshold:
                pressure = 0.0
            elif current_memory_mb < self.memory_critical_threshold:
                # Linear scaling between warning and critical
                pressure = (current_memory_mb - self.memory_warning_threshold) / \
                          (self.memory_critical_threshold - self.memory_warning_threshold)
            else:
                pressure = 1.0

            # Track pressure history
            self.memory_pressure_history.append(pressure)
            if len(self.memory_pressure_history) > self.max_pressure_samples:
                self.memory_pressure_history.pop(0)

            # Update adaptive batch size
            if self.adaptive_batch_sizing:
                self._update_adaptive_batch_size(pressure)

        except Exception as e:
            logger.warning(f"Error tracking memory pressure: {str(e)}")

    def _update_adaptive_batch_size(self, current_pressure: float):
        """Update batch size based on memory pressure."""
        try:
            # Get average pressure over recent samples
            if len(self.memory_pressure_history) >= 5:
                avg_pressure = statistics.mean(self.memory_pressure_history[-5:])
            else:
                avg_pressure = current_pressure

            # Adjust batch size based on pressure
            if avg_pressure > 0.8:  # High pressure
                new_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
            elif avg_pressure > 0.5:  # Medium pressure
                new_batch_size = max(self.min_batch_size,
                                   min(self.default_batch_size, self.current_batch_size))
            elif avg_pressure < 0.2:  # Low pressure
                new_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            else:
                new_batch_size = self.current_batch_size

            if new_batch_size != self.current_batch_size:
                logger.debug(f"Adaptive batch size changed: {self.current_batch_size} -> {new_batch_size} "
                           f"(pressure: {avg_pressure:.2f})")
                self.current_batch_size = new_batch_size

        except Exception as e:
            logger.warning(f"Error updating adaptive batch size: {str(e)}")

    def get_memory_pressure_level(self) -> str:
        """Get current memory pressure level."""
        if not self.memory_pressure_history:
            return 'unknown'

        current_pressure = self.memory_pressure_history[-1]

        if current_pressure < 0.3:
            return 'low'
        elif current_pressure < 0.7:
            return 'medium'
        else:
            return 'high'

    def get_adaptive_batch_size(self) -> int:
        """Get current adaptive batch size."""
        return self.current_batch_size

    def should_throttle_processing(self) -> bool:
        """Check if processing should be throttled due to resource constraints."""
        try:
            current_memory = self.get_memory_usage()

            # Check if we're above critical threshold
            if current_memory > self.memory_critical_threshold:
                return True

            # Check if system memory is very high
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 90:
                return True

            # Check sustained high pressure
            if len(self.memory_pressure_history) >= 5:
                recent_pressure = self.memory_pressure_history[-5:]
                if all(p > 0.8 for p in recent_pressure):
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking throttle conditions: {str(e)}")
            return False

    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource monitoring statistics."""
        try:
            memory_info = self.get_system_memory_info()

            return {
                'memory_monitoring': {
                    'current_usage_mb': self.get_memory_usage(),
                    'average_usage_mb': self.get_average_memory_usage(),
                    'pressure_level': self.get_memory_pressure_level(),
                    'should_throttle': self.should_throttle_processing(),
                    'samples_count': len(self.memory_samples),
                    'pressure_samples_count': len(self.memory_pressure_history)
                },
                'adaptive_batching': {
                    'enabled': self.adaptive_batch_sizing,
                    'current_batch_size': self.current_batch_size,
                    'default_batch_size': self.default_batch_size,
                    'min_batch_size': self.min_batch_size,
                    'max_batch_size': self.max_batch_size
                },
                'thresholds': {
                    'memory_warning_mb': self.memory_warning_threshold,
                    'memory_critical_mb': self.memory_critical_threshold
                },
                'system_info': memory_info
            }
        except Exception as e:
            logger.warning(f"Error getting resource statistics: {str(e)}")
            return {
                'error': str(e),
                'current_batch_size': self.current_batch_size
            }


# Global optimizer instance
_optimizer_instance = None
_optimizer_lock = threading.Lock()


def get_multilingual_optimizer() -> MultilingualOptimizer:
    """Get the global multilingual optimizer instance."""
    global _optimizer_instance

    if _optimizer_instance is None:
        with _optimizer_lock:
            if _optimizer_instance is None:
                _optimizer_instance = MultilingualOptimizer()

    return _optimizer_instance