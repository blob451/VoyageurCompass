"""
Model preloading and warm-up service for optimized LLM performance.
Implements intelligent model caching, preloading strategies, and warm-up routines.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from django.core.cache import cache
from django.conf import settings

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.model_resource_manager import get_model_resource_manager

logger = logging.getLogger(__name__)


class ModelPreloadConfig:
    """Configuration for model preloading strategies."""

    def __init__(self):
        self.preload_on_startup = getattr(settings, "LLM_PRELOAD_ON_STARTUP", True)
        self.warmup_on_startup = getattr(settings, "LLM_WARMUP_ON_STARTUP", True)
        self.auto_preload_threshold = getattr(settings, "LLM_AUTO_PRELOAD_THRESHOLD", 5)  # requests per hour
        self.warmup_batch_size = getattr(settings, "LLM_WARMUP_BATCH_SIZE", 3)
        self.preload_cache_ttl = getattr(settings, "LLM_PRELOAD_CACHE_TTL", 3600)  # 1 hour
        self.warmup_timeout = getattr(settings, "LLM_WARMUP_TIMEOUT", 300)  # 5 minutes

        # Default models to preload
        self.default_preload_models = [
            "phi3:3.8b",
            "llama3.1:8b",
            "qwen2:latest"
        ]


class ModelWarmupService:
    """Service for intelligent model preloading and warm-up operations."""

    def __init__(self):
        self.config = ModelPreloadConfig()
        self.llm_service = get_local_llm_service()
        self.resource_manager = get_model_resource_manager()

        # Track model usage patterns
        self.model_usage_stats = {}
        self.preloaded_models: Set[str] = set()
        self.warmup_history: Dict[str, datetime] = {}

        # Thread pool for warmup operations
        self.warmup_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="model_warmup"
        )

        # Warmup status tracking
        self.warmup_status = {
            "in_progress": set(),
            "completed": set(),
            "failed": set(),
            "last_warmup_time": None
        }
        self._status_lock = threading.Lock()

        # Performance metrics
        self.metrics = {
            "total_warmups": 0,
            "successful_warmups": 0,
            "failed_warmups": 0,
            "average_warmup_time": 0.0,
            "models_preloaded": 0,
            "cache_hits": 0
        }
        self._metrics_lock = threading.Lock()

        logger.info("ModelWarmupService initialized")

    def startup_initialization(self) -> Dict[str, Any]:
        """Initialize models on service startup."""
        logger.info("Starting model initialization sequence")

        initialization_results = {
            "preload_results": {},
            "warmup_results": {},
            "total_time": 0.0,
            "status": "starting"
        }

        start_time = time.time()

        try:
            # Preload models based on configuration
            if self.config.preload_on_startup:
                logger.info("Preloading models on startup")
                preload_results = self.preload_priority_models()
                initialization_results["preload_results"] = preload_results

            # Warm up models
            if self.config.warmup_on_startup:
                logger.info("Warming up models on startup")
                warmup_results = asyncio.run(self.warmup_models_async(
                    models=self.config.default_preload_models,
                    batch_size=self.config.warmup_batch_size
                ))
                initialization_results["warmup_results"] = warmup_results

            initialization_results["total_time"] = time.time() - start_time
            initialization_results["status"] = "completed"

            logger.info(f"Model initialization completed in {initialization_results['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            initialization_results["status"] = "failed"
            initialization_results["error"] = str(e)

        return initialization_results

    def preload_priority_models(self) -> Dict[str, Any]:
        """Preload high-priority models based on usage patterns."""

        # Get models to preload
        models_to_preload = self._determine_preload_candidates()

        if not models_to_preload:
            logger.info("No models identified for preloading")
            return {"preloaded": 0, "models": []}

        logger.info(f"Preloading {len(models_to_preload)} priority models")

        preload_results = {
            "preloaded": 0,
            "models": [],
            "failures": [],
            "resource_check": None
        }

        # Check resource availability
        if self.resource_manager:
            resource_status = self.resource_manager.get_resource_status()
            preload_results["resource_check"] = resource_status

            # Limit preloading based on available resources
            available_memory = resource_status.get("available_memory_gb", 0)
            max_preload = min(len(models_to_preload), int(available_memory / 2))  # 2GB per model estimate
            models_to_preload = models_to_preload[:max_preload]

        # Preload each model
        for model_name in models_to_preload:
            try:
                success = self._preload_single_model(model_name)
                if success:
                    self.preloaded_models.add(model_name)
                    preload_results["preloaded"] += 1
                    preload_results["models"].append(model_name)

                    with self._metrics_lock:
                        self.metrics["models_preloaded"] += 1
                else:
                    preload_results["failures"].append({
                        "model": model_name,
                        "error": "Preload returned False"
                    })

            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {str(e)}")
                preload_results["failures"].append({
                    "model": model_name,
                    "error": str(e)
                })

        logger.info(f"Preloading completed: {preload_results['preloaded']} models loaded")
        return preload_results

    def _determine_preload_candidates(self) -> List[str]:
        """Determine which models should be preloaded based on usage patterns."""

        candidates = []

        # Always include default models
        candidates.extend(self.config.default_preload_models)

        # Add models with high recent usage
        usage_data = self._get_model_usage_patterns()
        for model, stats in usage_data.items():
            recent_usage = stats.get("requests_last_hour", 0)
            if recent_usage >= self.config.auto_preload_threshold:
                if model not in candidates:
                    candidates.append(model)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for model in candidates:
            if model not in seen:
                seen.add(model)
                unique_candidates.append(model)

        return unique_candidates

    def _preload_single_model(self, model_name: str) -> bool:
        """Preload a single model."""
        try:
            # Check if model is already loaded
            if self.llm_service.is_model_loaded(model_name):
                logger.debug(f"Model {model_name} already loaded")
                return True

            # Check resource constraints
            if self.resource_manager:
                should_load, reason = self.resource_manager.should_load_model(model_name)
                if not should_load:
                    logger.warning(f"Resource manager prevents loading {model_name}: {reason}")
                    return False

            # Load the model
            logger.info(f"Preloading model: {model_name}")
            success = self.llm_service.load_model(model_name)

            if success:
                # Cache preload status
                cache_key = f"model_preloaded:{model_name}"
                cache.set(cache_key, True, self.config.preload_cache_ttl)
                logger.info(f"Successfully preloaded model: {model_name}")
            else:
                logger.warning(f"Failed to preload model: {model_name}")

            return success

        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {str(e)}")
            return False

    async def warmup_models_async(
        self,
        models: List[str] = None,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """Warm up models asynchronously with dummy requests."""

        models = models or self.config.default_preload_models
        batch_size = batch_size or self.config.warmup_batch_size

        logger.info(f"Starting async warmup for {len(models)} models")

        warmup_results = {
            "total_models": len(models),
            "successful_warmups": 0,
            "failed_warmups": 0,
            "warmup_times": {},
            "errors": []
        }

        # Update status
        with self._status_lock:
            self.warmup_status["in_progress"].update(models)
            self.warmup_status["last_warmup_time"] = datetime.now()

        # Create warmup tasks
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [
            self._warmup_single_model_async(semaphore, model, warmup_results)
            for model in models
        ]

        # Execute warmup tasks
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.warmup_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Model warmup timed out after {self.config.warmup_timeout}s")
            warmup_results["errors"].append("Warmup timeout exceeded")

        # Update metrics
        with self._metrics_lock:
            self.metrics["total_warmups"] += len(models)
            self.metrics["successful_warmups"] += warmup_results["successful_warmups"]
            self.metrics["failed_warmups"] += warmup_results["failed_warmups"]

        logger.info(
            f"Model warmup completed: {warmup_results['successful_warmups']}/{len(models)} successful"
        )

        return warmup_results

    async def _warmup_single_model_async(
        self,
        semaphore: asyncio.Semaphore,
        model_name: str,
        results_dict: Dict[str, Any]
    ):
        """Warm up a single model with semaphore control."""
        async with semaphore:
            start_time = time.time()

            try:
                # Run warmup in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self.warmup_executor,
                    self._execute_model_warmup,
                    model_name
                )

                warmup_time = time.time() - start_time
                results_dict["warmup_times"][model_name] = warmup_time

                if success:
                    results_dict["successful_warmups"] += 1
                    with self._status_lock:
                        self.warmup_status["completed"].add(model_name)
                        self.warmup_status["in_progress"].discard(model_name)

                    # Update warmup history
                    self.warmup_history[model_name] = datetime.now()

                    logger.info(f"Model {model_name} warmed up successfully in {warmup_time:.2f}s")
                else:
                    results_dict["failed_warmups"] += 1
                    with self._status_lock:
                        self.warmup_status["failed"].add(model_name)
                        self.warmup_status["in_progress"].discard(model_name)

                    logger.warning(f"Model {model_name} warmup failed")

            except Exception as e:
                logger.error(f"Error warming up model {model_name}: {str(e)}")
                results_dict["failed_warmups"] += 1
                results_dict["errors"].append(f"{model_name}: {str(e)}")

                with self._status_lock:
                    self.warmup_status["failed"].add(model_name)
                    self.warmup_status["in_progress"].discard(model_name)

    def _execute_model_warmup(self, model_name: str) -> bool:
        """Execute warmup for a single model with dummy inference."""
        try:
            # Create dummy analysis data for warmup
            dummy_analysis_data = {
                "symbol": "WARMUP",
                "score_0_10": 5.0,
                "recommendation": "HOLD",
                "technical_score": 5.0,
                "indicators": {
                    "rsi": {"value": 50.0, "signal": "neutral"},
                    "macd": {"value": 0.0, "signal": "neutral"}
                },
                "analysis_date": datetime.now().isoformat()
            }

            # Set the model for warmup
            original_model = self.llm_service.current_model
            self.llm_service.set_model(model_name)

            try:
                # Generate a dummy explanation to warm up the model
                from Analytics.services.explanation_service import get_explanation_service
                explanation_service = get_explanation_service()

                result = explanation_service.explain_prediction_single(
                    dummy_analysis_data,
                    detail_level="summary"
                )

                # Check if warmup was successful
                success = bool(result and result.get("content"))

                if success:
                    logger.debug(f"Model {model_name} warmup generated response")
                else:
                    logger.warning(f"Model {model_name} warmup did not generate response")

                return success

            finally:
                # Restore original model
                self.llm_service.set_model(original_model)

        except Exception as e:
            logger.error(f"Warmup execution failed for {model_name}: {str(e)}")
            return False

    def _get_model_usage_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get model usage patterns from cache and statistics."""

        # Try to get usage data from cache
        usage_data = cache.get("model_usage_patterns", {})

        # If no cached data, return empty patterns
        if not usage_data:
            return {}

        # Filter for recent usage (last hour)
        current_time = datetime.now()
        filtered_data = {}

        for model, stats in usage_data.items():
            if isinstance(stats, dict):
                # Calculate recent usage
                recent_requests = 0
                request_times = stats.get("request_times", [])

                for request_time_str in request_times:
                    try:
                        request_time = datetime.fromisoformat(request_time_str)
                        if current_time - request_time <= timedelta(hours=1):
                            recent_requests += 1
                    except ValueError:
                        continue

                filtered_data[model] = {
                    **stats,
                    "requests_last_hour": recent_requests
                }

        return filtered_data

    def register_model_usage(self, model_name: str, success: bool = True):
        """Register model usage for usage pattern tracking."""
        try:
            # Get current usage data
            usage_data = cache.get("model_usage_patterns", {})

            if model_name not in usage_data:
                usage_data[model_name] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "request_times": []
                }

            # Update statistics
            usage_data[model_name]["total_requests"] += 1
            if success:
                usage_data[model_name]["successful_requests"] += 1
            else:
                usage_data[model_name]["failed_requests"] += 1

            # Add timestamp
            usage_data[model_name]["request_times"].append(
                datetime.now().isoformat()
            )

            # Keep only last 100 request times to prevent memory bloat
            usage_data[model_name]["request_times"] = \
                usage_data[model_name]["request_times"][-100:]

            # Update cache
            cache.set("model_usage_patterns", usage_data, 86400)  # 24 hours

        except Exception as e:
            logger.error(f"Failed to register model usage for {model_name}: {str(e)}")

    def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup status."""
        with self._status_lock:
            return {
                "in_progress": list(self.warmup_status["in_progress"]),
                "completed": list(self.warmup_status["completed"]),
                "failed": list(self.warmup_status["failed"]),
                "last_warmup_time": self.warmup_status["last_warmup_time"].isoformat()
                    if self.warmup_status["last_warmup_time"] else None,
                "preloaded_models": list(self.preloaded_models)
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get warmup service performance metrics."""
        with self._metrics_lock:
            metrics = self.metrics.copy()

        # Add additional status information
        metrics.update({
            "preloaded_models_count": len(self.preloaded_models),
            "warmup_history_count": len(self.warmup_history),
            "config": {
                "preload_on_startup": self.config.preload_on_startup,
                "warmup_on_startup": self.config.warmup_on_startup,
                "auto_preload_threshold": self.config.auto_preload_threshold
            }
        })

        return metrics

    def force_warmup_model(self, model_name: str) -> Dict[str, Any]:
        """Force warmup of a specific model."""
        logger.info(f"Force warming up model: {model_name}")

        start_time = time.time()
        success = self._execute_model_warmup(model_name)
        warmup_time = time.time() - start_time

        result = {
            "model": model_name,
            "success": success,
            "warmup_time": warmup_time
        }

        if success:
            with self._status_lock:
                self.warmup_status["completed"].add(model_name)
                self.warmup_status["failed"].discard(model_name)

            self.warmup_history[model_name] = datetime.now()
        else:
            with self._status_lock:
                self.warmup_status["failed"].add(model_name)

        return result

    def shutdown(self):
        """Shutdown the warmup service."""
        logger.info("Shutting down model warmup service")
        self.warmup_executor.shutdown(wait=True)


# Singleton instance
_model_warmup_service = None


def get_model_warmup_service() -> ModelWarmupService:
    """Get singleton instance of ModelWarmupService."""
    global _model_warmup_service
    if _model_warmup_service is None:
        _model_warmup_service = ModelWarmupService()
    return _model_warmup_service


# Auto-initialize on import if configured
if getattr(settings, "LLM_AUTO_INITIALIZE_WARMUP", False):
    warmup_service = get_model_warmup_service()
    warmup_service.startup_initialization()