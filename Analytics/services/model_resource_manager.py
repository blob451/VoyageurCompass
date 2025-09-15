"""
Model resource management system for LLM services.
Provides intelligent model loading, unloading, and resource optimization.
"""

import logging
import threading
import time
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import psutil
import gc

from django.conf import settings

logger = logging.getLogger(__name__)

# Conditional import for ollama
try:
    import ollama
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Ollama not available - model resource manager will operate in fallback mode")
    ollama = None
    Client = None
    OLLAMA_AVAILABLE = False


class ModelUsageTracker:
    """Tracks model usage patterns for intelligent resource management."""

    def __init__(self):
        self.usage_stats = defaultdict(lambda: {
            "total_requests": 0,
            "last_used": None,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "error_count": 0,
            "success_rate": 0.0,
            "memory_usage_mb": 0.0
        })
        self.usage_history = defaultdict(list)  # Time-series data
        self.lock = threading.Lock()

    def record_usage(self, model_name: str, response_time: float, success: bool,
                    memory_usage_mb: float = 0.0) -> None:
        """Record model usage statistics."""
        with self.lock:
            stats = self.usage_stats[model_name]
            current_time = datetime.now()

            # Update basic stats
            stats["total_requests"] += 1
            stats["last_used"] = current_time

            if success:
                stats["total_response_time"] += response_time
                stats["average_response_time"] = (
                    stats["total_response_time"] / (stats["total_requests"] - stats["error_count"])
                )
            else:
                stats["error_count"] += 1

            # Update success rate
            stats["success_rate"] = (
                (stats["total_requests"] - stats["error_count"]) / stats["total_requests"]
            )

            # Update memory usage
            if memory_usage_mb > 0:
                stats["memory_usage_mb"] = memory_usage_mb

            # Add to history (keep last 100 entries)
            self.usage_history[model_name].append({
                "timestamp": current_time,
                "response_time": response_time,
                "success": success,
                "memory_usage_mb": memory_usage_mb
            })

            # Trim history to last 100 entries
            if len(self.usage_history[model_name]) > 100:
                self.usage_history[model_name] = self.usage_history[model_name][-100:]

    def get_usage_stats(self, model_name: str) -> Dict[str, Any]:
        """Get usage statistics for a model."""
        with self.lock:
            return dict(self.usage_stats[model_name])

    def get_least_recently_used_models(self, count: int = 1) -> List[str]:
        """Get the least recently used models."""
        with self.lock:
            # Sort by last_used timestamp
            sorted_models = sorted(
                [(name, stats) for name, stats in self.usage_stats.items()
                 if stats["last_used"] is not None],
                key=lambda x: x[1]["last_used"]
            )
            return [name for name, _ in sorted_models[:count]]

    def get_performance_metrics(self, time_window_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for models within a time window."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics = {}

            for model_name, history in self.usage_history.items():
                recent_history = [
                    entry for entry in history
                    if entry["timestamp"] >= cutoff_time
                ]

                if recent_history:
                    successful_entries = [e for e in recent_history if e["success"]]
                    total_requests = len(recent_history)
                    successful_requests = len(successful_entries)

                    metrics[model_name] = {
                        "recent_requests": total_requests,
                        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                        "avg_response_time": (
                            sum(e["response_time"] for e in successful_entries) / successful_requests
                            if successful_requests > 0 else 0
                        ),
                        "avg_memory_usage": (
                            sum(e["memory_usage_mb"] for e in recent_history) / total_requests
                            if total_requests > 0 else 0
                        )
                    }

            return metrics


class SystemResourceMonitor:
    """Monitors system resources for intelligent model management."""

    def __init__(self):
        self.memory_threshold_warning = 0.85  # 85% memory usage
        self.memory_threshold_critical = 0.95  # 95% memory usage
        self.cpu_threshold_warning = 0.80  # 80% CPU usage
        self.monitoring_interval = 30  # seconds

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent / 100,
                "free_gb": memory.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {str(e)}")
            return {"percent_used": 0.5}  # Default fallback

    def get_cpu_info(self) -> Dict[str, float]:
        """Get current CPU usage information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            return {
                "cpu_percent": cpu_percent / 100,
                "cpu_count": cpu_count,
                "load_1min": load_avg[0],
                "load_5min": load_avg[1],
                "load_15min": load_avg[2]
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {str(e)}")
            return {"cpu_percent": 0.5}

    def is_memory_pressure(self) -> Tuple[bool, str]:
        """Check if system is under memory pressure."""
        memory_info = self.get_memory_info()
        memory_usage = memory_info.get("percent_used", 0.5)

        if memory_usage >= self.memory_threshold_critical:
            return True, "critical"
        elif memory_usage >= self.memory_threshold_warning:
            return True, "warning"
        else:
            return False, "normal"

    def is_cpu_pressure(self) -> Tuple[bool, str]:
        """Check if system is under CPU pressure."""
        cpu_info = self.get_cpu_info()
        cpu_usage = cpu_info.get("cpu_percent", 0.5)

        if cpu_usage >= self.cpu_threshold_warning:
            return True, "warning"
        else:
            return False, "normal"

    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        memory_pressure, memory_level = self.is_memory_pressure()
        cpu_pressure, cpu_level = self.is_cpu_pressure()

        return {
            "memory": self.get_memory_info(),
            "cpu": self.get_cpu_info(),
            "memory_pressure": memory_pressure,
            "memory_level": memory_level,
            "cpu_pressure": cpu_pressure,
            "cpu_level": cpu_level,
            "timestamp": datetime.now()
        }


class ModelResourceManager:
    """Main model resource management system."""

    def __init__(self):
        self.usage_tracker = ModelUsageTracker()
        self.resource_monitor = SystemResourceMonitor()

        # Configuration
        self.max_concurrent_models = getattr(settings, "MAX_CONCURRENT_MODELS", 3)
        self.model_memory_limit_gb = getattr(settings, "MODEL_MEMORY_LIMIT_GB", 8.0)
        self.model_idle_timeout_minutes = getattr(settings, "MODEL_IDLE_TIMEOUT_MINUTES", 30)

        # GPU acceleration integration
        self.gpu_detection = None
        self.gpu_available = False
        self.gpu_config = {}
        self._initialize_gpu_support()

        # Model state tracking
        self.loaded_models = OrderedDict()  # model_name -> load_info
        self.model_load_times = {}  # model_name -> load_timestamp
        self.model_memory_usage = {}  # model_name -> memory_usage_mb

        # Threading
        self.lock = threading.Lock()
        self.cleanup_thread = None
        self.cleanup_running = False

        # Start background cleanup
        self.start_background_cleanup()

    def _initialize_gpu_support(self) -> None:
        """Initialize GPU support if available."""
        try:
            import sys
            sys.path.append('..')
            from infrastructure.gpu_detection import GPUDetector

            self.gpu_detection = GPUDetector()
            gpu_result = self.gpu_detection.run_full_detection()

            self.gpu_available = gpu_result.get("gpu_available", False)
            self.gpu_config = gpu_result.get("configuration", {})

            if self.gpu_available:
                logger.info(f"GPU acceleration available: {gpu_result['gpu_info']}")
                # Update model limits based on GPU memory
                self._configure_gpu_limits(gpu_result["gpu_info"])
            else:
                logger.info("GPU acceleration not available, using CPU-only configuration")

        except ImportError:
            logger.info("GPU detection module not available")
        except Exception as e:
            logger.error(f"GPU initialization failed: {str(e)}")

    def _configure_gpu_limits(self, gpu_info: Dict[str, Any]) -> None:
        """Configure model limits based on GPU specifications."""
        if not gpu_info:
            return

        # Get first GPU memory info
        first_gpu = list(gpu_info.values())[0]
        gpu_memory_mb = first_gpu.get("memory_total_mb", 0)

        # Adjust model limits based on GPU memory
        if gpu_memory_mb >= 12000:  # 12GB+ (RTX 3080/4070 Ti/4080)
            self.max_concurrent_models = 3
            self.model_memory_limit_gb = 10.0
            logger.info(f"High-end GPU detected ({gpu_memory_mb}MB): 3 concurrent models")

        elif gpu_memory_mb >= 8000:  # 8GB+ (RTX 2070/3070/4060 Ti)
            self.max_concurrent_models = 2
            self.model_memory_limit_gb = 7.0
            logger.info(f"Mid-range GPU detected ({gpu_memory_mb}MB): 2 concurrent models")

        elif gpu_memory_mb >= 6000:  # 6GB+ (RTX 2060/3060)
            self.max_concurrent_models = 1
            self.model_memory_limit_gb = 5.5
            logger.info(f"Entry-level GPU detected ({gpu_memory_mb}MB): 1 concurrent model")

        else:  # Lower memory GPU
            self.max_concurrent_models = 1
            self.model_memory_limit_gb = 4.0
            logger.info(f"Low memory GPU detected ({gpu_memory_mb}MB): 1 concurrent model, reduced limits")

    def start_background_cleanup(self) -> None:
        """Start background thread for model cleanup."""
        if not self.cleanup_running:
            self.cleanup_running = True
            self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
            self.cleanup_thread.start()
            logger.info("Model resource manager background cleanup started")

    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

    def _background_cleanup(self) -> None:
        """Background thread for periodic model cleanup."""
        while self.cleanup_running:
            try:
                time.sleep(300)  # Check every 5 minutes
                self._cleanup_idle_models()
                self._handle_resource_pressure()
                gc.collect()  # Force garbage collection
            except Exception as e:
                logger.error(f"Error in background cleanup: {str(e)}")

    def should_load_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Determine if a model should be loaded based on resource constraints.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (should_load, reason)
        """
        with self.lock:
            # Check if model is already loaded
            if model_name in self.loaded_models:
                return True, "already_loaded"

            # Check maximum concurrent models
            if len(self.loaded_models) >= self.max_concurrent_models:
                # Check if we can unload least recently used model
                if self._can_unload_lru_model():
                    return True, "can_unload_lru"
                else:
                    return False, "max_concurrent_reached"

            # Check memory pressure
            memory_pressure, memory_level = self.resource_monitor.is_memory_pressure()
            if memory_pressure and memory_level == "critical":
                return False, "critical_memory_pressure"

            # Check available memory
            memory_info = self.resource_monitor.get_memory_info()
            if memory_info.get("available_gb", 0) < 2.0:  # Need at least 2GB free
                return False, "insufficient_memory"

            return True, "resources_available"

    def prepare_model_loading(self, model_name: str) -> bool:
        """
        Prepare system for model loading by freeing resources if needed.

        Args:
            model_name: Name of the model to load

        Returns:
            True if preparation successful, False otherwise
        """
        with self.lock:
            should_load, reason = self.should_load_model(model_name)

            if not should_load:
                if reason == "max_concurrent_reached":
                    # Try to unload LRU model
                    if self._unload_least_recently_used():
                        logger.info(f"Freed resources for {model_name} by unloading LRU model")
                        return True
                    else:
                        logger.warning(f"Cannot load {model_name}: {reason}")
                        return False

                elif reason == "critical_memory_pressure":
                    # Aggressive cleanup
                    self._emergency_cleanup()
                    # Recheck after cleanup
                    should_load, new_reason = self.should_load_model(model_name)
                    if should_load:
                        logger.info(f"Emergency cleanup successful for {model_name}")
                        return True
                    else:
                        logger.error(f"Emergency cleanup failed for {model_name}: {new_reason}")
                        return False

                else:
                    logger.warning(f"Cannot load {model_name}: {reason}")
                    return False

            return True

    def register_model_loaded(self, model_name: str, memory_usage_mb: float = 0.0) -> None:
        """Register that a model has been loaded."""
        with self.lock:
            self.loaded_models[model_name] = {
                "load_time": datetime.now(),
                "memory_usage_mb": memory_usage_mb,
                "last_used": datetime.now()
            }
            self.model_load_times[model_name] = datetime.now()
            self.model_memory_usage[model_name] = memory_usage_mb

            logger.info(
                f"Registered model {model_name} "
                f"(memory: {memory_usage_mb:.1f}MB, "
                f"total loaded: {len(self.loaded_models)})"
            )

    def register_model_usage(self, model_name: str, response_time: float, success: bool) -> None:
        """Register model usage for tracking."""
        # Update usage tracker
        memory_usage = self.model_memory_usage.get(model_name, 0.0)
        self.usage_tracker.record_usage(model_name, response_time, success, memory_usage)

        # Update last used time
        with self.lock:
            if model_name in self.loaded_models:
                self.loaded_models[model_name]["last_used"] = datetime.now()

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if unloaded successfully, False otherwise
        """
        with self.lock:
            if model_name not in self.loaded_models:
                return True  # Already unloaded

            try:
                # In a real implementation, this would call ollama client to unload
                # For now, we'll just remove from tracking
                model_info = self.loaded_models.pop(model_name)
                self.model_load_times.pop(model_name, None)
                self.model_memory_usage.pop(model_name, None)

                logger.info(
                    f"Unloaded model {model_name} "
                    f"(freed: {model_info['memory_usage_mb']:.1f}MB)"
                )

                # Force garbage collection
                gc.collect()

                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {str(e)}")
                return False

    def _can_unload_lru_model(self) -> bool:
        """Check if we can unload the least recently used model."""
        if len(self.loaded_models) == 0:
            return False

        # Find LRU model
        lru_model = min(
            self.loaded_models.items(),
            key=lambda x: x[1]["last_used"]
        )[0]

        # Check if LRU model is not recently used (> 5 minutes)
        lru_last_used = self.loaded_models[lru_model]["last_used"]
        if datetime.now() - lru_last_used > timedelta(minutes=5):
            return True

        return False

    def _unload_least_recently_used(self) -> bool:
        """Unload the least recently used model."""
        if len(self.loaded_models) == 0:
            return False

        lru_model = min(
            self.loaded_models.items(),
            key=lambda x: x[1]["last_used"]
        )[0]

        return self.unload_model(lru_model)

    def _cleanup_idle_models(self) -> None:
        """Clean up models that have been idle for too long."""
        with self.lock:
            current_time = datetime.now()
            idle_threshold = timedelta(minutes=self.model_idle_timeout_minutes)

            models_to_unload = []
            for model_name, model_info in self.loaded_models.items():
                if current_time - model_info["last_used"] > idle_threshold:
                    models_to_unload.append(model_name)

            for model_name in models_to_unload:
                logger.info(f"Unloading idle model: {model_name}")
                self.unload_model(model_name)

    def _handle_resource_pressure(self) -> None:
        """Handle resource pressure by unloading models."""
        memory_pressure, memory_level = self.resource_monitor.is_memory_pressure()

        if memory_pressure:
            logger.warning(f"Memory pressure detected: {memory_level}")

            if memory_level == "critical":
                self._emergency_cleanup()
            elif memory_level == "warning":
                # Unload one LRU model
                self._unload_least_recently_used()

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup - unload all but the most recently used model."""
        with self.lock:
            if len(self.loaded_models) <= 1:
                return

            # Sort by last used time, keep only the most recent
            sorted_models = sorted(
                self.loaded_models.items(),
                key=lambda x: x[1]["last_used"],
                reverse=True
            )

            # Unload all but the most recent
            for model_name, _ in sorted_models[1:]:
                logger.warning(f"Emergency unloading model: {model_name}")
                self.unload_model(model_name)

            # Force garbage collection
            gc.collect()

    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        with self.lock:
            loaded_models_info = {}
            for model_name, model_info in self.loaded_models.items():
                loaded_models_info[model_name] = {
                    "load_time": model_info["load_time"].isoformat(),
                    "last_used": model_info["last_used"].isoformat(),
                    "memory_usage_mb": model_info["memory_usage_mb"],
                    "idle_minutes": (datetime.now() - model_info["last_used"]).total_seconds() / 60
                }

            resource_status = self.resource_monitor.get_resource_status()

            return {
                "loaded_models": loaded_models_info,
                "model_count": len(self.loaded_models),
                "max_concurrent_models": self.max_concurrent_models,
                "total_memory_usage_mb": sum(self.model_memory_usage.values()),
                "resource_status": resource_status,
                "usage_stats": dict(self.usage_tracker.usage_stats),
                "performance_metrics": self.usage_tracker.get_performance_metrics()
            }

    def optimize_model_allocation(self) -> Dict[str, Any]:
        """Optimize model allocation based on usage patterns."""
        performance_metrics = self.usage_tracker.get_performance_metrics(60)  # Last hour
        optimization_result = {
            "actions_taken": [],
            "recommendations": [],
            "resource_savings": 0.0
        }

        # Identify underperforming models
        for model_name, metrics in performance_metrics.items():
            if metrics["success_rate"] < 0.7:  # Less than 70% success rate
                optimization_result["recommendations"].append(
                    f"Model {model_name} has low success rate ({metrics['success_rate']:.1%}), consider replacement"
                )

            if metrics["recent_requests"] == 0:  # No recent usage
                if model_name in self.loaded_models:
                    self.unload_model(model_name)
                    optimization_result["actions_taken"].append(f"Unloaded unused model: {model_name}")
                    optimization_result["resource_savings"] += self.model_memory_usage.get(model_name, 0)

        return optimization_result


# Singleton instance
_model_resource_manager = None


def get_model_resource_manager() -> ModelResourceManager:
    """Get singleton instance of ModelResourceManager."""
    global _model_resource_manager
    if _model_resource_manager is None:
        _model_resource_manager = ModelResourceManager()
    return _model_resource_manager