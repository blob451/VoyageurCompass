"""
Asynchronous LLM service for high-performance concurrent explanation generation.
Provides async wrappers and batch processing capabilities for optimal throughput.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import threading

from django.core.cache import cache
from django.conf import settings

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.explanation_service import get_explanation_service
from Analytics.services.translation_service import get_translation_service
from Analytics.services.model_resource_manager import get_model_resource_manager

logger = logging.getLogger(__name__)


class AsyncLLMCoordinator:
    """Coordinates asynchronous LLM operations for optimal performance."""

    def __init__(self):
        self.llm_service = get_local_llm_service()
        self.explanation_service = get_explanation_service()
        self.translation_service = get_translation_service()
        self.resource_manager = get_model_resource_manager()

        # Thread pools for different operation types
        self.explanation_executor = ThreadPoolExecutor(
            max_workers=getattr(settings, "LLM_EXPLANATION_WORKERS", 4),
            thread_name_prefix="llm_explanation"
        )
        self.translation_executor = ThreadPoolExecutor(
            max_workers=getattr(settings, "LLM_TRANSLATION_WORKERS", 2),
            thread_name_prefix="llm_translation"
        )

        # Performance tracking
        self.performance_metrics = {
            "concurrent_requests": 0,
            "total_async_requests": 0,
            "average_batch_time": 0.0,
            "throughput_per_second": 0.0,
        }
        self._metrics_lock = threading.Lock()

    async def generate_explanation_async(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str = "standard"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate explanation asynchronously using thread pool.

        Args:
            analysis_data: Analysis data for explanation generation
            detail_level: Detail level for explanation

        Returns:
            Generated explanation or None if failed
        """
        loop = asyncio.get_event_loop()

        try:
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                self.explanation_executor,
                self._generate_explanation_sync,
                analysis_data,
                detail_level
            )
            return result

        except Exception as e:
            logger.error(f"Async explanation generation failed: {str(e)}")
            return None

    async def translate_explanation_async(
        self,
        english_text: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Translate explanation asynchronously using thread pool.

        Args:
            english_text: English text to translate
            target_language: Target language code
            context: Additional context for translation

        Returns:
            Translation result or None if failed
        """
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                self.translation_executor,
                self._translate_explanation_sync,
                english_text,
                target_language,
                context
            )
            return result

        except Exception as e:
            logger.error(f"Async translation failed: {str(e)}")
            return None

    async def generate_multilingual_explanation_batch(
        self,
        analysis_data_list: List[Dict[str, Any]],
        detail_level: str = "standard",
        target_languages: List[str] = None,
        max_concurrent: int = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple analysis results with optional translations.

        Args:
            analysis_data_list: List of analysis data
            detail_level: Detail level for explanations
            target_languages: Languages to translate to (fr, es)
            max_concurrent: Maximum concurrent operations

        Returns:
            List of explanation results with translations
        """
        if not analysis_data_list:
            return []

        if max_concurrent is None:
            max_concurrent = min(len(analysis_data_list), 6)

        target_languages = target_languages or []
        start_time = time.time()

        logger.info(
            f"Starting async batch generation: {len(analysis_data_list)} explanations, "
            f"{len(target_languages)} languages, max_concurrent={max_concurrent}"
        )

        # Update metrics
        with self._metrics_lock:
            self.performance_metrics["total_async_requests"] += len(analysis_data_list)
            self.performance_metrics["concurrent_requests"] = max_concurrent

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Generate all explanations concurrently
        explanation_tasks = [
            self._generate_with_semaphore(semaphore, analysis_data, detail_level)
            for analysis_data in analysis_data_list
        ]

        explanations = await asyncio.gather(*explanation_tasks, return_exceptions=True)

        # Process results and generate translations
        results = []
        translation_tasks = []

        for i, explanation in enumerate(explanations):
            if isinstance(explanation, Exception):
                logger.error(f"Explanation generation failed: {str(explanation)}")
                results.append({
                    "symbol": analysis_data_list[i].get("symbol", "unknown"),
                    "error": str(explanation),
                    "explanation": None,
                    "translations": {}
                })
                continue

            if not explanation:
                results.append({
                    "symbol": analysis_data_list[i].get("symbol", "unknown"),
                    "error": "No explanation generated",
                    "explanation": None,
                    "translations": {}
                })
                continue

            result_entry = {
                "symbol": analysis_data_list[i].get("symbol", "unknown"),
                "explanation": explanation,
                "translations": {},
                "error": None
            }
            results.append(result_entry)

            # Create translation tasks for each target language
            if target_languages and explanation.get("content"):
                for lang in target_languages:
                    if lang != "en":  # Skip English (original)
                        task_info = (i, lang, explanation["content"])
                        translation_task = self._translate_with_semaphore(
                            semaphore, task_info
                        )
                        translation_tasks.append(translation_task)

        # Execute translations concurrently
        if translation_tasks:
            logger.info(f"Starting {len(translation_tasks)} translation tasks")
            translations = await asyncio.gather(*translation_tasks, return_exceptions=True)

            # Process translation results
            for translation_result in translations:
                if isinstance(translation_result, Exception):
                    logger.error(f"Translation failed: {str(translation_result)}")
                    continue

                if translation_result:
                    result_idx, lang, translation_data = translation_result
                    if 0 <= result_idx < len(results):
                        results[result_idx]["translations"][lang] = translation_data

        # Calculate performance metrics
        total_time = time.time() - start_time
        throughput = len(analysis_data_list) / total_time if total_time > 0 else 0

        with self._metrics_lock:
            self.performance_metrics["average_batch_time"] = (
                (self.performance_metrics["average_batch_time"] + total_time) / 2
            )
            self.performance_metrics["throughput_per_second"] = throughput

        logger.info(
            f"Async batch completed: {len(results)} results in {total_time:.2f}s "
            f"(throughput: {throughput:.2f} req/s)"
        )

        return results

    async def _generate_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        analysis_data: Dict[str, Any],
        detail_level: str
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation with semaphore for concurrency control."""
        async with semaphore:
            return await self.generate_explanation_async(analysis_data, detail_level)

    async def _translate_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        task_info: Tuple[int, str, str]
    ) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        """Translate text with semaphore for concurrency control."""
        async with semaphore:
            result_idx, lang, text = task_info
            translation = await self.translate_explanation_async(text, lang)
            if translation:
                return (result_idx, lang, translation)
            return None

    def _generate_explanation_sync(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str
    ) -> Optional[Dict[str, Any]]:
        """Synchronous explanation generation for thread pool execution."""
        try:
            # Check resource availability before generation
            symbol = analysis_data.get("symbol", "unknown")

            if self.resource_manager:
                should_load, reason = self.resource_manager.should_load_model(
                    self.llm_service.standard_model
                )
                if not should_load:
                    logger.warning(f"Resource constraints prevent model loading for {symbol}: {reason}")
                    # Fall back to template explanation
                    return self.explanation_service._generate_template_explanation(
                        analysis_data, detail_level
                    )

            # Generate explanation using existing service
            result = self.explanation_service.explain_prediction_single(
                analysis_data, detail_level
            )

            # Register resource usage
            if self.resource_manager and result:
                generation_time = result.get("generation_time", 0.0)
                success = bool(result.get("content"))
                self.resource_manager.register_model_usage(
                    self.llm_service.standard_model, generation_time, success
                )

            return result

        except Exception as e:
            logger.error(f"Sync explanation generation failed: {str(e)}")
            return None

    def _translate_explanation_sync(
        self,
        english_text: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Synchronous translation for thread pool execution."""
        try:
            return self.translation_service.translate_explanation(
                english_text, target_language, context
            )
        except Exception as e:
            logger.error(f"Sync translation failed: {str(e)}")
            return None

    async def warm_models_async(self, models: List[str] = None) -> Dict[str, bool]:
        """
        Warm up models asynchronously for better performance.

        Args:
            models: List of model names to warm up

        Returns:
            Dictionary with model warming results
        """
        if not models:
            models = [
                self.llm_service.summary_model,
                self.llm_service.standard_model,
                self.llm_service.detailed_model,
            ]

        logger.info(f"Starting async model warming for {len(models)} models")

        # Create dummy analysis data for warming
        dummy_data = {
            "symbol": "WARM",
            "score_0_10": 5.0,
            "recommendation": "HOLD",
            "technical_score": 5.0,
            "indicators": {"rsi": {"value": 50.0, "signal": "neutral"}},
        }

        warming_tasks = []
        for model in models:
            # Temporarily set model for warming
            original_model = self.llm_service.current_model
            self.llm_service.current_model = model

            task = self.generate_explanation_async(dummy_data, "summary")
            warming_tasks.append((model, task))

            # Restore original model
            self.llm_service.current_model = original_model

        # Execute warming tasks
        results = {}
        for model, task in warming_tasks:
            try:
                result = await task
                results[model] = bool(result)
                logger.info(f"Model {model} warming: {'SUCCESS' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Model {model} warming failed: {str(e)}")
                results[model] = False

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._metrics_lock:
            return {
                **self.performance_metrics,
                "explanation_pool_active": self.explanation_executor._threads,
                "translation_pool_active": self.translation_executor._threads,
                "resource_manager_status": (
                    self.resource_manager.get_resource_status()
                    if self.resource_manager else {}
                ),
            }

    def shutdown(self):
        """Gracefully shutdown thread pools."""
        logger.info("Shutting down async LLM coordinator")
        self.explanation_executor.shutdown(wait=True)
        self.translation_executor.shutdown(wait=True)


# Singleton instance
_async_llm_coordinator = None


def get_async_llm_coordinator() -> AsyncLLMCoordinator:
    """Get singleton instance of AsyncLLMCoordinator."""
    global _async_llm_coordinator
    if _async_llm_coordinator is None:
        _async_llm_coordinator = AsyncLLMCoordinator()
    return _async_llm_coordinator


# Convenience functions for easy async usage
async def generate_explanation_async(
    analysis_data: Dict[str, Any],
    detail_level: str = "standard"
) -> Optional[Dict[str, Any]]:
    """Generate explanation asynchronously."""
    coordinator = get_async_llm_coordinator()
    return await coordinator.generate_explanation_async(analysis_data, detail_level)


async def translate_explanation_async(
    english_text: str,
    target_language: str,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Translate explanation asynchronously."""
    coordinator = get_async_llm_coordinator()
    return await coordinator.translate_explanation_async(english_text, target_language, context)


async def generate_multilingual_batch_async(
    analysis_data_list: List[Dict[str, Any]],
    detail_level: str = "standard",
    target_languages: List[str] = None,
    max_concurrent: int = None
) -> List[Dict[str, Any]]:
    """Generate multilingual explanations for batch data."""
    coordinator = get_async_llm_coordinator()
    return await coordinator.generate_multilingual_explanation_batch(
        analysis_data_list, detail_level, target_languages, max_concurrent
    )