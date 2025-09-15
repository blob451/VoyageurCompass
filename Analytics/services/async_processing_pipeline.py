"""
Asynchronous Processing Pipeline implementing concurrent analysis capabilities.
Enables parallel processing of multiple analysis requests for optimised performance.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class AsyncTaskStatus:
    """Task status management system for asynchronous processing operations."""        

    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def create_task(self, task_id: str, task_type: str, symbol: str) -> Dict[str, Any]:
        """Generate new task status entry with initialised tracking parameters."""        
        with self.lock:
            task_info = {
                "task_id": task_id,
                "task_type": task_type,
                "symbol": symbol,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "progress": 0.0,
            }
            self.tasks[task_id] = task_info
            return task_info

    def update_task_status(
        self, task_id: str, status: str, progress: float = None, result: Any = None, error: str = None
    ):
        """Update task status."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task["status"] = status

                if progress is not None:
                    task["progress"] = progress
                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error

                if status == "running" and task["started_at"] is None:
                    task["started_at"] = datetime.now().isoformat()
                elif status in ["completed", "failed"]:
                    task["completed_at"] = datetime.now().isoformat()

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        with self.lock:
            return self.tasks.get(task_id, None)

    def get_all_tasks(self) -> Dict[str, Any]:
        """Get all tasks."""
        with self.lock:
            return self.tasks.copy()


class AsyncProcessingPipeline:
    """Asynchronous processing pipeline implementing concurrent analysis architecture."""        

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_status = AsyncTaskStatus()

        # Performance tracking
        self.total_batch_requests = 0
        self.successful_batch_requests = 0
        self.average_batch_time = 0.0

        logger.info(f"AsyncProcessingPipeline initialized with {max_workers} workers")

    def process_batch_analysis(
        self, analysis_requests: List[Dict[str, Any]], processor_func: Callable, batch_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute concurrent processing of multiple analysis requests through batch operations.

        Args:
            analysis_requests: Analysis request dictionary collection
            processor_func: Processing function for individual requests
            batch_id: Batch identification parameter

        Returns:
            Comprehensive batch processing results with metadata
        """
        if not batch_id:
            batch_id = f"batch_{int(time.time() * 1000)}"

        self.total_batch_requests += 1
        start_time = time.time()

        logger.info(f"[ASYNC PIPELINE] Starting batch processing: {batch_id} ({len(analysis_requests)} items)")

        # Create task entries
        task_ids = []
        for i, request in enumerate(analysis_requests):
            task_id = f"{batch_id}_task_{i}"
            symbol = request.get("symbol", f"UNKNOWN_{i}")
            self.task_status.create_task(task_id, "analysis", symbol)
            task_ids.append(task_id)

        # Submit tasks to thread pool
        future_to_task = {}
        for i, (request, task_id) in enumerate(zip(analysis_requests, task_ids)):
            future = self.executor.submit(
                self._process_single_request, processor_func, request, task_id, i, len(analysis_requests)
            )
            future_to_task[future] = task_id

        # Collect results as they complete
        results = []
        failed_tasks = []

        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                self.task_status.update_task_status(task_id, "completed", 100.0, result)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[ASYNC PIPELINE] Task {task_id} failed: {error_msg}")
                failed_tasks.append(task_id)
                self.task_status.update_task_status(task_id, "failed", error=error_msg)
                results.append(None)

        # Calculate batch metrics
        batch_time = time.time() - start_time
        success_count = len([r for r in results if r is not None])
        failure_count = len(failed_tasks)

        # Update performance metrics
        if success_count > 0:
            self.successful_batch_requests += 1

        if self.successful_batch_requests == 1:
            self.average_batch_time = batch_time
        else:
            self.average_batch_time = (
                self.average_batch_time * (self.successful_batch_requests - 1) + batch_time
            ) / self.successful_batch_requests

        # Convert results to symbol-keyed dictionary for BatchAnalysisService compatibility
        results_dict = {}
        for i, result in enumerate(results):
            if result:
                symbol = result.get('symbol') or analysis_requests[i].get('symbol', f'UNKNOWN_{i}')
                results_dict[symbol] = result

        batch_result = {
            "batch_id": batch_id,
            "results": results_dict,  # Now returns dict keyed by symbol
            "task_ids": task_ids,
            "processing_time": batch_time,
            "total_requests": len(analysis_requests),
            "successful_requests": success_count,
            "failed_requests": failure_count,
            "success_rate": success_count / len(analysis_requests) if analysis_requests else 0,
            "failed_task_ids": failed_tasks,
            "errors": [{"symbol": analysis_requests[i].get('symbol', f'UNKNOWN_{i}'),
                       "error": f"Task failed: {task_id}"} for i, task_id in enumerate(failed_tasks)],
            "average_time_per_request": batch_time / len(analysis_requests) if analysis_requests else 0,
            "completed_at": datetime.now().isoformat(),
        }

        logger.info(
            f"[ASYNC PIPELINE] Batch {batch_id} completed: {success_count}/{len(analysis_requests)} successful in {batch_time:.2f}s"
        )

        return batch_result

    def _process_single_request(
        self, processor_func: Callable, request: Dict[str, Any], task_id: str, task_index: int, total_tasks: int
    ) -> Optional[Dict[str, Any]]:
        """
        Execute individual analysis request with comprehensive progress monitoring.

        Args:
            processor_func: Request processing function implementation
            request: Analysis request data structure
            task_id: Unique task identification parameter
            task_index: Task position within batch sequence
            total_tasks: Total batch size parameter

        Returns:
            Processing result structure or None upon failure
        """
        symbol = request.get("symbol", f"UNKNOWN_{task_index}")

        try:
            self.task_status.update_task_status(task_id, "running", 10.0)
            logger.debug(f"[ASYNC WORKER] Processing {symbol} (task {task_index + 1}/{total_tasks})")

            # Call the processor function
            start_time = time.time()
            result = processor_func(request)
            processing_time = time.time() - start_time

            if result:
                # Add async processing metadata
                if isinstance(result, dict):
                    result["async_processing"] = {
                        "task_id": task_id,
                        "batch_position": task_index + 1,
                        "batch_size": total_tasks,
                        "processing_time": processing_time,
                        "processed_at": datetime.now().isoformat(),
                    }

                self.task_status.update_task_status(task_id, "completed", 100.0, result)
                logger.debug(f"[ASYNC WORKER] Completed {symbol} in {processing_time:.2f}s")
                return result
            else:
                logger.warning(f"[ASYNC WORKER] No result for {symbol}")
                return None

        except Exception as e:
            logger.error(f"[ASYNC WORKER] Error processing {symbol}: {str(e)}")
            self.task_status.update_task_status(task_id, "failed", error=str(e))
            return None

    async def process_batch_analysis_async(
        self, analysis_requests: List[Dict[str, Any]], processor_func: Callable, batch_id: str = None
    ) -> Dict[str, Any]:
        """
        Async version of batch processing using asyncio.

        Args:
            analysis_requests: List of analysis requests
            processor_func: Processing function
            batch_id: Optional batch identifier

        Returns:
            Batch processing results
        """
        if not batch_id:
            batch_id = f"async_batch_{int(time.time() * 1000)}"

        logger.info(f"[ASYNC PIPELINE] Starting async batch: {batch_id} ({len(analysis_requests)} items)")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(request, task_id, index):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_single_request,
                    processor_func,
                    request,
                    task_id,
                    index,
                    len(analysis_requests),
                )

        # Create tasks
        tasks = []
        task_ids = []
        for i, request in enumerate(analysis_requests):
            task_id = f"{batch_id}_task_{i}"
            symbol = request.get("symbol", f"UNKNOWN_{i}")
            self.task_status.create_task(task_id, "async_analysis", symbol)
            task_ids.append(task_id)

            task = asyncio.create_task(process_with_semaphore(request, task_id, i))
            tasks.append(task)

        # Wait for all tasks to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_time = time.time() - start_time

        # Process results and exceptions
        processed_results = []
        failed_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[ASYNC PIPELINE] Task exception: {str(result)}")
                processed_results.append(None)
                failed_count += 1
            else:
                processed_results.append(result)

        success_count = len([r for r in processed_results if r is not None])

        batch_result = {
            "batch_id": batch_id,
            "results": processed_results,
            "task_ids": task_ids,
            "processing_time": batch_time,
            "total_requests": len(analysis_requests),
            "successful_requests": success_count,
            "failed_requests": failed_count,
            "success_rate": success_count / len(analysis_requests) if analysis_requests else 0,
            "average_time_per_request": batch_time / len(analysis_requests) if analysis_requests else 0,
            "completed_at": datetime.now().isoformat(),
            "processing_type": "asyncio",
        }

        logger.info(
            f"[ASYNC PIPELINE] Async batch {batch_id} completed: {success_count}/{len(analysis_requests)} successful in {batch_time:.2f}s"
        )

        return batch_result

    def process_sentiment_explanation_batch(
        self, analysis_data_list: List[Dict[str, Any]], detail_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Process batch of sentiment-enhanced explanations concurrently.

        Args:
            analysis_data_list: List of analysis data dictionaries
            detail_level: Detail level for explanations

        Returns:
            Batch processing results
        """
        from Analytics.services.hybrid_analysis_coordinator import (
            get_hybrid_analysis_coordinator,
        )

        hybrid_coordinator = get_hybrid_analysis_coordinator()

        def process_explanation(analysis_data):
            """Wrapper function for explanation processing."""
            return hybrid_coordinator.generate_enhanced_explanation(
                analysis_data=analysis_data, detail_level=detail_level
            )

        # Create request format expected by batch processor
        requests = [{"analysis_data": data, "detail_level": detail_level} for data in analysis_data_list]

        return self.process_batch_analysis(
            requests,
            lambda req: process_explanation(req["analysis_data"]),
            f"explanation_batch_{int(time.time() * 1000)}",
        )

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        return self.task_status.get_task_status(task_id)

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of all tasks in a batch."""
        all_tasks = self.task_status.get_all_tasks()
        batch_tasks = {tid: task for tid, task in all_tasks.items() if batch_id in tid}

        if not batch_tasks:
            return {"error": "Batch not found"}

        # Calculate batch summary
        total_tasks = len(batch_tasks)
        completed_tasks = sum(1 for task in batch_tasks.values() if task["status"] == "completed")
        failed_tasks = sum(1 for task in batch_tasks.values() if task["status"] == "failed")
        running_tasks = sum(1 for task in batch_tasks.values() if task["status"] == "running")

        return {
            "batch_id": batch_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "progress": (completed_tasks + failed_tasks) / total_tasks if total_tasks > 0 else 0,
            "tasks": batch_tasks,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary statistics for pipeline operations."""        
        return {
            "max_workers": self.max_workers,
            "total_batch_requests": self.total_batch_requests,
            "successful_batch_requests": self.successful_batch_requests,
            "batch_success_rate": self.successful_batch_requests / max(1, self.total_batch_requests),
            "average_batch_time": self.average_batch_time,
            "executor_active": not self.executor._shutdown,
            "current_tasks": len(self.task_status.get_all_tasks()),
        }

    def shutdown(self):
        """Shutdown the async processing pipeline."""
        logger.info("[ASYNC PIPELINE] Shutting down...")
        self.executor.shutdown(wait=True)
        logger.info("[ASYNC PIPELINE] Shutdown complete")


# Singleton instance
_async_pipeline = None


def get_async_processing_pipeline(max_workers: int = 4) -> AsyncProcessingPipeline:
    """Retrieve singleton AsyncProcessingPipeline instance with worker configuration."""        
    global _async_pipeline
    if _async_pipeline is None:
        _async_pipeline = AsyncProcessingPipeline(max_workers=max_workers)
    return _async_pipeline
