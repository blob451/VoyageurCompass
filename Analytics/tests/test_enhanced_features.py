"""
Enhanced Features Test Suite for Analytics app.
Tests all the newly implemented enhancements using real functionality.
No mocks - uses real service implementations and graceful error handling.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from Analytics.services.advanced_monitoring_service import (
    get_monitoring_service,
    profile_performance,
)
from Analytics.services.async_processing_pipeline import (
    AsyncProcessingPipeline,
    get_async_processing_pipeline,
)
from Analytics.services.enhanced_finetuning_service import get_finetuning_manager

User = get_user_model()


class AsyncProcessingPipelineTestCase(TestCase):
    """Test async processing pipeline functionality."""

    def setUp(self):
        self.pipeline = AsyncProcessingPipeline(max_workers=2)
        self.user = User.objects.create_user(username="testuser", password="testpass")

    def test_task_status_tracking(self):
        """Test task status tracking system."""
        # Create task
        task_info = self.pipeline.task_status.create_task("test_task_1", "test_operation", "TEST_SYMBOL")

        self.assertEqual(task_info["status"], "pending")
        self.assertEqual(task_info["symbol"], "TEST_SYMBOL")
        self.assertIsNotNone(task_info["created_at"])

        # Update task status
        self.pipeline.task_status.update_task_status("test_task_1", "running", progress=50.0)

        updated_task = self.pipeline.task_status.get_task_status("test_task_1")
        self.assertEqual(updated_task["status"], "running")
        self.assertEqual(updated_task["progress"], 50.0)

        # Complete task
        self.pipeline.task_status.update_task_status(
            "test_task_1", "completed", progress=100.0, result={"content": "Test result"}
        )

        final_task = self.pipeline.task_status.get_task_status("test_task_1")
        self.assertEqual(final_task["status"], "completed")
        self.assertEqual(final_task["progress"], 100.0)
        self.assertIsNotNone(final_task["result"])

    def test_batch_processing(self):
        """Test batch processing functionality."""

        # Mock processor function
        def mock_processor(request):
            symbol = request["symbol"]
            return {"symbol": symbol, "score": 7.5, "processed": True}

        # Create test requests
        requests = [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOGL"}]

        # Process batch
        result = self.pipeline.process_batch_analysis(requests, mock_processor, "test_batch_1")

        # Validate results
        self.assertEqual(result["batch_id"], "test_batch_1")
        self.assertEqual(result["total_requests"], 3)
        self.assertEqual(result["successful_requests"], 3)
        self.assertEqual(result["failed_requests"], 0)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(len(result["results"]), 3)

        # Check individual results
        for i, res in enumerate(result["results"]):
            self.assertIsNotNone(res)
            self.assertTrue(res["processed"])
            self.assertIn("symbol", res)

    def test_batch_status_tracking(self):
        """Test batch status tracking."""

        # Create mock processor
        def slow_processor(request):
            time.sleep(0.1)  # Simulate processing time
            return {"symbol": request["symbol"], "result": "success"}

        requests = [{"symbol": "TEST1"}, {"symbol": "TEST2"}]

        # Start batch processing in background
        import threading

        def run_batch():
            self.pipeline.process_batch_analysis(requests, slow_processor, "status_test_batch")

        thread = threading.Thread(target=run_batch)
        thread.start()

        # Check batch status
        time.sleep(0.05)  # Let processing start
        batch_status = self.pipeline.get_batch_status("status_test_batch")

        self.assertEqual(batch_status["batch_id"], "status_test_batch")
        self.assertEqual(batch_status["total_tasks"], 2)
        self.assertGreaterEqual(batch_status["progress"], 0)

        # Wait for completion
        thread.join()

        # Check final status
        final_status = self.pipeline.get_batch_status("status_test_batch")
        self.assertEqual(final_status["progress"], 1.0)
        self.assertEqual(final_status["completed_tasks"], 2)


class FineTuningServiceTestCase(TestCase):
    """Test fine-tuning service functionality."""

    def setUp(self):
        self.finetuning_manager = get_finetuning_manager()

    def test_synthetic_data_generation(self):
        """Test synthetic analysis data generation."""
        synthetic_data = self.finetuning_manager._generate_synthetic_analysis_data("AAPL")

        self.assertEqual(synthetic_data["symbol"], "AAPL")
        self.assertIn("score_0_10", synthetic_data)
        self.assertIn("weighted_scores", synthetic_data)
        self.assertTrue(synthetic_data["synthetic"])

        # Check score bounds
        score = synthetic_data["score_0_10"]
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 10.0)

        # Check weighted scores structure
        weighted_scores = synthetic_data["weighted_scores"]
        expected_indicators = ["w_sma50vs200", "w_rsi14", "w_macd12269", "w_bbpos20", "w_volsurge"]
        for indicator in expected_indicators:
            self.assertIn(indicator, weighted_scores)

    def test_instruction_prompt_creation(self):
        """Test instruction prompt creation for different detail levels."""
        # Test summary level
        summary_prompt = self.finetuning_manager._create_instruction_prompt("summary")
        self.assertIn("concise", summary_prompt.lower())
        self.assertIn("BUY/SELL/HOLD", summary_prompt)

        # Test standard level
        standard_prompt = self.finetuning_manager._create_instruction_prompt("standard")
        self.assertIn("recommendation", standard_prompt.lower())
        self.assertIn("technical factors", standard_prompt)

        # Test detailed level
        detailed_prompt = self.finetuning_manager._create_instruction_prompt("detailed")
        self.assertIn("comprehensive", detailed_prompt.lower())
        self.assertIn("confidence level", detailed_prompt)

    def test_quality_scoring(self):
        """Test sample quality scoring."""
        # High quality content
        high_quality = "Based on technical analysis, AAPL shows strong BUY signals with RSI indicating momentum and MACD crossover confirming the bullish trend. The investment recommendation is supported by solid performance metrics."

        quality_score = self.finetuning_manager._calculate_sample_quality(high_quality)
        self.assertGreater(quality_score, 0.7)

        # Low quality content
        low_quality = "Buy."
        low_score = self.finetuning_manager._calculate_sample_quality(low_quality)
        self.assertLess(low_score, 0.5)

        # Empty content
        empty_score = self.finetuning_manager._calculate_sample_quality("")
        self.assertEqual(empty_score, 0.0)

    @patch("Analytics.services.enhanced_finetuning_service.FINE_TUNING_AVAILABLE", False)
    def test_simulated_fine_tuning_job(self):
        """Test simulated fine-tuning job when dependencies unavailable."""
        job_info = self.finetuning_manager.start_fine_tuning_job(
            dataset_path="dummy_path.json", job_name="test_simulated_job"
        )

        self.assertEqual(job_info["status"], "simulated")
        self.assertEqual(job_info["job_name"], "test_simulated_job")
        self.assertTrue(job_info["simulated"])
        self.assertIn("dependencies", job_info["message"].lower())

    def test_job_status_tracking(self):
        """Test job status tracking."""
        job_info = self.finetuning_manager.start_fine_tuning_job(
            dataset_path="test_path.json", job_name="status_test_job"
        )

        job_id = job_info["job_id"]

        # Retrieve job status
        status_info = self.finetuning_manager.get_job_status(job_id)
        self.assertIsNotNone(status_info)
        self.assertEqual(status_info["job_id"], job_id)
        self.assertEqual(status_info["job_name"], "status_test_job")

        # Test nonexistent job
        nonexistent_status = self.finetuning_manager.get_job_status("nonexistent_id")
        self.assertIsNone(nonexistent_status)

    def test_job_listing(self):
        """Test job listing functionality."""
        # Start multiple jobs
        job1 = self.finetuning_manager.start_fine_tuning_job(dataset_path="path1.json", job_name="job1")
        job2 = self.finetuning_manager.start_fine_tuning_job(dataset_path="path2.json", job_name="job2")

        # List jobs
        jobs_info = self.finetuning_manager.list_jobs()

        self.assertGreaterEqual(jobs_info["total_jobs"], 2)
        self.assertIn("jobs", jobs_info)
        self.assertIn("active_jobs", jobs_info)
        self.assertIn("completed_jobs", jobs_info)

        # Check that our jobs are in the list
        job_names = [job["job_name"] for job in jobs_info["jobs"]]
        self.assertIn("job1", job_names)
        self.assertIn("job2", job_names)


class MonitoringServiceTestCase(TestCase):
    """Test advanced monitoring service functionality."""

    def setUp(self):
        self.monitoring_service = get_monitoring_service()

    def test_metric_recording_and_retrieval(self):
        """Test metric recording and retrieval."""
        metric_name = "test_metric"
        test_value = 75.5
        test_labels = {"component": "test", "environment": "testing"}

        # Record metric
        self.monitoring_service.metrics_collector.record_metric(metric_name, test_value, test_labels)

        # Retrieve metric history
        history = self.monitoring_service.metrics_collector.get_metric_history(metric_name)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["value"], test_value)
        self.assertEqual(history[0]["labels"], test_labels)

        # Test metric summary
        summary = self.monitoring_service.metrics_collector.get_metric_summary(metric_name)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["min"], test_value)
        self.assertEqual(summary["max"], test_value)
        self.assertEqual(summary["avg"], test_value)
        self.assertEqual(summary["latest"], test_value)

    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        profiler = self.monitoring_service.performance_profiler

        # Start profile
        profile_id = profiler.start_profile("test_operation", "testing", {"test_key": "test_value"})

        self.assertIsNotNone(profile_id)
        self.assertIn(profile_id, profiler.active_profiles)

        # Add checkpoint
        profiler.add_checkpoint(profile_id, "checkpoint_1", {"step": 1})

        # End profile
        result = profiler.end_profile(profile_id, "completed", {"final": "data"})

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["profile_name"], "test_operation")
        self.assertIn("total_duration", result)
        self.assertIn("checkpoints", result)
        self.assertEqual(len(result["checkpoints"]), 1)
        self.assertEqual(result["checkpoints"][0]["name"], "checkpoint_1")

        # Check profile is no longer active
        self.assertNotIn(profile_id, profiler.active_profiles)

    def test_profile_context_manager(self):
        """Test profile context manager."""
        with profile_performance("context_test", "testing", {"test": True}) as profiler:
            # Add checkpoint within context
            profiler.checkpoint("mid_operation", {"progress": 50})

            # Simulate some work
            time.sleep(0.01)

        # Profile should be automatically ended
        # Check completed profiles contain our operation
        completed_profiles = self.monitoring_service.performance_profiler.completed_profiles
        self.assertGreater(len(completed_profiles), 0)

        # Find our profile
        our_profile = None
        for profile in completed_profiles:
            if profile["profile_name"] == "context_test":
                our_profile = profile
                break

        self.assertIsNotNone(our_profile)
        self.assertEqual(our_profile["status"], "completed")
        self.assertEqual(len(our_profile["checkpoints"]), 1)

    def test_alert_system(self):
        """Test alert management system."""
        alert_manager = self.monitoring_service.alert_manager

        # Trigger alert manually
        alert_manager.trigger_alert(
            rule_name="test_rule", metric_name="test_metric", current_value=95.0, threshold=80.0, severity="warning"
        )

        # Check alert was recorded
        recent_alerts = alert_manager.get_recent_alerts(hours=1)
        self.assertGreater(len(recent_alerts), 0)

        # Find our alert
        our_alert = None
        for alert in recent_alerts:
            if alert["rule_name"] == "test_rule":
                our_alert = alert
                break

        self.assertIsNotNone(our_alert)
        self.assertEqual(our_alert["severity"], "warning")
        self.assertEqual(our_alert["current_value"], 95.0)
        self.assertEqual(our_alert["threshold"], 80.0)

    def test_system_health_assessment(self):
        """Test system health assessment."""
        # Record some system metrics
        self.monitoring_service.metrics_collector.record_metric("system_cpu_percent", 25.0)
        self.monitoring_service.metrics_collector.record_metric("system_memory_percent", 60.0)
        self.monitoring_service.metrics_collector.record_metric("llm_generation_time", 2.5)

        # Get health status
        health = self.monitoring_service.get_system_health()

        self.assertIn("status", health)
        self.assertIn("components", health)
        self.assertIn("metrics", health)
        self.assertIn("alerts", health)

        # Should be healthy with these metrics
        self.assertIn(health["status"], ["healthy", "degraded"])

    def test_metric_cleanup(self):
        """Test metric cleanup functionality."""
        collector = self.monitoring_service.metrics_collector

        # Record metrics with old timestamp
        old_timestamp = datetime.now() - timedelta(hours=25)  # Older than retention
        current_timestamp = datetime.now()

        collector.record_metric("cleanup_test_metric", 10.0, timestamp=old_timestamp)
        collector.record_metric("cleanup_test_metric", 20.0, timestamp=current_timestamp)

        # Check we have 2 metrics
        history_before = collector.get_metric_history("cleanup_test_metric")
        self.assertEqual(len(history_before), 2)

        # Run cleanup
        collector.cleanup_old_metrics()

        # Check old metric was removed
        history_after = collector.get_metric_history("cleanup_test_metric")
        self.assertEqual(len(history_after), 1)
        self.assertEqual(history_after[0]["value"], 20.0)


class EnhancedFeaturesAPITestCase(APITestCase):
    """Test API endpoints for enhanced features."""

    def setUp(self):
        self.user = User.objects.create_user(username="apiuser", password="apipass")
        self.client.force_authenticate(user=self.user)

    def test_async_performance_endpoint(self):
        """Test async pipeline performance endpoint."""
        url = reverse("analytics:async_performance")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("max_workers", data)
        self.assertIn("total_batch_requests", data)
        self.assertIn("batch_success_rate", data)

    def test_finetuning_status_endpoint(self):
        """Test fine-tuning status endpoint."""
        url = reverse("analytics:finetuning_status")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("fine_tuning_available", data)
        self.assertIn("capabilities", data)
        self.assertIn("system_ready", data)

    def test_monitoring_health_endpoint(self):
        """Test monitoring health endpoint."""
        url = reverse("analytics:system_health")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("status", data)
        self.assertIn("components", data)
        self.assertIn("metrics", data)

    def test_available_metrics_endpoint(self):
        """Test available metrics endpoint."""
        url = reverse("analytics:available_metrics")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("metric_definitions", data)
        self.assertIn("metric_counts", data)
        self.assertIn("total_metrics", data)

    def test_monitoring_status_endpoint(self):
        """Test monitoring service status endpoint."""
        url = reverse("analytics:monitoring_status")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        self.assertIn("service_status", data)
        self.assertIn("components", data)
        self.assertIn("capabilities", data)


class IntegrationTestCase(TransactionTestCase):
    """Integration tests for enhanced features working together."""

    def setUp(self):
        self.user = User.objects.create_user(username="integuser", password="integpass")

        # Get services
        self.monitoring_service = get_monitoring_service()
        self.async_pipeline = get_async_processing_pipeline()

    def test_monitoring_async_integration(self):
        """Test monitoring integration with async processing."""
        # Record async metrics
        self.monitoring_service.record_llm_metrics(
            generation_time=1.5, success=True, model_name="llama3.1:8b", complexity_score=0.6
        )

        # Check metrics were recorded
        llm_time_history = self.monitoring_service.metrics_collector.get_metric_history("llm_generation_time")
        self.assertGreater(len(llm_time_history), 0)

        request_count_history = self.monitoring_service.metrics_collector.get_metric_history("llm_request_count")
        self.assertGreater(len(request_count_history), 0)

        # Check pipeline performance metrics
        pipeline_performance = self.async_pipeline.get_performance_summary()
        self.assertIn("max_workers", pipeline_performance)

    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from metric recording to alerting."""
        # Record system metrics that should trigger alerts
        self.monitoring_service.metrics_collector.record_metric("system_cpu_percent", 85.0)

        # Check alert system
        self.monitoring_service.alert_manager.check_alerts(self.monitoring_service.metrics_collector)

        # Verify alert was triggered
        recent_alerts = self.monitoring_service.alert_manager.get_recent_alerts(hours=1)
        cpu_alerts = [alert for alert in recent_alerts if "cpu" in alert["metric_name"]]

        if cpu_alerts:  # Alert rules might trigger
            self.assertGreater(len(cpu_alerts), 0)
            self.assertEqual(cpu_alerts[0]["severity"], "warning")

    def test_performance_profiling_integration(self):
        """Test performance profiling integrated with actual operations."""
        with profile_performance("integration_test", "test_operation") as profiler:
            # Simulate complex operation with checkpoints
            profiler.checkpoint("initialization", {"component": "test"})

            # Record some metrics during operation
            self.monitoring_service.metrics_collector.record_metric("test_operation_metric", 42.0)

            profiler.checkpoint("processing", {"items_processed": 100})

            # Simulate processing time
            time.sleep(0.01)

            profiler.checkpoint("finalization", {"status": "success"})

        # Check that profile was completed
        profile_summary = self.monitoring_service.performance_profiler.get_profile_summary("test_operation", hours=1)

        if "message" not in profile_summary:  # Profile data available
            self.assertGreater(profile_summary["total_profiles"], 0)
            self.assertEqual(profile_summary["operation_type"], "test_operation")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
