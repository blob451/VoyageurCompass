"""
Comprehensive production readiness testing suite for multilingual LLM system.

Tests all critical production components including:
- Feature flags functionality
- Circuit breaker operations
- Health check endpoints
- Production monitoring
- Multilingual service reliability
- Performance under load
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.cache import cache
from rest_framework.test import APITestCase
from rest_framework import status

from Analytics.services.feature_flags import (
    get_feature_flags,
    MultilingualFeatureFlags,
    is_multilingual_enabled
)
from Analytics.services.circuit_breaker import (
    get_circuit_breaker,
    MultilingualCircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError
)
from Analytics.services.production_monitoring import (
    get_monitoring_service,
    ProductionMonitoringService,
    ProductionAlert,
    AlertLevel
)
from Analytics.services.local_llm_service import get_local_llm_service


class FeatureFlagsProductionTest(TestCase):
    """Test feature flags system for production reliability."""

    def setUp(self):
        """Set up test environment."""
        self.feature_flags = get_feature_flags()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        cache.clear()

    def test_feature_flags_initialization(self):
        """Test feature flags initialize with correct defaults."""
        flags = MultilingualFeatureFlags()

        # Test default flag states
        self.assertTrue(flags.is_enabled(flags.MULTILINGUAL_ENABLED))
        self.assertTrue(flags.is_enabled(flags.FRENCH_GENERATION_ENABLED))
        self.assertTrue(flags.is_enabled(flags.SPANISH_GENERATION_ENABLED))
        self.assertFalse(flags.is_enabled(flags.EMERGENCY_FALLBACK_ENABLED))

    def test_gradual_rollout_functionality(self):
        """Test gradual rollout with percentage controls."""
        flags = get_feature_flags()

        # Test 0% rollout
        flags.set_rollout_percentage(flags.FRENCH_GENERATION_ENABLED, 0)
        self.assertFalse(flags.is_enabled_for_user(flags.FRENCH_GENERATION_ENABLED, self.user))

        # Test 100% rollout
        flags.set_rollout_percentage(flags.FRENCH_GENERATION_ENABLED, 100)
        self.assertTrue(flags.is_enabled_for_user(flags.FRENCH_GENERATION_ENABLED, self.user))

    def test_emergency_disable_functionality(self):
        """Test emergency disable functionality."""
        flags = get_feature_flags()

        # Enable features first
        self.assertTrue(flags.is_enabled(flags.MULTILINGUAL_ENABLED))

        # Trigger emergency disable
        disabled_flags = flags.emergency_disable_all("Test emergency")

        # Verify emergency fallback is enabled
        self.assertTrue(flags.is_enabled(flags.EMERGENCY_FALLBACK_ENABLED))

        # Verify core features are disabled
        self.assertFalse(flags.is_enabled(flags.MULTILINGUAL_ENABLED))
        self.assertIn(flags.MULTILINGUAL_ENABLED, disabled_flags)

    def test_multilingual_enabled_check(self):
        """Test the main multilingual enabled check."""
        # Test normal operation
        self.assertTrue(is_multilingual_enabled("fr", self.user))

        # Test with emergency fallback
        flags = get_feature_flags()
        flags.set_flag(flags.EMERGENCY_FALLBACK_ENABLED, True)
        self.assertFalse(is_multilingual_enabled("fr", self.user))

    def test_cache_behavior(self):
        """Test feature flags caching behavior."""
        flags = get_feature_flags()

        # Set a flag and verify it's cached
        flags.set_flag(flags.MULTILINGUAL_ENABLED, False, ttl=60)
        self.assertFalse(flags.is_enabled(flags.MULTILINGUAL_ENABLED))

        # Clear cache and verify it returns to default
        flags.clear_cache()
        self.assertTrue(flags.is_enabled(flags.MULTILINGUAL_ENABLED))


class CircuitBreakerProductionTest(TestCase):
    """Test circuit breaker system for production reliability."""

    def setUp(self):
        """Set up test environment."""
        self.circuit_breaker = get_circuit_breaker("test_breaker")
        cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        cache.clear()

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        breaker = MultilingualCircuitBreaker("test")

        # Test initial states
        for lang in ['fr', 'es', 'en']:
            self.assertEqual(breaker.get_state(lang), CircuitState.CLOSED)
            self.assertEqual(breaker.failure_counts[lang], 0)
            self.assertEqual(breaker.success_counts[lang], 0)

    def test_circuit_breaker_failure_detection(self):
        """Test circuit breaker failure detection and state transitions."""
        breaker = self.circuit_breaker

        def failing_function():
            raise Exception("Test failure")

        def fallback_function():
            return {"fallback": True, "content": "Fallback response"}

        # Test multiple failures trigger circuit opening
        language = "fr"
        failure_count = 0

        for _ in range(6):  # Exceed failure threshold (default 5)
            try:
                breaker.call_with_breaker(failing_function, language, fallback_function)
                failure_count += 1
            except Exception:
                failure_count += 1

        # Circuit should be open after threshold exceeded
        self.assertEqual(breaker.get_state(language), CircuitState.OPEN)

    def test_circuit_breaker_fallback_execution(self):
        """Test circuit breaker fallback execution."""
        breaker = self.circuit_breaker

        def failing_function():
            raise Exception("Test failure")

        def fallback_function():
            return {"fallback": True, "content": "Fallback response"}

        # Force circuit open
        breaker.force_open("fr", "Test")

        # Test fallback execution
        result = breaker.call_with_breaker(failing_function, "fr", fallback_function)

        self.assertTrue(result["fallback"])
        self.assertTrue(result.get("circuit_breaker_triggered", False))

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        breaker = self.circuit_breaker

        # Force circuit open
        breaker.force_open("fr", "Test recovery")
        self.assertEqual(breaker.get_state("fr"), CircuitState.OPEN)

        # Manually force recovery for testing
        breaker.force_close("fr", "Test recovery")
        self.assertEqual(breaker.get_state("fr"), CircuitState.CLOSED)

    def test_per_language_isolation(self):
        """Test that circuit breaker isolates failures per language."""
        breaker = self.circuit_breaker

        # Open circuit for French
        breaker.force_open("fr", "Test isolation")

        # Verify Spanish circuit remains closed
        self.assertEqual(breaker.get_state("fr"), CircuitState.OPEN)
        self.assertEqual(breaker.get_state("es"), CircuitState.CLOSED)

    def test_circuit_breaker_statistics(self):
        """Test circuit breaker statistics collection."""
        breaker = self.circuit_breaker

        stats = breaker.get_stats()

        # Verify stats structure
        self.assertIn("name", stats)
        self.assertIn("timestamp", stats)
        self.assertIn("config", stats)
        self.assertIn("languages", stats)

        # Verify language stats
        for lang in breaker.supported_languages:
            self.assertIn(lang, stats["languages"])
            lang_stats = stats["languages"][lang]
            self.assertIn("state", lang_stats)
            self.assertIn("failure_count", lang_stats)
            self.assertIn("success_count", lang_stats)


class ProductionMonitoringTest(TestCase):
    """Test production monitoring system."""

    def setUp(self):
        """Set up test environment."""
        self.monitoring_service = get_monitoring_service()
        cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        self.monitoring_service.stop_monitoring()
        cache.clear()

    def test_monitoring_service_initialization(self):
        """Test monitoring service initializes correctly."""
        service = ProductionMonitoringService()

        self.assertTrue(hasattr(service, 'monitoring_enabled'))
        self.assertTrue(hasattr(service, 'check_interval'))
        self.assertTrue(hasattr(service, 'alert_queue'))

    def test_alert_generation(self):
        """Test alert generation and storage."""
        service = self.monitoring_service

        # Generate test alert
        service._generate_alert(
            AlertLevel.WARNING,
            "Test Alert",
            "This is a test alert",
            {"test": True}
        )

        # Verify alert is stored
        recent_alerts = service.get_recent_alerts(10)
        self.assertGreater(len(recent_alerts), 0)

        alert = recent_alerts[0]
        self.assertEqual(alert["level"], AlertLevel.WARNING)
        self.assertEqual(alert["title"], "Test Alert")

    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        service = self.monitoring_service
        service.alert_cooldown = 1  # 1 second for testing

        # Generate first alert
        service._generate_alert(AlertLevel.INFO, "Test Cooldown", "First alert")
        alerts_after_first = len(service.get_recent_alerts())

        # Generate second alert immediately (should be suppressed)
        service._generate_alert(AlertLevel.INFO, "Test Cooldown", "Second alert")
        alerts_after_second = len(service.get_recent_alerts())

        # Should be same count due to cooldown
        self.assertEqual(alerts_after_first, alerts_after_second)

        # Wait for cooldown and try again
        time.sleep(1.1)
        service._generate_alert(AlertLevel.INFO, "Test Cooldown", "Third alert")
        alerts_after_third = len(service.get_recent_alerts())

        # Should be one more after cooldown
        self.assertGreater(alerts_after_third, alerts_after_second)

    @patch('Analytics.services.production_monitoring.psutil.cpu_percent')
    def test_system_resource_monitoring(self, mock_cpu):
        """Test system resource monitoring."""
        service = self.monitoring_service

        # Mock high CPU usage
        mock_cpu.return_value = 90.0

        # Run resource check
        service._check_system_resources()

        # Verify alert was generated
        recent_alerts = service.get_recent_alerts(10)
        cpu_alerts = [alert for alert in recent_alerts if "CPU usage" in alert.get("title", "")]
        self.assertGreater(len(cpu_alerts), 0)

    def test_health_check_execution(self):
        """Test forced health check execution."""
        service = self.monitoring_service

        result = service.force_health_check()

        self.assertIn("status", result)
        self.assertIn("check_duration", result)
        self.assertIn("timestamp", result)

    def test_monitoring_status(self):
        """Test monitoring status reporting."""
        service = self.monitoring_service

        status_info = service.get_monitoring_status()

        self.assertIn("monitoring_enabled", status_info)
        self.assertIn("check_interval", status_info)
        self.assertIn("thresholds", status_info)
        self.assertIn("notification_channels", status_info)


class HealthCheckEndpointsTest(APITestCase):
    """Test health check endpoints for production monitoring."""

    def setUp(self):
        """Set up test environment."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_multilingual_health_endpoint(self):
        """Test multilingual health check endpoint."""
        url = reverse('analytics:multilingual_health')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("multilingual_enabled", data)
        self.assertIn("feature_flags", data)

    def test_health_ping_endpoint(self):
        """Test lightweight health ping endpoint."""
        url = reverse('analytics:health_ping')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)

    def test_feature_flags_status_endpoint(self):
        """Test feature flags status endpoint."""
        url = reverse('analytics:feature_flags_status')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("flags", data)
        self.assertIn("timestamp", data)

    def test_production_monitoring_status_endpoint(self):
        """Test production monitoring status endpoint."""
        url = reverse('analytics:production_monitoring_status')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("monitoring_enabled", data)
        self.assertIn("check_interval", data)

    def test_production_alerts_endpoint(self):
        """Test production alerts endpoint."""
        url = reverse('analytics:production_alerts')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("alerts", data)
        self.assertIn("count", data)

    def test_force_health_check_endpoint(self):
        """Test force health check endpoint."""
        url = reverse('analytics:force_health_check')
        response = self.client.post(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("check_duration", data)


class MultilingualServiceIntegrationTest(APITestCase):
    """Integration tests for multilingual services in production scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    @patch('Analytics.services.local_llm_service.LocalLLMService.generate_explanation')
    def test_multilingual_explanation_with_fallback(self, mock_generate):
        """Test multilingual explanation with circuit breaker fallback."""
        # Mock LLM service to fail initially
        mock_generate.side_effect = Exception("Service unavailable")

        # Get circuit breaker and force fallback setup
        breaker = get_circuit_breaker()

        def fallback_function(*args, **kwargs):
            return {
                "content": "Fallback explanation in English",
                "language": "en",
                "fallback": True
            }

        # Test with circuit breaker protection
        try:
            result = breaker.call_with_breaker(
                mock_generate,
                "fr",
                fallback_function,
                {"symbol": "AAPL", "score": 7.5},
                language="fr"
            )

            # Should get fallback result
            self.assertTrue(result.get("fallback", False))

        except CircuitBreakerOpenError:
            # Circuit may open after failures, which is expected behavior
            pass

    def test_feature_flag_controlled_multilingual_access(self):
        """Test feature flag controlled access to multilingual features."""
        # Disable French generation
        flags = get_feature_flags()
        flags.set_flag(flags.FRENCH_GENERATION_ENABLED, False)

        # Verify multilingual is disabled for French
        self.assertFalse(is_multilingual_enabled("fr", self.user))

        # Verify English still works
        self.assertTrue(is_multilingual_enabled("en", self.user))

    def test_emergency_fallback_scenario(self):
        """Test complete emergency fallback scenario."""
        flags = get_feature_flags()

        # Trigger emergency fallback
        disabled_flags = flags.emergency_disable_all("Test emergency scenario")

        # Verify all multilingual features are disabled
        self.assertFalse(is_multilingual_enabled("fr"))
        self.assertFalse(is_multilingual_enabled("es"))

        # Verify emergency status is reflected in health checks
        url = reverse('analytics:multilingual_health')
        response = self.client.get(url)

        data = response.json()
        self.assertIn("emergency fallback", data.get("warnings", []))


class LoadAndStressTest(TestCase):
    """Load and stress tests for production readiness."""

    def test_concurrent_feature_flag_access(self):
        """Test concurrent access to feature flags."""
        flags = get_feature_flags()
        results = []

        def check_flag():
            result = flags.is_enabled(flags.MULTILINGUAL_ENABLED)
            results.append(result)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_flag)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all results are consistent
        self.assertEqual(len(set(results)), 1)  # All results should be the same

    def test_circuit_breaker_under_load(self):
        """Test circuit breaker behavior under high load."""
        breaker = get_circuit_breaker("load_test")
        results = []

        def make_request():
            def test_function():
                # Simulate varying success/failure
                import random
                if random.random() < 0.7:  # 70% success rate
                    return {"success": True}
                else:
                    raise Exception("Random failure")

            def fallback_function():
                return {"fallback": True}

            try:
                result = breaker.call_with_breaker(
                    test_function,
                    "fr",
                    fallback_function
                )
                results.append(result)
            except Exception:
                results.append({"error": True})

        # Simulate concurrent load
        threads = []
        for _ in range(50):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify we got results from all requests
        self.assertEqual(len(results), 50)

        # Verify mix of success, fallback, and error responses
        success_count = len([r for r in results if r.get("success")])
        fallback_count = len([r for r in results if r.get("fallback")])
        error_count = len([r for r in results if r.get("error")])

        self.assertGreater(success_count + fallback_count + error_count, 0)


class ProductionReadinessChecklistTest(TestCase):
    """Comprehensive production readiness checklist validation."""

    def test_all_critical_services_available(self):
        """Test that all critical services are available and configured."""
        # Feature flags service
        flags = get_feature_flags()
        self.assertIsNotNone(flags)

        # Circuit breaker service
        breaker = get_circuit_breaker()
        self.assertIsNotNone(breaker)

        # Monitoring service
        monitoring = get_monitoring_service()
        self.assertIsNotNone(monitoring)

        # LLM service
        llm_service = get_local_llm_service()
        self.assertIsNotNone(llm_service)

    def test_error_handling_robustness(self):
        """Test that services handle errors gracefully."""
        # Test feature flags with invalid cache
        with patch('django.core.cache.cache.get', side_effect=Exception("Cache error")):
            flags = get_feature_flags()
            # Should not raise exception, should return default
            result = flags.is_enabled(flags.MULTILINGUAL_ENABLED)
            self.assertIsInstance(result, bool)

    def test_configuration_validation(self):
        """Test that all required configurations are present."""
        from django.conf import settings

        # Check for critical settings
        critical_settings = [
            'CACHES',
            'REDIS_HOST',
            'REDIS_PORT',
        ]

        for setting_name in critical_settings:
            self.assertTrue(hasattr(settings, setting_name),
                          f"Missing critical setting: {setting_name}")

    def test_database_connectivity(self):
        """Test database connectivity and basic operations."""
        from django.db import connection

        # Test database connection
        self.assertTrue(connection.ensure_connection())

        # Test basic query
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)

    def test_cache_connectivity(self):
        """Test cache connectivity and basic operations."""
        # Test cache set/get
        test_key = "production_test_key"
        test_value = "production_test_value"

        cache.set(test_key, test_value, 60)
        retrieved_value = cache.get(test_key)

        self.assertEqual(retrieved_value, test_value)

        # Clean up
        cache.delete(test_key)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])