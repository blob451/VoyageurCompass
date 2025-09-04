"""
Real tests for advanced monitoring service.
Tests metrics collection, alert management, and performance profiling.
No mocks - uses real functionality.
"""

import time
from datetime import datetime, timedelta
from django.test import TestCase
from django.utils import timezone

from Analytics.services.advanced_monitoring_service import get_monitoring_service, profile_performance


class RealMonitoringServiceTestCase(TestCase):
    """Real test cases for advanced monitoring service."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitoring_service = get_monitoring_service()
        # Service is initialized and ready for testing
    
    def test_real_service_initialization(self):
        """Test real monitoring service initialization."""
        service = get_monitoring_service()
        
        self.assertIsNotNone(service)
        self.assertTrue(hasattr(service, 'record_llm_metrics'))
        self.assertTrue(hasattr(service, 'get_system_health'))
        self.assertTrue(hasattr(service, 'get_performance_dashboard'))
        self.assertTrue(hasattr(service, 'record_analysis_metrics'))
        
        # Verify singleton pattern
        service2 = get_monitoring_service()
        self.assertIs(service, service2)
    
    def test_real_metrics_collection(self):
        """Test real metrics collection functionality."""
        service = self.monitoring_service
        
        # Record various metrics
        service.record_metric('test_requests', 100, 'counter')
        service.record_metric('test_response_time', 250.5, 'gauge', unit='ms')
        service.record_metric('test_memory_usage', 1024.0, 'gauge', unit='MB')
        service.record_metric('test_success_rate', 0.95, 'gauge', unit='percent')
        
        # Get recorded metrics
        metrics = service.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_requests', metrics)
        self.assertIn('test_response_time', metrics)
        self.assertIn('test_memory_usage', metrics)
        self.assertIn('test_success_rate', metrics)
        
        # Check metric structure
        request_metric = metrics['test_requests']
        self.assertEqual(request_metric['value'], 100)
        self.assertEqual(request_metric['type'], 'counter')
        
        response_time_metric = metrics['test_response_time']
        self.assertEqual(response_time_metric['value'], 250.5)
        self.assertEqual(response_time_metric['unit'], 'ms')
    
    def test_real_system_health_monitoring(self):
        """Test real system health monitoring."""
        service = self.monitoring_service
        
        # Record some system metrics
        service.record_metric('cpu_usage', 45.2, 'gauge', unit='percent')
        service.record_metric('memory_usage', 2048.0, 'gauge', unit='MB')
        service.record_metric('disk_usage', 75.8, 'gauge', unit='percent')
        service.record_metric('active_connections', 125, 'gauge')
        
        # Get system health
        health = service.get_system_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('status', health)
        self.assertIn('metrics', health)
        self.assertIn('timestamp', health)
        
        # Check health status calculation
        self.assertIn(health['status'], ['healthy', 'warning', 'critical'])
        
        # Verify metrics are included
        health_metrics = health['metrics']
        self.assertIn('cpu_usage', health_metrics)
        self.assertIn('memory_usage', health_metrics)
        self.assertIn('disk_usage', health_metrics)
    
    def test_real_performance_dashboard(self):
        """Test real performance dashboard generation."""
        service = self.monitoring_service
        
        # Record performance metrics over time
        for i in range(10):
            service.record_metric('requests_per_second', 50 + i * 5, 'gauge')
            service.record_metric('avg_response_time', 200 + i * 10, 'gauge', unit='ms')
            service.record_metric('error_rate', 0.01 + i * 0.005, 'gauge', unit='percent')
            time.sleep(0.01)  # Small delay to create time separation
        
        # Get performance dashboard
        dashboard = service.get_performance_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn('summary', dashboard)
        self.assertIn('metrics_history', dashboard)
        self.assertIn('trends', dashboard)
        self.assertIn('generated_at', dashboard)
        
        # Check summary statistics
        summary = dashboard['summary']
        self.assertIn('total_metrics', summary)
        self.assertIn('active_alerts', summary)
        self.assertIn('system_uptime', summary)
        
        # Check metrics history
        history = dashboard['metrics_history']
        self.assertGreater(len(history), 0)
    
    def test_real_alert_management(self):
        """Test real alert management functionality."""
        service = self.monitoring_service
        
        # Create test alerts
        alert1 = service.create_alert(
            'high_cpu_usage',
            'CPU usage exceeded threshold',
            'warning',
            {'cpu_usage': 85.5, 'threshold': 80.0}
        )
        
        alert2 = service.create_alert(
            'memory_leak_detected',
            'Memory usage growing rapidly',
            'critical',
            {'memory_growth_rate': '15MB/min', 'current_usage': '4GB'}
        )
        
        # Verify alert creation
        self.assertIsNotNone(alert1)
        self.assertIsNotNone(alert2)
        
        # Get recent alerts
        recent_alerts = service.get_recent_alerts()
        
        self.assertIsInstance(recent_alerts, list)
        self.assertGreaterEqual(len(recent_alerts), 2)
        
        # Check alert structure
        for alert in recent_alerts:
            self.assertIn('alert_id', alert)
            self.assertIn('alert_type', alert)
            self.assertIn('message', alert)
            self.assertIn('severity', alert)
            self.assertIn('created_at', alert)
            self.assertIn('metadata', alert)
        
        # Test alert filtering by severity
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'critical']
        self.assertGreaterEqual(len(critical_alerts), 1)
    
    def test_real_performance_profiling(self):
        """Test real performance profiling decorator."""
        
        @profile_performance('test_operation')
        def test_operation(duration=0.1):
            """Test operation to profile."""
            time.sleep(duration)
            return {'result': 'success', 'processed_items': 100}
        
        # Execute profiled operation
        result = test_operation(0.05)
        
        # Check operation result
        self.assertEqual(result['result'], 'success')
        self.assertEqual(result['processed_items'], 100)
        
        # Check that profiling data was recorded
        service = self.monitoring_service
        metrics = service.get_metrics()
        
        # Should have profiling metrics
        profile_metrics = {k: v for k, v in metrics.items() if 'test_operation' in k}
        self.assertGreater(len(profile_metrics), 0)
        
        # Check for expected profiling metrics
        expected_patterns = ['_duration', '_calls', '_avg_duration']
        found_patterns = []
        
        for metric_name in profile_metrics.keys():
            for pattern in expected_patterns:
                if pattern in metric_name:
                    found_patterns.append(pattern)
        
        self.assertGreater(len(found_patterns), 0)
    
    def test_real_metric_history_tracking(self):
        """Test real metric history tracking."""
        service = self.monitoring_service
        
        # Record metric values over time
        test_values = [10, 15, 12, 20, 18, 25, 22, 30]
        
        for value in test_values:
            service.record_metric('test_history_metric', value, 'gauge')
            time.sleep(0.01)  # Small delay
        
        # Get metric history
        history = service.get_metric_history('test_history_metric')
        
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), len(test_values))
        
        # Check history structure
        for entry in history:
            self.assertIn('value', entry)
            self.assertIn('timestamp', entry)
            self.assertIsInstance(entry['timestamp'], (str, datetime))
    
    def test_real_performance_thresholds(self):
        """Test real performance threshold monitoring."""
        service = self.monitoring_service
        
        # Set up performance thresholds
        thresholds = {
            'response_time': {'warning': 500, 'critical': 1000, 'unit': 'ms'},
            'error_rate': {'warning': 0.05, 'critical': 0.10, 'unit': 'percent'},
            'cpu_usage': {'warning': 70, 'critical': 90, 'unit': 'percent'}
        }
        
        service.set_performance_thresholds(thresholds)
        
        # Record metrics that exceed thresholds
        service.record_metric('response_time', 750, 'gauge', unit='ms')  # Warning
        service.record_metric('error_rate', 0.12, 'gauge', unit='percent')  # Critical
        service.record_metric('cpu_usage', 45, 'gauge', unit='percent')  # Normal
        
        # Check threshold violations
        violations = service.get_threshold_violations()
        
        self.assertIsInstance(violations, list)
        self.assertGreaterEqual(len(violations), 2)  # Should have response_time and error_rate
        
        # Check violation structure
        for violation in violations:
            self.assertIn('metric', violation)
            self.assertIn('value', violation)
            self.assertIn('threshold', violation)
            self.assertIn('severity', violation)
            self.assertIn('timestamp', violation)


class RealMonitoringIntegrationTestCase(TestCase):
    """Integration tests for monitoring service with other components."""
    
    def test_service_status_monitoring(self):
        """Test monitoring of service status."""
        service = get_monitoring_service()
        
        # Get overall service status
        status = service.get_service_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('monitoring_enabled', status)
        self.assertIn('metrics_collected', status)
        self.assertIn('alerts_active', status)
        self.assertIn('uptime', status)
        
        # Should be monitoring enabled
        self.assertTrue(status['monitoring_enabled'])
        self.assertGreaterEqual(status['metrics_collected'], 0)
    
    def test_monitoring_resource_usage(self):
        """Test monitoring of resource usage."""
        service = get_monitoring_service()
        
        # Record resource metrics
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        service.record_metric('process_memory_rss', memory_info.rss / 1024 / 1024, 'gauge', unit='MB')
        service.record_metric('process_cpu_percent', process.cpu_percent(), 'gauge', unit='percent')
        
        # Get resource dashboard
        dashboard = service.get_performance_dashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIn('summary', dashboard)
        
        # Check that resource metrics are included
        metrics = service.get_metrics()
        self.assertIn('process_memory_rss', metrics)
        self.assertIn('process_cpu_percent', metrics)


if __name__ == '__main__':
    import django
    django.setup()
    from django.test import TestRunner
    runner = TestRunner()
    runner.run_tests(['Analytics.tests.test_monitoring_service_real'])