"""
Advanced Monitoring and Analytics Service for VoyageurCompass.
Provides comprehensive system monitoring, performance tracking, and operational analytics.
"""

import json
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import sqlite3
import uuid

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.utils import timezone

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates system and application metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))  # Ring buffer for metrics
        self.lock = threading.Lock()
        
        # Metric definitions
        self.metric_definitions = {
            'system_cpu_percent': {'type': 'gauge', 'unit': 'percent'},
            'system_memory_percent': {'type': 'gauge', 'unit': 'percent'},
            'system_disk_usage_percent': {'type': 'gauge', 'unit': 'percent'},
            'django_db_connections': {'type': 'gauge', 'unit': 'count'},
            'llm_generation_time': {'type': 'histogram', 'unit': 'seconds'},
            'llm_request_count': {'type': 'counter', 'unit': 'count'},
            'llm_error_count': {'type': 'counter', 'unit': 'count'},
            'cache_hit_rate': {'type': 'gauge', 'unit': 'percent'},
            'analysis_request_count': {'type': 'counter', 'unit': 'count'},
            'analysis_success_rate': {'type': 'gauge', 'unit': 'percent'},
            'sentiment_analysis_time': {'type': 'histogram', 'unit': 'seconds'},
            'hybrid_coordination_time': {'type': 'histogram', 'unit': 'seconds'},
            'async_pipeline_queue_size': {'type': 'gauge', 'unit': 'count'},
            'async_pipeline_processing_time': {'type': 'histogram', 'unit': 'seconds'}
        }
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None, timestamp: datetime = None):
        """
        Record a metric value with optional labels and timestamp.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'timestamp': timestamp.isoformat(),
            'value': value,
            'labels': labels or {}
        }
        
        with self.lock:
            self.metrics_buffer[metric_name].append(metric_entry)
    
    def get_metric_history(self, 
                          metric_name: str, 
                          start_time: datetime = None, 
                          end_time: datetime = None) -> List[Dict[str, Any]]:
        """Get historical values for a metric within time range."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=self.retention_hours)
        if end_time is None:
            end_time = datetime.now()
        
        with self.lock:
            metric_data = self.metrics_buffer.get(metric_name, [])
            
            filtered_data = []
            for entry in metric_data:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if start_time <= entry_time <= end_time:
                    filtered_data.append(entry)
            
            return sorted(filtered_data, key=lambda x: x['timestamp'])
    
    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistical summary for a metric over specified hours."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        history = self.get_metric_history(metric_name, start_time, end_time)
        
        if not history:
            return {'error': f'No data for metric {metric_name}'}
        
        values = [entry['value'] for entry in history]
        
        return {
            'metric_name': metric_name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else None,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('system_cpu_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('system_memory_percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('system_disk_usage_percent', disk_percent)
            
            # Database connections
            db_connections = len(connection.queries)
            self.record_metric('django_db_connections', db_connections)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for metric_name in self.metrics_buffer:
                metric_data = self.metrics_buffer[metric_name]
                # Remove old entries
                while metric_data and datetime.fromisoformat(metric_data[0]['timestamp']) < cutoff_time:
                    metric_data.popleft()


class PerformanceProfiler:
    """Profiles performance of key application components."""
    
    def __init__(self):
        self.active_profiles = {}
        self.completed_profiles = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def start_profile(self, profile_name: str, operation_type: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new performance profile."""
        profile_id = str(uuid.uuid4())[:8]
        
        profile_info = {
            'profile_id': profile_id,
            'profile_name': profile_name,
            'operation_type': operation_type,
            'metadata': metadata or {},
            'start_time': time.time(),
            'start_timestamp': datetime.now().isoformat(),
            'checkpoints': []
        }
        
        with self.lock:
            self.active_profiles[profile_id] = profile_info
        
        return profile_id
    
    def add_checkpoint(self, profile_id: str, checkpoint_name: str, metadata: Dict[str, Any] = None):
        """Add a checkpoint to an active profile."""
        with self.lock:
            if profile_id in self.active_profiles:
                profile = self.active_profiles[profile_id]
                checkpoint_time = time.time()
                
                checkpoint = {
                    'name': checkpoint_name,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_time': checkpoint_time - profile['start_time'],
                    'metadata': metadata or {}
                }
                
                profile['checkpoints'].append(checkpoint)
    
    def end_profile(self, profile_id: str, status: str = 'completed', metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """End a performance profile and return results."""
        with self.lock:
            if profile_id not in self.active_profiles:
                return {'error': 'Profile not found'}
            
            profile = self.active_profiles.pop(profile_id)
            end_time = time.time()
            
            # Calculate final results
            profile.update({
                'end_time': end_time,
                'end_timestamp': datetime.now().isoformat(),
                'total_duration': end_time - profile['start_time'],
                'status': status,
                'final_metadata': metadata or {}
            })
            
            # Calculate checkpoint intervals
            prev_time = profile['start_time']
            for checkpoint in profile['checkpoints']:
                checkpoint['interval_duration'] = checkpoint['elapsed_time'] - (prev_time - profile['start_time'])
                prev_time = profile['start_time'] + checkpoint['elapsed_time']
            
            # Store completed profile
            self.completed_profiles.append(profile)
            
            return profile
    
    def get_profile_summary(self, operation_type: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for profiles."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Filter profiles by time and operation type
        filtered_profiles = []
        for profile in self.completed_profiles:
            profile_time = datetime.fromisoformat(profile['start_timestamp'])
            if start_time <= profile_time <= end_time:
                if operation_type is None or profile['operation_type'] == operation_type:
                    filtered_profiles.append(profile)
        
        if not filtered_profiles:
            return {'message': 'No profiles found for criteria'}
        
        # Calculate statistics
        durations = [p['total_duration'] for p in filtered_profiles]
        successful_profiles = [p for p in filtered_profiles if p['status'] == 'completed']
        
        return {
            'operation_type': operation_type or 'all',
            'total_profiles': len(filtered_profiles),
            'successful_profiles': len(successful_profiles),
            'success_rate': len(successful_profiles) / len(filtered_profiles) if filtered_profiles else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
        }


class AlertManager:
    """Manages monitoring alerts and notifications."""
    
    def __init__(self):
        self.alerts = deque(maxlen=500)  # Store recent alerts
        self.alert_rules = {
            'high_cpu_usage': {
                'metric': 'system_cpu_percent',
                'threshold': 80,
                'condition': 'greater_than',
                'severity': 'warning',
                'enabled': True
            },
            'high_memory_usage': {
                'metric': 'system_memory_percent', 
                'threshold': 85,
                'condition': 'greater_than',
                'severity': 'warning',
                'enabled': True
            },
            'slow_llm_generation': {
                'metric': 'llm_generation_time',
                'threshold': 30,
                'condition': 'greater_than',
                'severity': 'warning',
                'enabled': True
            },
            'high_error_rate': {
                'metric': 'llm_error_count',
                'threshold': 10,
                'condition': 'greater_than',
                'severity': 'critical',
                'enabled': True
            }
        }
        self.lock = threading.Lock()
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check metrics against alert rules and trigger alerts."""
        for rule_name, rule_config in self.alert_rules.items():
            if not rule_config.get('enabled', True):
                continue
            
            try:
                metric_name = rule_config['metric']
                threshold = rule_config['threshold']
                condition = rule_config['condition']
                severity = rule_config['severity']
                
                # Get latest metric value
                recent_metrics = metrics_collector.get_metric_history(
                    metric_name, 
                    datetime.now() - timedelta(minutes=5)
                )
                
                if recent_metrics:
                    latest_value = recent_metrics[-1]['value']
                    
                    # Check condition
                    alert_triggered = False
                    if condition == 'greater_than' and latest_value > threshold:
                        alert_triggered = True
                    elif condition == 'less_than' and latest_value < threshold:
                        alert_triggered = True
                    
                    if alert_triggered:
                        self.trigger_alert(
                            rule_name=rule_name,
                            metric_name=metric_name,
                            current_value=latest_value,
                            threshold=threshold,
                            severity=severity
                        )
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {str(e)}")
    
    def trigger_alert(self, 
                     rule_name: str, 
                     metric_name: str, 
                     current_value: float, 
                     threshold: float, 
                     severity: str):
        """Trigger an alert."""
        alert = {
            'alert_id': str(uuid.uuid4())[:8],
            'rule_name': rule_name,
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'message': f"{rule_name}: {metric_name} = {current_value:.2f} (threshold: {threshold})"
        }
        
        with self.lock:
            self.alerts.append(alert)
        
        logger.warning(f"ALERT: {alert['message']}")
    
    def get_recent_alerts(self, severity: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts with optional severity filter."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time >= cutoff_time:
                if severity is None or alert['severity'] == severity:
                    filtered_alerts.append(alert)
        
        return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)


class AdvancedMonitoringService:
    """Main monitoring service orchestrating all monitoring components."""
    
    def __init__(self, enable_background_collection: bool = True):
        self.metrics_collector = MetricsCollector()
        self.performance_profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
        self.start_time = time.time()  # Track service start time
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        if enable_background_collection:
            self.start_background_monitoring()
        
        logger.info("Advanced Monitoring Service initialized")
    
    def start_background_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Background monitoring stopped")
    
    def _background_monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(self.metrics_collector)
                
                # Cleanup old data
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.metrics_collector.cleanup_old_metrics()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def record_llm_metrics(self, 
                          generation_time: float, 
                          success: bool, 
                          model_name: str = None,
                          complexity_score: float = None):
        """Record LLM operation metrics."""
        labels = {}
        if model_name:
            labels['model'] = model_name
        if complexity_score is not None:
            labels['complexity'] = str(int(complexity_score * 10))  # Bucket by complexity
        
        self.metrics_collector.record_metric('llm_generation_time', generation_time, labels)
        self.metrics_collector.record_metric('llm_request_count', 1, labels)
        
        if not success:
            self.metrics_collector.record_metric('llm_error_count', 1, labels)
    
    def record_analysis_metrics(self, 
                               processing_time: float, 
                               success: bool,
                               analysis_type: str = 'technical'):
        """Record analysis operation metrics."""
        labels = {'type': analysis_type}
        
        self.metrics_collector.record_metric('analysis_request_count', 1, labels)
        
        if success:
            success_rate = 1.0
        else:
            success_rate = 0.0
        
        self.metrics_collector.record_metric('analysis_success_rate', success_rate, labels)
    
    def record_cache_metrics(self, hits: int, misses: int):
        """Record cache performance metrics."""
        total_requests = hits + misses
        if total_requests > 0:
            hit_rate = (hits / total_requests) * 100
            self.metrics_collector.record_metric('cache_hit_rate', hit_rate)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': {},
            'alerts': {
                'critical': 0,
                'warning': 0
            }
        }
        
        # System metrics
        for metric_name in ['system_cpu_percent', 'system_memory_percent', 'system_disk_usage_percent']:
            summary = self.metrics_collector.get_metric_summary(metric_name, hours=0.1)  # Last 6 minutes
            if 'error' not in summary:
                health_status['metrics'][metric_name] = summary['latest']
                
                # Check health thresholds
                if summary['latest'] > 90:
                    health_status['status'] = 'degraded'
        
        # LLM service metrics
        llm_summary = self.metrics_collector.get_metric_summary('llm_generation_time', hours=1)
        if 'error' not in llm_summary:
            health_status['components']['llm_service'] = {
                'status': 'healthy' if llm_summary['avg'] < 10 else 'degraded',
                'avg_response_time': llm_summary['avg'],
                'request_count': llm_summary['count']
            }
        
        # Recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(hours=1)
        for alert in recent_alerts:
            health_status['alerts'][alert['severity']] += 1
        
        # Overall status based on alerts
        if health_status['alerts']['critical'] > 0:
            health_status['status'] = 'unhealthy'
        elif health_status['alerts']['warning'] > 5:
            health_status['status'] = 'degraded'
        
        return health_status
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        dashboard_data = {
            'system_health': self.get_system_health(),
            'performance_metrics': {},
            'recent_alerts': self.alert_manager.get_recent_alerts(hours=24),
            'profile_summaries': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # Key performance metrics
        key_metrics = [
            'llm_generation_time', 'llm_request_count', 'analysis_request_count',
            'cache_hit_rate', 'sentiment_analysis_time', 'hybrid_coordination_time'
        ]
        
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, hours=24)
            if 'error' not in summary:
                dashboard_data['performance_metrics'][metric_name] = summary
        
        # Performance profile summaries
        for operation_type in ['llm_generation', 'analysis', 'sentiment_analysis', 'hybrid_coordination']:
            profile_summary = self.performance_profiler.get_profile_summary(operation_type, hours=24)
            if 'message' not in profile_summary:
                dashboard_data['profile_summaries'][operation_type] = profile_summary
        
        return dashboard_data
    
    def record_metric(self, metric_name: str, value: float, metric_type: str = 'gauge', unit: str = None):
        """
        Record a metric value using the underlying metrics collector.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (gauge, counter, histogram)
            unit: Optional unit for the metric
        """
        labels = {'unit': unit, 'type': metric_type} if unit else {'type': metric_type}
        # Store metric type in definitions if not already present
        if metric_name not in self.metrics_collector.metric_definitions:
            self.metrics_collector.metric_definitions[metric_name] = {'type': metric_type, 'unit': unit}
        self.metrics_collector.record_metric(metric_name, value, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary of metrics with their current values and metadata
        """
        metrics_dict = {}
        for metric_name, metric_data in self.metrics_collector.metrics_buffer.items():
            if metric_data:
                latest_entry = metric_data[-1]
                metrics_dict[metric_name] = {
                    'value': latest_entry['value'],
                    'type': self.metrics_collector.metric_definitions.get(metric_name, {}).get('type', 'gauge'),
                    'unit': latest_entry['labels'].get('unit') if latest_entry.get('labels') else None,
                    'timestamp': latest_entry['timestamp']
                }
        return metrics_dict
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of historical entries for the metric
        """
        return self.metrics_collector.get_metric_history(metric_name)
    
    def set_performance_thresholds(self, thresholds: Dict[str, Dict[str, Any]]):
        """
        Set performance thresholds for metrics.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold configurations
        """
        # Store thresholds for later use (could be implemented as needed)
        self.performance_thresholds = thresholds
    
    def get_threshold_violations(self) -> List[Dict[str, Any]]:
        """
        Get metrics that are violating configured thresholds.
        
        Returns:
            List of threshold violations
        """
        violations = []
        if not hasattr(self, 'performance_thresholds'):
            return violations
        
        for metric_name, threshold_config in self.performance_thresholds.items():
            # Get latest metric value
            history = self.metrics_collector.get_metric_history(metric_name)
            if history:
                latest_value = history[-1]['value']
                
                # Check thresholds
                severity = None
                threshold = None
                
                if 'critical' in threshold_config and latest_value > threshold_config['critical']:
                    severity = 'critical'
                    threshold = threshold_config['critical']
                elif 'warning' in threshold_config and latest_value > threshold_config['warning']:
                    severity = 'warning'
                    threshold = threshold_config['warning']
                
                if severity:
                    violations.append({
                        'metric': metric_name,
                        'value': latest_value,
                        'threshold': threshold,
                        'severity': severity,
                        'timestamp': history[-1]['timestamp']
                    })
        
        return violations
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get overall service status information.
        
        Returns:
            Dictionary with service status details
        """
        return {
            'monitoring_enabled': self.monitoring_active,
            'metrics_collected': sum(len(data) for data in self.metrics_collector.metrics_buffer.values()),
            'alerts_active': len(self.alert_manager.alerts),
            'uptime': time.time() - (self.start_time if hasattr(self, 'start_time') else time.time())
        }
    
    def create_alert(self, alert_type: str, message: str, severity: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (warning, critical, etc.)
            metadata: Additional metadata
            
        Returns:
            Created alert information
        """
        alert = {
            'alert_id': str(uuid.uuid4())[:8],
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        with self.alert_manager.lock:
            self.alert_manager.alerts.append(alert)
        
        return alert
    
    def get_recent_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts with optional severity filter.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of recent alerts
        """
        return self.alert_manager.get_recent_alerts(severity=severity)


# Singleton instance
_monitoring_service = None


def get_monitoring_service() -> AdvancedMonitoringService:
    """Get singleton instance of AdvancedMonitoringService."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = AdvancedMonitoringService()
    return _monitoring_service


# Context manager for performance profiling
class profile_performance:
    """Context manager for easy performance profiling."""
    
    def __init__(self, operation_name: str, operation_type: str, metadata: Dict[str, Any] = None):
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.metadata = metadata or {}
        self.monitoring_service = get_monitoring_service()
        self.profile_id = None
    
    def __enter__(self):
        self.profile_id = self.monitoring_service.performance_profiler.start_profile(
            self.operation_name, self.operation_type, self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id:
            status = 'completed' if exc_type is None else 'failed'
            error_metadata = {'error': str(exc_val)} if exc_val else {}
            self.monitoring_service.performance_profiler.end_profile(
                self.profile_id, status, error_metadata
            )
    
    def checkpoint(self, checkpoint_name: str, metadata: Dict[str, Any] = None):
        """Add a checkpoint to the current profile."""
        if self.profile_id:
            self.monitoring_service.performance_profiler.add_checkpoint(
                self.profile_id, checkpoint_name, metadata
            )