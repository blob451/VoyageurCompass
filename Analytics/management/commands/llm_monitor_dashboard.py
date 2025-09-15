"""
LLM Monitoring Dashboard Management Command

Display comprehensive LLM monitoring data including:
- Real-time performance metrics
- Error tracking and analysis
- Resource utilisation statistics
- Alert notifications
- Performance recommendations
"""

import json
import logging
import time
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Display LLM monitoring dashboard data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--format',
            choices=['table', 'json', 'summary'],
            default='table',
            help='Output format (default: table)'
        )
        
        parser.add_argument(
            '--watch',
            action='store_true',
            help='Continuously monitor and refresh display every 30 seconds'
        )
        
        parser.add_argument(
            '--alerts-only',
            action='store_true',
            help='Show only current alerts'
        )

    def handle(self, *args, **options):
        """Execute LLM monitoring dashboard display."""
        
        try:
            # Import monitoring components
            from Analytics.monitoring.llm_monitor import get_llm_monitoring_dashboard_data
        except ImportError:
            raise CommandError("LLM monitoring system not available")

        if options['watch']:
            self._watch_dashboard(options)
        else:
            self._display_dashboard(options)

    def _display_dashboard(self, options):
        """Display dashboard data once."""
        try:
            from Analytics.monitoring.llm_monitor import get_llm_monitoring_dashboard_data
            
            dashboard_data = get_llm_monitoring_dashboard_data()
            
            if options['alerts_only']:
                self._display_alerts(dashboard_data.get('alerts', []))
            elif options['format'] == 'json':
                self.stdout.write(json.dumps(dashboard_data, indent=2))
            elif options['format'] == 'summary':
                self._display_summary(dashboard_data)
            else:
                self._display_table(dashboard_data)
                
        except Exception as e:
            raise CommandError(f"Failed to retrieve monitoring data: {str(e)}")

    def _watch_dashboard(self, options):
        """Continuously watch and display dashboard data."""
        self.stdout.write(self.style.SUCCESS('Starting LLM monitoring dashboard (Ctrl+C to stop)...'))
        
        try:
            while True:
                # Clear screen (works on most terminals)
                self.stdout.write('\033[2J\033[H')  # ANSI escape codes
                
                # Display timestamp
                self.stdout.write(self.style.SUCCESS(f'LLM Monitoring Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'))
                self.stdout.write('-' * 80)
                
                # Display current data
                self._display_dashboard(options)
                
                # Wait for next update
                time.sleep(30)
                
        except KeyboardInterrupt:
            self.stdout.write('\nDashboard monitoring stopped.')

    def _display_table(self, dashboard_data):
        """Display comprehensive dashboard in table format."""
        health = dashboard_data.get('health', {})
        metrics = dashboard_data.get('metrics', {})
        trends = dashboard_data.get('trends', {})
        alerts = dashboard_data.get('alerts', [])
        recommendations = dashboard_data.get('recommendations', [])
        
        # System Health
        status = health.get('status', 'unknown')
        status_color = {
            'healthy': self.style.SUCCESS,
            'warning': self.style.WARNING,
            'critical': self.style.ERROR
        }.get(status, self.style.WARNING)
        
        self.stdout.write(f"\n{status_color('System Health:')} {status.upper()}")
        
        if health.get('issues'):
            for issue in health['issues']:
                self.stdout.write(f"  ⚠️  {issue}")
        
        # Request Metrics
        requests = metrics.get('requests', {})
        if requests:
            self.stdout.write('\nRequest Metrics:')
            self.stdout.write(f"  Total Requests: {requests.get('total', 0):,}")
            self.stdout.write(f"  Successful: {requests.get('successful', 0):,}")
            self.stdout.write(f"  Failed: {requests.get('failed', 0):,}")
            self.stdout.write(f"  Success Rate: {requests.get('success_rate', 0):.1f}%")
            self.stdout.write(f"  Requests/sec: {requests.get('requests_per_second', 0):.2f}")
        
        # Response Time Metrics
        response_times = metrics.get('response_times', {})
        if response_times and response_times.get('count', 0) > 0:
            self.stdout.write('\nResponse Time Metrics:')
            self.stdout.write(f"  Average: {response_times.get('avg', 0):.2f}s")
            self.stdout.write(f"  Median (95th percentile): {response_times.get('p95', 0):.2f}s")
            self.stdout.write(f"  Min: {response_times.get('min', 0):.2f}s")
            self.stdout.write(f"  Max: {response_times.get('max', 0):.2f}s")
        
        # Quality Metrics
        quality = metrics.get('quality', {})
        if quality and quality.get('count', 0) > 0:
            self.stdout.write('\nQuality Metrics:')
            self.stdout.write(f"  Average Score: {quality.get('avg', 0):.3f}")
            self.stdout.write(f"  Min Score: {quality.get('min', 0):.3f}")
            self.stdout.write(f"  Max Score: {quality.get('max', 0):.3f}")
        
        # Model Usage
        models = metrics.get('models', {})
        if models:
            self.stdout.write('\nModel Usage:')
            total_model_usage = sum(models.values())
            for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_model_usage) * 100 if total_model_usage > 0 else 0
                self.stdout.write(f"  {model}: {count:,} ({percentage:.1f}%)")
        
        # Detail Level Usage
        detail_levels = metrics.get('detail_levels', {})
        if detail_levels:
            self.stdout.write('\nDetail Level Usage:')
            total_detail_usage = sum(detail_levels.values())
            for level, count in sorted(detail_levels.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detail_usage) * 100 if total_detail_usage > 0 else 0
                self.stdout.write(f"  {level}: {count:,} ({percentage:.1f}%)")
        
        # Cache Metrics
        cache_metrics = metrics.get('cache', {})
        if cache_metrics:
            self.stdout.write('\nCache Performance:')
            self.stdout.write(f"  Hits: {cache_metrics.get('hits', 0):,}")
            self.stdout.write(f"  Misses: {cache_metrics.get('misses', 0):,}")
            self.stdout.write(f"  Hit Rate: {cache_metrics.get('hit_rate', 0):.1f}%")
        
        # Error Summary
        errors = metrics.get('errors', {})
        if errors and errors.get('total_errors', 0) > 0:
            self.stdout.write('\nError Summary:')
            self.stdout.write(f"  Total Errors: {errors.get('total_errors', 0):,}")
            self.stdout.write(f"  Error Rate: {errors.get('error_rate', 0):.2f}%")
            
            error_types = errors.get('error_types', {})
            if error_types:
                self.stdout.write('  Error Types:')
                for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    self.stdout.write(f"    {error_type}: {count:,}")
        
        # Trends
        if trends:
            self.stdout.write('\nPerformance Trends:')
            
            response_time_change = trends.get('response_time_change_percent')
            if response_time_change is not None:
                trend_symbol = "📈" if response_time_change > 0 else "📉" if response_time_change < 0 else "➡️"
                self.stdout.write(f"  Response Time: {trend_symbol} {response_time_change:+.1f}%")
            
            success_rate_change = trends.get('success_rate_change_percent')
            if success_rate_change is not None:
                trend_symbol = "📈" if success_rate_change > 0 else "📉" if success_rate_change < 0 else "➡️"
                self.stdout.write(f"  Success Rate: {trend_symbol} {success_rate_change:+.1f}%")
            
            volume_change = trends.get('request_volume_change')
            if volume_change is not None:
                trend_symbol = "📈" if volume_change > 0 else "📉" if volume_change < 0 else "➡️"
                self.stdout.write(f"  Request Volume: {trend_symbol} {volume_change:+,} requests")
        
        # Current Alerts
        if alerts:
            self.stdout.write(self.style.ERROR('\nActive Alerts:'))
            for alert in alerts:
                self.stdout.write(f"  🚨 {alert}")
        
        # Recommendations
        if recommendations:
            self.stdout.write('\nOptimisation Recommendations:')
            for recommendation in recommendations:
                self.stdout.write(f"  💡 {recommendation}")
        
        # Footer
        uptime = metrics.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        self.stdout.write(f"\nSystem Uptime: {uptime_hours:.1f} hours")

    def _display_summary(self, dashboard_data):
        """Display condensed summary format."""
        health = dashboard_data.get('health', {})
        metrics = dashboard_data.get('metrics', {})
        alerts = dashboard_data.get('alerts', [])
        
        # One-line status
        status = health.get('status', 'unknown')
        requests = metrics.get('requests', {})
        response_times = metrics.get('response_times', {})
        
        status_icon = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🚨'
        }.get(status, '❓')
        
        summary_line = (f"{status_icon} Status: {status.upper()} | "
                       f"Requests: {requests.get('total', 0):,} "
                       f"({requests.get('success_rate', 0):.1f}% success) | "
                       f"Avg Response: {response_times.get('avg', 0):.2f}s")
        
        if alerts:
            summary_line += f" | Alerts: {len(alerts)}"
        
        self.stdout.write(summary_line)
        
        # Show alerts if any
        if alerts:
            for alert in alerts:
                self.stdout.write(f"  🚨 {alert}")

    def _display_alerts(self, alerts):
        """Display only current alerts."""
        if not alerts:
            self.stdout.write(self.style.SUCCESS('No active alerts'))
        else:
            self.stdout.write(self.style.ERROR(f'Active Alerts ({len(alerts)}):'))
            for i, alert in enumerate(alerts, 1):
                self.stdout.write(f"{i}. {alert}")

    def _clear_metrics(self, options):
        """Clear monitoring metrics (admin function)."""
        try:
            from Analytics.monitoring.llm_monitor import llm_metrics
            
            llm_metrics.reset_metrics()
            self.stdout.write(self.style.SUCCESS('LLM monitoring metrics cleared'))
            
        except ImportError:
            raise CommandError("LLM monitoring system not available")
        except Exception as e:
            raise CommandError(f"Failed to clear metrics: {str(e)}")