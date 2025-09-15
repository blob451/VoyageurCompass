"""
Django management command for performance monitoring dashboard.
"""

import json
import time
from datetime import datetime

from django.core.management.base import BaseCommand
from django.utils import timezone

from Analytics.services.performance_monitoring_service import get_performance_monitoring_service


class Command(BaseCommand):
    help = 'Display performance monitoring dashboard and system health status'

    def add_arguments(self, parser):
        parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)',
        )

        parser.add_argument(
            '--watch',
            action='store_true',
            help='Continuously update the dashboard every 10 seconds',
        )

        parser.add_argument(
            '--metrics-only',
            action='store_true',
            help='Show only metrics without health status',
        )

        parser.add_argument(
            '--health-only',
            action='store_true',
            help='Show only health status without detailed metrics',
        )

        parser.add_argument(
            '--service',
            type=str,
            help='Show metrics for specific service only',
        )

        parser.add_argument(
            '--hours',
            type=int,
            default=1,
            help='Number of hours of metrics to display (default: 1)',
        )

    def handle(self, *args, **options):
        monitoring_service = get_performance_monitoring_service()

        if options['watch']:
            self._watch_dashboard(monitoring_service, options)
        else:
            self._show_dashboard(monitoring_service, options)

    def _watch_dashboard(self, monitoring_service, options):
        """Continuously update and display the dashboard."""
        try:
            while True:
                # Clear screen
                import os
                os.system('cls' if os.name == 'nt' else 'clear')

                # Show current time
                self.stdout.write(
                    self.style.SUCCESS(f"Performance Dashboard - {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
                )
                self.stdout.write("=" * 80)

                # Show dashboard
                self._show_dashboard(monitoring_service, options)

                # Wait for next update
                self.stdout.write("\nPress Ctrl+C to exit. Refreshing in 10 seconds...")
                time.sleep(10)

        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS("\nDashboard monitoring stopped."))

    def _show_dashboard(self, monitoring_service, options):
        """Display the monitoring dashboard."""
        try:
            if options['health_only']:
                self._show_health_status(monitoring_service, options)
            elif options['metrics_only']:
                self._show_metrics_only(monitoring_service, options)
            else:
                # Show full dashboard
                dashboard_data = monitoring_service.get_performance_dashboard()

                if options['format'] == 'json':
                    self.stdout.write(json.dumps(dashboard_data, indent=2, default=str))
                else:
                    self._display_full_dashboard(dashboard_data, options)

        except Exception as e:
            error_msg = f"Failed to get dashboard data: {str(e)}"
            if options['format'] == 'json':
                self.stdout.write(json.dumps({"error": error_msg}))
            else:
                self.stdout.write(self.style.ERROR(error_msg))

    def _show_health_status(self, monitoring_service, options):
        """Show only health status."""
        health_status = monitoring_service.get_health_status()

        if options['format'] == 'json':
            self.stdout.write(json.dumps(health_status.to_dict(), indent=2, default=str))
        else:
            self._display_health_status(health_status)

    def _show_metrics_only(self, monitoring_service, options):
        """Show only metrics data."""
        all_metrics = monitoring_service.collect_all_metrics()

        # Filter by service if specified
        if options['service']:
            service_name = options['service']
            if service_name in all_metrics.get("services", {}):
                filtered_metrics = {
                    "timestamp": all_metrics["timestamp"],
                    "service": service_name,
                    "metrics": all_metrics["services"][service_name]
                }
            else:
                self.stdout.write(
                    self.style.ERROR(f"Service '{service_name}' not found. Available services: {list(all_metrics.get('services', {}).keys())}")
                )
                return
        else:
            filtered_metrics = all_metrics

        if options['format'] == 'json':
            self.stdout.write(json.dumps(filtered_metrics, indent=2, default=str))
        else:
            self._display_metrics(filtered_metrics, options['service'])

    def _display_full_dashboard(self, dashboard_data, options):
        """Display the full dashboard in text format."""
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.HTTP_INFO("SYSTEM PERFORMANCE DASHBOARD"))
        self.stdout.write("="*80)

        # Display health status
        health_status = dashboard_data["health_status"]
        self._display_health_status_from_dict(health_status)

        # Display service overview
        self.stdout.write("\n" + "-"*60)
        self.stdout.write(self.style.HTTP_INFO("SERVICE OVERVIEW"))
        self.stdout.write("-"*60)

        service_overview = dashboard_data.get("service_overview", {})
        self.stdout.write(f"Total Services: {service_overview.get('total_services', 0)}")
        self.stdout.write(f"Uptime: {self._format_duration(service_overview.get('uptime', 0))}")

        # Display performance trends
        trends = dashboard_data.get("performance_trends", {})
        if trends:
            self.stdout.write("\n" + "-"*60)
            self.stdout.write(self.style.HTTP_INFO("PERFORMANCE TRENDS"))
            self.stdout.write("-"*60)

            for metric_name, trend_data in trends.items():
                trend_symbol = {
                    "improving": "↗️",
                    "declining": "↘️",
                    "stable": "→"
                }.get(trend_data.get("trend", "stable"), "→")

                change = trend_data.get("change_percent", 0)
                self.stdout.write(f"{trend_symbol} {metric_name}: {change:+.1f}%")

        # Display recent issues
        issues = dashboard_data.get("performance_issues", [])
        if issues:
            self.stdout.write("\n" + "-"*60)
            self.stdout.write(self.style.WARNING("PERFORMANCE ISSUES"))
            self.stdout.write("-"*60)

            for issue in issues:
                severity_style = self.style.ERROR if issue.get("severity") == "critical" else self.style.WARNING
                self.stdout.write(
                    severity_style(f"[{issue.get('severity', 'unknown').upper()}] {issue.get('type', 'unknown')}: "
                                  f"{issue.get('service', 'unknown')} - {issue.get('value', 'N/A')}")
                )

        # Display recommendations
        recommendations = dashboard_data.get("recommendations", [])
        if recommendations:
            self.stdout.write("\n" + "-"*60)
            self.stdout.write(self.style.HTTP_INFO("RECOMMENDATIONS"))
            self.stdout.write("-"*60)

            for i, recommendation in enumerate(recommendations, 1):
                self.stdout.write(f"{i}. {recommendation}")

    def _display_health_status(self, health_status):
        """Display health status object."""
        self._display_health_status_from_dict(health_status.to_dict())

    def _display_health_status_from_dict(self, health_data):
        """Display health status from dictionary."""
        self.stdout.write("\n" + "-"*60)
        self.stdout.write(self.style.HTTP_INFO("SYSTEM HEALTH STATUS"))
        self.stdout.write("-"*60)

        # Overall health with color coding
        overall_health = health_data.get("overall_health", "unknown")
        health_style = {
            "healthy": self.style.SUCCESS,
            "degraded": self.style.WARNING,
            "unhealthy": self.style.ERROR
        }.get(overall_health, self.style.HTTP_INFO)

        self.stdout.write(f"Overall Health: {health_style(overall_health.upper())}")
        self.stdout.write(f"Total Requests: {health_data.get('total_requests', 0):,}")
        self.stdout.write(f"Error Rate: {health_data.get('error_rate', 0):.2f}%")
        self.stdout.write(f"Uptime: {self._format_duration(health_data.get('uptime', 0))}")
        self.stdout.write(f"Last Updated: {health_data.get('last_updated', 'Unknown')}")

        # Service status
        services_status = health_data.get("services_status", {})
        if services_status:
            self.stdout.write("\nService Status:")
            for service, status in services_status.items():
                status_style = {
                    "healthy": self.style.SUCCESS,
                    "degraded": self.style.WARNING,
                    "unhealthy": self.style.ERROR,
                    "not_available": self.style.HTTP_INFO
                }.get(status, self.style.HTTP_INFO)

                self.stdout.write(f"  • {service}: {status_style(status.upper())}")

        # Critical issues
        critical_issues = health_data.get("critical_issues", [])
        if critical_issues:
            self.stdout.write(self.style.ERROR("\nCritical Issues:"))
            for issue in critical_issues:
                self.stdout.write(f"  🚨 {issue}")

        # Warnings
        warnings = health_data.get("warnings", [])
        if warnings:
            self.stdout.write(self.style.WARNING("\nWarnings:"))
            for warning in warnings:
                self.stdout.write(f"  ⚠️  {warning}")

    def _display_metrics(self, metrics_data, service_filter=None):
        """Display metrics data in text format."""
        self.stdout.write("\n" + "-"*60)
        self.stdout.write(self.style.HTTP_INFO("CURRENT METRICS"))
        self.stdout.write("-"*60)

        timestamp = metrics_data.get("timestamp", "Unknown")
        self.stdout.write(f"Timestamp: {timestamp}")

        if service_filter:
            # Show single service metrics
            service_metrics = metrics_data.get("metrics", {})
            self.stdout.write(f"\nService: {service_filter}")
            self._display_service_metrics(service_metrics)
        else:
            # Show all services
            services = metrics_data.get("services", {})
            system_metrics = metrics_data.get("system", {})

            # Display system metrics first
            if system_metrics and "error" not in system_metrics:
                self.stdout.write(f"\nSystem Metrics:")
                self._display_service_metrics(system_metrics, indent="  ")

            # Display service metrics
            for service_name, service_metrics in services.items():
                self.stdout.write(f"\n{service_name}:")
                if "error" in service_metrics:
                    self.stdout.write(f"  Error: {service_metrics['error']}")
                else:
                    self._display_service_metrics(service_metrics, indent="  ")

    def _display_service_metrics(self, metrics, indent=""):
        """Display metrics for a single service."""
        for key, value in metrics.items():
            if key == "health_status":
                status_style = {
                    "healthy": self.style.SUCCESS,
                    "degraded": self.style.WARNING,
                    "unhealthy": self.style.ERROR
                }.get(value, self.style.HTTP_INFO)
                self.stdout.write(f"{indent}{key}: {status_style(str(value).upper())}")
            elif isinstance(value, (int, float)):
                if "percent" in key or "rate" in key:
                    self.stdout.write(f"{indent}{key}: {value:.2f}%")
                elif "time" in key:
                    self.stdout.write(f"{indent}{key}: {value:.3f}s")
                else:
                    self.stdout.write(f"{indent}{key}: {value:,}")
            elif isinstance(value, list):
                self.stdout.write(f"{indent}{key}: {len(value)} items")
            elif isinstance(value, dict):
                if len(value) <= 5:  # Only show small dicts
                    self.stdout.write(f"{indent}{key}: {value}")
                else:
                    self.stdout.write(f"{indent}{key}: {len(value)} items")
            else:
                self.stdout.write(f"{indent}{key}: {value}")

    def _format_duration(self, seconds):
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"