"""
Management command to start production monitoring service.

Usage:
    python manage.py start_production_monitoring
    python manage.py start_production_monitoring --daemon
    python manage.py start_production_monitoring --status
"""

import json
import signal
import sys
import time
from django.core.management.base import BaseCommand, CommandError

from Analytics.services.production_monitoring import (
    get_monitoring_service,
    start_production_monitoring,
    stop_production_monitoring
)


class Command(BaseCommand):
    help = 'Start and manage production monitoring service'

    def add_arguments(self, parser):
        parser.add_argument(
            '--daemon',
            action='store_true',
            help='Run monitoring service in daemon mode (keeps running until stopped)',
        )
        parser.add_argument(
            '--status',
            action='store_true',
            help='Show current monitoring service status',
        )
        parser.add_argument(
            '--stop',
            action='store_true',
            help='Stop the monitoring service',
        )
        parser.add_argument(
            '--restart',
            action='store_true',
            help='Restart the monitoring service',
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results in JSON format',
        )
        parser.add_argument(
            '--alerts',
            type=int,
            default=0,
            help='Show recent alerts (specify number of alerts to show)',
        )

    def handle(self, *args, **options):
        """Execute monitoring management command."""
        try:
            if options['status']:
                self.show_status(options)
            elif options['stop']:
                self.stop_monitoring(options)
            elif options['restart']:
                self.restart_monitoring(options)
            elif options['alerts']:
                self.show_alerts(options['alerts'], options)
            else:
                self.start_monitoring(options)

        except KeyboardInterrupt:
            self.stdout.write('\nShutting down monitoring service...')
            stop_production_monitoring()
            sys.exit(0)
        except Exception as e:
            raise CommandError(f'Monitoring operation failed: {str(e)}')

    def start_monitoring(self, options):
        """Start the production monitoring service."""
        try:
            monitoring_service = get_monitoring_service()

            # Check if already running
            status = monitoring_service.get_monitoring_status()
            if status.get('thread_running', False):
                message = 'Production monitoring is already running'
                if options['json']:
                    self.stdout.write(json.dumps({
                        'status': 'already_running',
                        'message': message
                    }))
                else:
                    self.stdout.write(self.style.WARNING(message))
                return

            # Start monitoring
            start_production_monitoring()

            # Verify it started
            time.sleep(1)  # Give it a moment to start
            status = monitoring_service.get_monitoring_status()

            if status.get('thread_running', False):
                message = 'Production monitoring started successfully'
                if options['json']:
                    self.stdout.write(json.dumps({
                        'status': 'started',
                        'message': message,
                        'monitoring_status': status
                    }))
                else:
                    self.stdout.write(self.style.SUCCESS(message))
                    self.stdout.write(f'Check interval: {status.get("check_interval", "unknown")} seconds')
                    self.stdout.write(f'Monitoring enabled: {status.get("monitoring_enabled", False)}')

                    # Show notification channels
                    notification_channels = status.get('notification_channels', {})
                    if notification_channels:
                        self.stdout.write('\nNotification channels:')
                        for channel, enabled in notification_channels.items():
                            status_text = 'enabled' if enabled else 'disabled'
                            self.stdout.write(f'  {channel}: {status_text}')

                if options['daemon']:
                    self.run_daemon_mode()
            else:
                raise CommandError('Failed to start monitoring service')

        except Exception as e:
            raise CommandError(f'Failed to start monitoring: {str(e)}')

    def stop_monitoring(self, options):
        """Stop the production monitoring service."""
        try:
            monitoring_service = get_monitoring_service()

            # Check if running
            status = monitoring_service.get_monitoring_status()
            if not status.get('thread_running', False):
                message = 'Production monitoring is not running'
                if options['json']:
                    self.stdout.write(json.dumps({
                        'status': 'not_running',
                        'message': message
                    }))
                else:
                    self.stdout.write(self.style.WARNING(message))
                return

            # Stop monitoring
            stop_production_monitoring()

            # Verify it stopped
            time.sleep(1)
            status = monitoring_service.get_monitoring_status()

            if not status.get('thread_running', False):
                message = 'Production monitoring stopped successfully'
                if options['json']:
                    self.stdout.write(json.dumps({
                        'status': 'stopped',
                        'message': message
                    }))
                else:
                    self.stdout.write(self.style.SUCCESS(message))
            else:
                raise CommandError('Failed to stop monitoring service')

        except Exception as e:
            raise CommandError(f'Failed to stop monitoring: {str(e)}')

    def restart_monitoring(self, options):
        """Restart the production monitoring service."""
        self.stdout.write('Restarting production monitoring...')

        # Stop first
        self.stop_monitoring({'json': False})
        time.sleep(2)  # Wait a bit between stop and start

        # Then start
        self.start_monitoring(options)

    def show_status(self, options):
        """Show current monitoring service status."""
        try:
            monitoring_service = get_monitoring_service()
            status = monitoring_service.get_monitoring_status()

            if options['json']:
                self.stdout.write(json.dumps(status, indent=2))
                return

            self.stdout.write(self.style.SUCCESS('Production Monitoring Status\n'))

            # Basic status
            running_status = status.get('thread_running', False)
            status_color = self.style.SUCCESS if running_status else self.style.ERROR
            status_symbol = '✓' if running_status else '✗'

            self.stdout.write(f'{status_symbol} Service Running: {status_color(str(running_status).upper())}')
            self.stdout.write(f'Monitoring Enabled: {status.get("monitoring_enabled", False)}')
            self.stdout.write(f'Check Interval: {status.get("check_interval", "unknown")} seconds')
            self.stdout.write(f'Last Check: {status.get("last_check", "unknown")}')

            # Thresholds
            thresholds = status.get('thresholds', {})
            if thresholds:
                self.stdout.write('\nAlert Thresholds:')
                self.stdout.write(f'  CPU: {thresholds.get("cpu_threshold", "unknown")}%')
                self.stdout.write(f'  Memory: {thresholds.get("memory_threshold", "unknown")}%')
                self.stdout.write(f'  Disk: {thresholds.get("disk_threshold", "unknown")}%')
                self.stdout.write(f'  Response Time: {thresholds.get("response_time_threshold", "unknown")}s')
                self.stdout.write(f'  Error Rate: {thresholds.get("error_rate_threshold", "unknown")}')

            # Notification channels
            notification_channels = status.get('notification_channels', {})
            if notification_channels:
                self.stdout.write('\nNotification Channels:')
                for channel, enabled in notification_channels.items():
                    status_text = self.style.SUCCESS('enabled') if enabled else self.style.ERROR('disabled')
                    self.stdout.write(f'  {channel}: {status_text}')

            # Recent alerts summary
            recent_alerts = monitoring_service.get_recent_alerts(5)
            if recent_alerts:
                self.stdout.write(f'\nRecent Alerts ({len(recent_alerts)} of last 5):')
                for alert in recent_alerts:
                    level = alert.get('level', 'unknown')
                    title = alert.get('title', 'Unknown alert')
                    timestamp = alert.get('timestamp', 'Unknown time')

                    level_color = {
                        'critical': self.style.ERROR,
                        'error': self.style.ERROR,
                        'warning': self.style.WARNING,
                        'info': self.style.SUCCESS
                    }.get(level, self.style.HTTP_INFO)

                    self.stdout.write(f'  • {level_color(level.upper())}: {title} ({timestamp})')

        except Exception as e:
            raise CommandError(f'Failed to get monitoring status: {str(e)}')

    def show_alerts(self, count, options):
        """Show recent alerts."""
        try:
            monitoring_service = get_monitoring_service()
            recent_alerts = monitoring_service.get_recent_alerts(count)

            if options['json']:
                self.stdout.write(json.dumps({
                    'alerts': recent_alerts,
                    'count': len(recent_alerts),
                    'requested_count': count
                }, indent=2))
                return

            self.stdout.write(self.style.SUCCESS(f'Recent Alerts (showing {len(recent_alerts)} of {count} requested)\n'))

            if not recent_alerts:
                self.stdout.write('No recent alerts found.')
                return

            for i, alert in enumerate(recent_alerts, 1):
                level = alert.get('level', 'unknown')
                title = alert.get('title', 'Unknown alert')
                message = alert.get('message', 'No message')
                timestamp = alert.get('timestamp', 'Unknown time')

                level_color = {
                    'critical': self.style.ERROR,
                    'error': self.style.ERROR,
                    'warning': self.style.WARNING,
                    'info': self.style.SUCCESS
                }.get(level, self.style.HTTP_INFO)

                self.stdout.write(f'{i}. {level_color(level.upper())}: {title}')
                self.stdout.write(f'   Time: {timestamp}')
                self.stdout.write(f'   Message: {message}')

                # Show context if available
                context = alert.get('context', {})
                if context:
                    self.stdout.write('   Context:')
                    for key, value in context.items():
                        self.stdout.write(f'     {key}: {value}')

                self.stdout.write()  # Empty line between alerts

        except Exception as e:
            raise CommandError(f'Failed to get alerts: {str(e)}')

    def run_daemon_mode(self):
        """Run monitoring in daemon mode."""
        self.stdout.write(self.style.SUCCESS('Running in daemon mode. Press Ctrl+C to stop.'))

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.stdout.write('\nReceived shutdown signal. Stopping monitoring...')
            stop_production_monitoring()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Keep the process alive
            while True:
                time.sleep(60)  # Check every minute if we should continue
                monitoring_service = get_monitoring_service()
                status = monitoring_service.get_monitoring_status()

                if not status.get('thread_running', False):
                    self.stdout.write(self.style.ERROR('Monitoring thread stopped unexpectedly. Exiting.'))
                    sys.exit(1)

        except KeyboardInterrupt:
            self.stdout.write('\nShutting down monitoring service...')
            stop_production_monitoring()
            sys.exit(0)