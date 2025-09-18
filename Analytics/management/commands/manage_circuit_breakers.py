"""
Management command for circuit breaker operations.

Usage:
    python manage.py manage_circuit_breakers --status
    python manage.py manage_circuit_breakers --open multilingual fr
    python manage.py manage_circuit_breakers --close multilingual fr
    python manage.py manage_circuit_breakers --reset multilingual
    python manage.py manage_circuit_breakers --stats --json
"""

import json
from django.core.management.base import BaseCommand, CommandError

from Analytics.services.circuit_breaker import (
    get_circuit_breaker,
    get_all_circuit_breaker_stats,
    CircuitState
)


class Command(BaseCommand):
    help = 'Manage circuit breakers for multilingual LLM system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--status',
            action='store_true',
            help='Show status of all circuit breakers',
        )
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Show detailed statistics for all circuit breakers',
        )
        parser.add_argument(
            '--open',
            type=str,
            nargs=2,
            metavar=('BREAKER_NAME', 'LANGUAGE'),
            help='Manually open a circuit breaker for a specific language',
        )
        parser.add_argument(
            '--close',
            type=str,
            nargs=2,
            metavar=('BREAKER_NAME', 'LANGUAGE'),
            help='Manually close a circuit breaker for a specific language',
        )
        parser.add_argument(
            '--reset',
            type=str,
            help='Reset all circuits for a specific breaker',
        )
        parser.add_argument(
            '--reset-all',
            action='store_true',
            help='Reset all circuit breakers',
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results in JSON format',
        )

    def handle(self, *args, **options):
        """Execute circuit breaker management command."""
        try:
            if options['status']:
                self.show_status(options)
            elif options['stats']:
                self.show_stats(options)
            elif options['open']:
                breaker_name, language = options['open']
                self.open_circuit(breaker_name, language, options)
            elif options['close']:
                breaker_name, language = options['close']
                self.close_circuit(breaker_name, language, options)
            elif options['reset']:
                self.reset_breaker(options['reset'], options)
            elif options['reset_all']:
                self.reset_all_breakers(options)
            else:
                self.print_help('manage.py', 'manage_circuit_breakers')

        except Exception as e:
            raise CommandError(f'Circuit breaker operation failed: {str(e)}')

    def show_status(self, options):
        """Show status of all circuit breakers."""
        try:
            circuit_stats = get_all_circuit_breaker_stats()

            if options['json']:
                self.stdout.write(json.dumps(circuit_stats, indent=2))
                return

            self.stdout.write(self.style.SUCCESS('Circuit Breaker Status\n'))
            self.stdout.write(f'Timestamp: {circuit_stats["timestamp"]}')

            if 'circuit_breakers' not in circuit_stats:
                self.stdout.write('No circuit breakers found.')
                return

            for breaker_name, breaker_info in circuit_stats['circuit_breakers'].items():
                self.stdout.write(f'\n{self.style.HTTP_INFO(f"Breaker: {breaker_name}")}')

                if 'languages' in breaker_info:
                    for lang, lang_stats in breaker_info['languages'].items():
                        state = lang_stats.get('state', 'unknown')
                        failure_count = lang_stats.get('failure_count', 0)
                        success_count = lang_stats.get('success_count', 0)

                        # Choose color based on state
                        if state == 'closed':
                            color_style = self.style.SUCCESS
                            status_symbol = '✓'
                        elif state == 'half_open':
                            color_style = self.style.WARNING
                            status_symbol = '⚠'
                        else:  # open
                            color_style = self.style.ERROR
                            status_symbol = '✗'

                        self.stdout.write(
                            f'  {status_symbol} {lang}: {color_style(state.upper())} '
                            f'(failures: {failure_count}, successes: {success_count})'
                        )

                        # Show additional info for non-closed circuits
                        if state != 'closed':
                            last_failure = lang_stats.get('last_failure')
                            if last_failure:
                                self.stdout.write(f'      Last failure: {last_failure}')

                            time_since_failure = lang_stats.get('time_since_last_failure')
                            if time_since_failure is not None:
                                self.stdout.write(f'      Time since last failure: {time_since_failure:.1f}s')

                            can_recover = lang_stats.get('can_recover', False)
                            if can_recover:
                                self.stdout.write(f'      {self.style.SUCCESS("Can recover now")}')

        except Exception as e:
            raise CommandError(f'Failed to get circuit breaker status: {str(e)}')

    def show_stats(self, options):
        """Show detailed statistics for all circuit breakers."""
        try:
            circuit_stats = get_all_circuit_breaker_stats()

            if options['json']:
                self.stdout.write(json.dumps(circuit_stats, indent=2))
                return

            self.stdout.write(self.style.SUCCESS('Circuit Breaker Detailed Statistics\n'))

            if 'circuit_breakers' not in circuit_stats:
                self.stdout.write('No circuit breakers found.')
                return

            for breaker_name, breaker_info in circuit_stats['circuit_breakers'].items():
                self.stdout.write(f'\n{self.style.HTTP_INFO(f"=== {breaker_name.upper()} CIRCUIT BREAKER ===")}')

                # Show configuration
                if 'config' in breaker_info:
                    config = breaker_info['config']
                    self.stdout.write('\nConfiguration:')
                    self.stdout.write(f'  Failure Threshold: {config.get("failure_threshold", "N/A")}')
                    self.stdout.write(f'  Recovery Timeout: {config.get("recovery_timeout", "N/A")}s')
                    self.stdout.write(f'  Success Threshold: {config.get("success_threshold", "N/A")}')
                    self.stdout.write(f'  Half-Open Max Calls: {config.get("half_open_max_calls", "N/A")}')

                # Show language statistics
                if 'languages' in breaker_info:
                    self.stdout.write('\nLanguage Statistics:')
                    for lang, lang_stats in breaker_info['languages'].items():
                        self.stdout.write(f'\n  {lang.upper()}:')
                        self.stdout.write(f'    State: {lang_stats.get("state", "unknown").upper()}')
                        self.stdout.write(f'    Failure Count: {lang_stats.get("failure_count", 0)}')
                        self.stdout.write(f'    Success Count: {lang_stats.get("success_count", 0)}')
                        self.stdout.write(f'    Half-Open Calls: {lang_stats.get("half_open_calls", 0)}')

                        last_failure = lang_stats.get('last_failure')
                        if last_failure:
                            self.stdout.write(f'    Last Failure: {last_failure}')

                        time_since_failure = lang_stats.get('time_since_last_failure')
                        if time_since_failure is not None:
                            self.stdout.write(f'    Time Since Last Failure: {time_since_failure:.1f}s')

                        can_recover = lang_stats.get('can_recover', False)
                        recovery_status = 'Yes' if can_recover else 'No'
                        self.stdout.write(f'    Can Recover: {recovery_status}')

        except Exception as e:
            raise CommandError(f'Failed to get circuit breaker statistics: {str(e)}')

    def open_circuit(self, breaker_name, language, options):
        """Manually open a circuit breaker for a specific language."""
        try:
            breaker = get_circuit_breaker(breaker_name)
            reason = f"Manual override via management command"

            breaker.force_open(language, reason)

            message = f'Successfully opened circuit breaker for {breaker_name}:{language}'
            if options['json']:
                self.stdout.write(json.dumps({
                    'status': 'success',
                    'message': message,
                    'breaker': breaker_name,
                    'language': language,
                    'action': 'open'
                }))
            else:
                self.stdout.write(self.style.SUCCESS(message))

        except Exception as e:
            raise CommandError(f'Failed to open circuit breaker: {str(e)}')

    def close_circuit(self, breaker_name, language, options):
        """Manually close a circuit breaker for a specific language."""
        try:
            breaker = get_circuit_breaker(breaker_name)
            reason = f"Manual override via management command"

            breaker.force_close(language, reason)

            message = f'Successfully closed circuit breaker for {breaker_name}:{language}'
            if options['json']:
                self.stdout.write(json.dumps({
                    'status': 'success',
                    'message': message,
                    'breaker': breaker_name,
                    'language': language,
                    'action': 'close'
                }))
            else:
                self.stdout.write(self.style.SUCCESS(message))

        except Exception as e:
            raise CommandError(f'Failed to close circuit breaker: {str(e)}')

    def reset_breaker(self, breaker_name, options):
        """Reset all circuits for a specific breaker."""
        try:
            breaker = get_circuit_breaker(breaker_name)
            breaker.reset_all()

            message = f'Successfully reset all circuits for breaker: {breaker_name}'
            if options['json']:
                self.stdout.write(json.dumps({
                    'status': 'success',
                    'message': message,
                    'breaker': breaker_name,
                    'action': 'reset'
                }))
            else:
                self.stdout.write(self.style.SUCCESS(message))

        except Exception as e:
            raise CommandError(f'Failed to reset circuit breaker: {str(e)}')

    def reset_all_breakers(self, options):
        """Reset all circuit breakers."""
        if not options.get('json'):
            self.stdout.write(self.style.WARNING('⚠️  WARNING: This will reset ALL circuit breakers!'))
            confirm = input('Are you sure you want to proceed? (yes/no): ')
            if confirm.lower() != 'yes':
                self.stdout.write('Operation cancelled.')
                return

        try:
            # Get all circuit breakers and reset them
            circuit_stats = get_all_circuit_breaker_stats()
            reset_count = 0

            if 'circuit_breakers' in circuit_stats:
                for breaker_name in circuit_stats['circuit_breakers'].keys():
                    breaker = get_circuit_breaker(breaker_name)
                    breaker.reset_all()
                    reset_count += 1

            message = f'Successfully reset {reset_count} circuit breakers'
            if options['json']:
                self.stdout.write(json.dumps({
                    'status': 'success',
                    'message': message,
                    'reset_count': reset_count,
                    'action': 'reset_all'
                }))
            else:
                self.stdout.write(self.style.SUCCESS(message))

        except Exception as e:
            raise CommandError(f'Failed to reset all circuit breakers: {str(e)}')