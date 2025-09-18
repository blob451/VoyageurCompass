"""
Management command for comprehensive multilingual system health checks.

Usage:
    python manage.py multilingual_health_check
    python manage.py multilingual_health_check --verbose
    python manage.py multilingual_health_check --json
"""

import json
import time
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from django.core.cache import cache

from Analytics.services.feature_flags import get_feature_flags
from Analytics.services.circuit_breaker import get_all_circuit_breaker_stats
from Analytics.services.multilingual_metrics import get_multilingual_metrics
from Analytics.services.production_monitoring import get_monitoring_service


class Command(BaseCommand):
    help = 'Perform comprehensive health check of multilingual LLM system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output with detailed information',
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results in JSON format',
        )
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Attempt to fix common issues automatically',
        )

    def handle(self, *args, **options):
        """Execute health check command."""
        start_time = time.time()

        try:
            health_results = self.perform_comprehensive_health_check(
                verbose=options['verbose'],
                fix_issues=options['fix']
            )

            health_results['check_duration'] = round(time.time() - start_time, 2)

            if options['json']:
                self.stdout.write(json.dumps(health_results, indent=2))
            else:
                self.display_health_results(health_results, options['verbose'])

        except Exception as e:
            raise CommandError(f'Health check failed: {str(e)}')

    def perform_comprehensive_health_check(self, verbose=False, fix_issues=False):
        """Perform comprehensive health check of all system components."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'components': {},
            'issues_found': [],
            'fixes_applied': [],
            'recommendations': []
        }

        # Check each component
        components_to_check = [
            ('feature_flags', self.check_feature_flags),
            ('circuit_breakers', self.check_circuit_breakers),
            ('cache_system', self.check_cache_system),
            ('multilingual_metrics', self.check_multilingual_metrics),
            ('production_monitoring', self.check_production_monitoring),
        ]

        all_healthy = True

        for component_name, check_function in components_to_check:
            if verbose:
                self.stdout.write(f'Checking {component_name}...', ending='')

            try:
                component_result = check_function(fix_issues)
                results['components'][component_name] = component_result

                if component_result['status'] != 'healthy':
                    all_healthy = False
                    results['issues_found'].extend(component_result.get('issues', []))

                if fix_issues:
                    results['fixes_applied'].extend(component_result.get('fixes_applied', []))

                results['recommendations'].extend(component_result.get('recommendations', []))

                if verbose:
                    status_symbol = '✓' if component_result['status'] == 'healthy' else '✗'
                    self.stdout.write(f' {status_symbol} {component_result["status"]}')

            except Exception as e:
                all_healthy = False
                error_result = {
                    'status': 'error',
                    'error': str(e),
                    'issues': [f'Component check failed: {str(e)}']
                }
                results['components'][component_name] = error_result
                results['issues_found'].append(f'{component_name}: {str(e)}')

                if verbose:
                    self.stdout.write(f' ✗ ERROR: {str(e)}')

        # Determine overall status
        results['overall_status'] = 'healthy' if all_healthy else 'degraded'

        return results

    def check_feature_flags(self, fix_issues=False):
        """Check feature flags system health."""
        result = {
            'status': 'healthy',
            'details': {},
            'issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

        try:
            feature_flags = get_feature_flags()
            flags_status = feature_flags.get_all_flags_status()

            # Check for emergency fallback
            emergency_status = flags_status.get('emergency_status', {})
            if emergency_status.get('emergency_fallback_enabled', False):
                result['status'] = 'degraded'
                result['issues'].append('Emergency fallback is enabled')
                result['recommendations'].append('Investigate why emergency fallback was triggered')

            # Check core flags
            flags = flags_status.get('flags', {})
            core_flags = ['multilingual_llm_enabled', 'french_generation_enabled', 'spanish_generation_enabled']

            disabled_core_flags = []
            for flag_name in core_flags:
                if flag_name in flags and not flags[flag_name].get('enabled', True):
                    disabled_core_flags.append(flag_name)

            if disabled_core_flags:
                result['status'] = 'degraded'
                result['issues'].extend([f'Core flag disabled: {flag}' for flag in disabled_core_flags])

                if fix_issues:
                    # Attempt to re-enable core flags (if not in emergency mode)
                    if not emergency_status.get('emergency_fallback_enabled', False):
                        for flag_name in disabled_core_flags:
                            feature_flags.set_flag(flag_name, True)
                            result['fixes_applied'].append(f'Re-enabled flag: {flag_name}')

            result['details'] = {
                'flags_count': len(flags),
                'emergency_fallback': emergency_status.get('emergency_fallback_enabled', False),
                'disabled_core_flags': disabled_core_flags
            }

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Feature flags check failed: {str(e)}')

        return result

    def check_circuit_breakers(self, fix_issues=False):
        """Check circuit breaker system health."""
        result = {
            'status': 'healthy',
            'details': {},
            'issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

        try:
            circuit_stats = get_all_circuit_breaker_stats()

            if 'circuit_breakers' in circuit_stats:
                open_circuits = []
                half_open_circuits = []

                for breaker_name, breaker_info in circuit_stats['circuit_breakers'].items():
                    if 'languages' in breaker_info:
                        for lang, lang_stats in breaker_info['languages'].items():
                            state = lang_stats.get('state', 'unknown')

                            if state == 'open':
                                open_circuits.append(f'{breaker_name}:{lang}')
                            elif state == 'half_open':
                                half_open_circuits.append(f'{breaker_name}:{lang}')

                if open_circuits:
                    result['status'] = 'degraded'
                    result['issues'].extend([f'Circuit breaker OPEN: {circuit}' for circuit in open_circuits])
                    result['recommendations'].append('Check service health for languages with open circuits')

                if half_open_circuits:
                    result['issues'].extend([f'Circuit breaker HALF-OPEN: {circuit}' for circuit in half_open_circuits])
                    result['recommendations'].append('Monitor half-open circuits for recovery')

                result['details'] = {
                    'open_circuits': open_circuits,
                    'half_open_circuits': half_open_circuits,
                    'total_circuits': len(circuit_stats.get('circuit_breakers', {}))
                }

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Circuit breaker check failed: {str(e)}')

        return result

    def check_cache_system(self, fix_issues=False):
        """Check cache system health."""
        result = {
            'status': 'healthy',
            'details': {},
            'issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

        try:
            # Test basic cache operations
            test_key = 'health_check_test'
            test_value = f'test_{int(time.time())}'

            cache.set(test_key, test_value, 30)
            retrieved_value = cache.get(test_key)

            if retrieved_value != test_value:
                result['status'] = 'error'
                result['issues'].append('Cache read/write test failed')
            else:
                cache.delete(test_key)

            # Test cache performance (basic timing)
            start_time = time.time()
            for i in range(10):
                cache.set(f'perf_test_{i}', f'value_{i}', 30)
                cache.get(f'perf_test_{i}')
                cache.delete(f'perf_test_{i}')

            cache_performance = time.time() - start_time

            if cache_performance > 1.0:  # If 10 operations take more than 1 second
                result['status'] = 'degraded'
                result['issues'].append(f'Cache performance degraded: {cache_performance:.2f}s for 10 operations')
                result['recommendations'].append('Check cache server performance and network connectivity')

            result['details'] = {
                'cache_test_passed': retrieved_value == test_value,
                'performance_seconds': round(cache_performance, 3)
            }

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Cache system check failed: {str(e)}')

        return result

    def check_multilingual_metrics(self, fix_issues=False):
        """Check multilingual metrics system health."""
        result = {
            'status': 'healthy',
            'details': {},
            'issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

        try:
            metrics_service = get_multilingual_metrics()

            # Get current metrics
            real_time_metrics = metrics_service.get_real_time_dashboard()

            if 'current_rates' in real_time_metrics:
                rates = real_time_metrics['current_rates']

                # Check error rate
                error_rate = rates.get('error_rate', 0)
                if error_rate > 0.1:  # 10% error rate
                    result['status'] = 'degraded'
                    result['issues'].append(f'High error rate: {error_rate:.2%}')
                    result['recommendations'].append('Investigate causes of high error rate')

                # Check response time
                avg_response_time = rates.get('avg_response_time', 0)
                if avg_response_time > 10:  # 10 seconds
                    result['status'] = 'degraded'
                    result['issues'].append(f'High response time: {avg_response_time:.2f}s')
                    result['recommendations'].append('Check system resources and optimize performance')

                # Check quality score
                quality_score = rates.get('quality_score', 1.0)
                if quality_score < 0.7:
                    result['status'] = 'degraded'
                    result['issues'].append(f'Low quality score: {quality_score:.2f}')
                    result['recommendations'].append('Review and improve model quality')

                result['details'] = {
                    'error_rate': error_rate,
                    'avg_response_time': avg_response_time,
                    'quality_score': quality_score,
                    'requests_per_minute': rates.get('requests_per_minute', 0)
                }

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Multilingual metrics check failed: {str(e)}')

        return result

    def check_production_monitoring(self, fix_issues=False):
        """Check production monitoring system health."""
        result = {
            'status': 'healthy',
            'details': {},
            'issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

        try:
            monitoring_service = get_monitoring_service()
            monitoring_status = monitoring_service.get_monitoring_status()

            # Check if monitoring is enabled and running
            if not monitoring_status.get('monitoring_enabled', False):
                result['status'] = 'degraded'
                result['issues'].append('Production monitoring is disabled')
                result['recommendations'].append('Enable production monitoring for better system observability')

            if not monitoring_status.get('thread_running', False):
                result['status'] = 'degraded'
                result['issues'].append('Monitoring background thread is not running')

                if fix_issues:
                    try:
                        monitoring_service.start_monitoring()
                        result['fixes_applied'].append('Started monitoring background thread')
                    except Exception as e:
                        result['issues'].append(f'Failed to start monitoring: {str(e)}')

            # Check recent alerts
            recent_alerts = monitoring_service.get_recent_alerts(10)
            critical_alerts = [alert for alert in recent_alerts if alert.get('level') == 'critical']

            if critical_alerts:
                result['status'] = 'degraded'
                result['issues'].append(f'{len(critical_alerts)} critical alerts in recent history')
                result['recommendations'].append('Review and address critical alerts')

            result['details'] = {
                'monitoring_enabled': monitoring_status.get('monitoring_enabled', False),
                'thread_running': monitoring_status.get('thread_running', False),
                'recent_alerts_count': len(recent_alerts),
                'critical_alerts_count': len(critical_alerts)
            }

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Production monitoring check failed: {str(e)}')

        return result

    def display_health_results(self, results, verbose=False):
        """Display health check results in human-readable format."""
        overall_status = results['overall_status']
        status_symbol = '✓' if overall_status == 'healthy' else '✗'

        self.stdout.write(
            self.style.SUCCESS(f'\n{status_symbol} Overall System Status: {overall_status.upper()}')
            if overall_status == 'healthy'
            else self.style.ERROR(f'\n{status_symbol} Overall System Status: {overall_status.upper()}')
        )

        self.stdout.write(f'Check Duration: {results["check_duration"]}s')
        self.stdout.write(f'Timestamp: {results["timestamp"]}\n')

        # Display component status
        self.stdout.write('Component Status:')
        for component, details in results['components'].items():
            status = details['status']
            symbol = '✓' if status == 'healthy' else '✗'

            color_style = self.style.SUCCESS if status == 'healthy' else self.style.ERROR
            self.stdout.write(f'  {symbol} {component}: {color_style(status.upper())}')

            if verbose and 'details' in details:
                for key, value in details['details'].items():
                    self.stdout.write(f'      {key}: {value}')

        # Display issues
        if results['issues_found']:
            self.stdout.write(f'\n{self.style.ERROR("Issues Found:")}')
            for issue in results['issues_found']:
                self.stdout.write(f'  • {issue}')

        # Display fixes applied
        if results['fixes_applied']:
            self.stdout.write(f'\n{self.style.SUCCESS("Fixes Applied:")}')
            for fix in results['fixes_applied']:
                self.stdout.write(f'  • {fix}')

        # Display recommendations
        if results['recommendations']:
            self.stdout.write(f'\n{self.style.WARNING("Recommendations:")}')
            for recommendation in results['recommendations']:
                self.stdout.write(f'  • {recommendation}')

        self.stdout.write()  # Empty line at end