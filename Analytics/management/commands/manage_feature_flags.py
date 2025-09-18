"""
Management command for feature flags operations.

Usage:
    python manage.py manage_feature_flags --list
    python manage.py manage_feature_flags --enable multilingual_llm_enabled
    python manage.py manage_feature_flags --disable french_generation_enabled
    python manage.py manage_feature_flags --rollout french_generation_enabled 50
    python manage.py manage_feature_flags --emergency-disable
    python manage.py manage_feature_flags --clear-cache
"""

import json
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User

from Analytics.services.feature_flags import (
    get_feature_flags,
    MultilingualFeatureFlags,
    emergency_disable_multilingual
)


class Command(BaseCommand):
    help = 'Manage feature flags for multilingual LLM system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--list',
            action='store_true',
            help='List all feature flags and their current status',
        )
        parser.add_argument(
            '--enable',
            type=str,
            help='Enable a specific feature flag',
        )
        parser.add_argument(
            '--disable',
            type=str,
            help='Disable a specific feature flag',
        )
        parser.add_argument(
            '--rollout',
            type=str,
            nargs=2,
            metavar=('FLAG_NAME', 'PERCENTAGE'),
            help='Set rollout percentage for a feature flag (0-100)',
        )
        parser.add_argument(
            '--emergency-disable',
            action='store_true',
            help='Emergency disable all multilingual features',
        )
        parser.add_argument(
            '--clear-cache',
            action='store_true',
            help='Clear feature flags cache',
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results in JSON format',
        )
        parser.add_argument(
            '--user',
            type=str,
            help='Username for user-specific flag checks',
        )

    def handle(self, *args, **options):
        """Execute feature flags management command."""
        try:
            feature_flags = get_feature_flags()

            if options['list']:
                self.list_feature_flags(feature_flags, options)
            elif options['enable']:
                self.enable_flag(feature_flags, options['enable'], options)
            elif options['disable']:
                self.disable_flag(feature_flags, options['disable'], options)
            elif options['rollout']:
                flag_name, percentage = options['rollout']
                self.set_rollout(feature_flags, flag_name, int(percentage), options)
            elif options['emergency_disable']:
                self.emergency_disable_all(options)
            elif options['clear_cache']:
                self.clear_cache(feature_flags, options)
            else:
                self.print_help('manage.py', 'manage_feature_flags')

        except Exception as e:
            raise CommandError(f'Feature flags operation failed: {str(e)}')

    def list_feature_flags(self, feature_flags, options):
        """List all feature flags and their status."""
        user = self.get_user(options.get('user')) if options.get('user') else None
        flags_status = feature_flags.get_all_flags_status(user)

        if options['json']:
            self.stdout.write(json.dumps(flags_status, indent=2))
            return

        self.stdout.write(self.style.SUCCESS('Feature Flags Status\n'))
        self.stdout.write(f'Timestamp: {flags_status["timestamp"]}')

        # Display emergency status
        emergency_status = flags_status.get('emergency_status', {})
        if emergency_status.get('emergency_fallback_enabled', False):
            self.stdout.write(self.style.ERROR('⚠️  EMERGENCY FALLBACK ENABLED'))

        self.stdout.write('\nCore Flags:')
        flags = flags_status.get('flags', {})

        # Group flags by category
        core_flags = [
            'multilingual_llm_enabled',
            'direct_generation_enabled',
            'translation_pipeline_enabled'
        ]

        language_flags = [
            'french_generation_enabled',
            'spanish_generation_enabled'
        ]

        advanced_flags = [
            'quality_enhancement_enabled',
            'adaptive_caching_enabled',
            'performance_monitoring_enabled',
            'parallel_processing_enabled'
        ]

        emergency_flags = [
            'emergency_fallback_enabled',
            'circuit_breaker_enabled'
        ]

        self.display_flag_group('Core Features', core_flags, flags, user)
        self.display_flag_group('Language Support', language_flags, flags, user)
        self.display_flag_group('Advanced Features', advanced_flags, flags, user)
        self.display_flag_group('Emergency Controls', emergency_flags, flags, user)

        # Display rollout percentages
        rollout_percentages = flags_status.get('rollout_percentages', {})
        if rollout_percentages:
            self.stdout.write('\nRollout Percentages:')
            for flag_name, percentage in rollout_percentages.items():
                self.stdout.write(f'  {flag_name}: {percentage}%')

    def display_flag_group(self, group_name, flag_names, flags_data, user):
        """Display a group of flags."""
        self.stdout.write(f'\n{group_name}:')

        for flag_name in flag_names:
            if flag_name in flags_data:
                flag_info = flags_data[flag_name]
                enabled = flag_info.get('enabled', False)
                user_enabled = flag_info.get('user_enabled')

                status_symbol = '✓' if enabled else '✗'
                color_style = self.style.SUCCESS if enabled else self.style.ERROR

                flag_display = f'  {status_symbol} {flag_name}: {color_style(str(enabled).upper())}'

                # Show user-specific status if different
                if user and user_enabled is not None and user_enabled != enabled:
                    user_symbol = '✓' if user_enabled else '✗'
                    user_color = self.style.SUCCESS if user_enabled else self.style.ERROR
                    flag_display += f' (user: {user_color(str(user_enabled).upper())})'

                self.stdout.write(flag_display)

    def enable_flag(self, feature_flags, flag_name, options):
        """Enable a specific feature flag."""
        if not self.validate_flag_name(flag_name):
            raise CommandError(f'Invalid flag name: {flag_name}')

        success = feature_flags.set_flag(flag_name, True)

        if success:
            message = f'Successfully enabled flag: {flag_name}'
            if options['json']:
                self.stdout.write(json.dumps({'status': 'success', 'message': message}))
            else:
                self.stdout.write(self.style.SUCCESS(message))
        else:
            raise CommandError(f'Failed to enable flag: {flag_name}')

    def disable_flag(self, feature_flags, flag_name, options):
        """Disable a specific feature flag."""
        if not self.validate_flag_name(flag_name):
            raise CommandError(f'Invalid flag name: {flag_name}')

        success = feature_flags.set_flag(flag_name, False)

        if success:
            message = f'Successfully disabled flag: {flag_name}'
            if options['json']:
                self.stdout.write(json.dumps({'status': 'success', 'message': message}))
            else:
                self.stdout.write(self.style.SUCCESS(message))
        else:
            raise CommandError(f'Failed to disable flag: {flag_name}')

    def set_rollout(self, feature_flags, flag_name, percentage, options):
        """Set rollout percentage for a feature flag."""
        if not self.validate_flag_name(flag_name):
            raise CommandError(f'Invalid flag name: {flag_name}')

        if not 0 <= percentage <= 100:
            raise CommandError('Percentage must be between 0 and 100')

        success = feature_flags.set_rollout_percentage(flag_name, percentage)

        if success:
            message = f'Successfully set rollout for {flag_name} to {percentage}%'
            if options['json']:
                self.stdout.write(json.dumps({
                    'status': 'success',
                    'message': message,
                    'flag_name': flag_name,
                    'percentage': percentage
                }))
            else:
                self.stdout.write(self.style.SUCCESS(message))
        else:
            raise CommandError(f'Failed to set rollout for flag: {flag_name}')

    def emergency_disable_all(self, options):
        """Emergency disable all multilingual features."""
        self.stdout.write(self.style.WARNING('⚠️  WARNING: This will disable ALL multilingual features!'))

        if not options.get('json'):
            confirm = input('Are you sure you want to proceed? (yes/no): ')
            if confirm.lower() != 'yes':
                self.stdout.write('Operation cancelled.')
                return

        disabled_flags = emergency_disable_multilingual('Manual emergency disable via management command')

        message = f'Emergency disable completed. Disabled {len(disabled_flags)} flags.'
        if options['json']:
            self.stdout.write(json.dumps({
                'status': 'success',
                'message': message,
                'disabled_flags': list(disabled_flags.keys())
            }))
        else:
            self.stdout.write(self.style.ERROR(message))
            self.stdout.write('Disabled flags:')
            for flag_name in disabled_flags.keys():
                self.stdout.write(f'  • {flag_name}')

    def clear_cache(self, feature_flags, options):
        """Clear feature flags cache."""
        success = feature_flags.clear_cache()

        if success:
            message = 'Successfully cleared feature flags cache'
            if options['json']:
                self.stdout.write(json.dumps({'status': 'success', 'message': message}))
            else:
                self.stdout.write(self.style.SUCCESS(message))
        else:
            raise CommandError('Failed to clear feature flags cache')

    def validate_flag_name(self, flag_name):
        """Validate that the flag name is a known feature flag."""
        valid_flags = [
            MultilingualFeatureFlags.MULTILINGUAL_ENABLED,
            MultilingualFeatureFlags.DIRECT_GENERATION_ENABLED,
            MultilingualFeatureFlags.TRANSLATION_PIPELINE_ENABLED,
            MultilingualFeatureFlags.PARALLEL_PROCESSING_ENABLED,
            MultilingualFeatureFlags.FRENCH_GENERATION_ENABLED,
            MultilingualFeatureFlags.SPANISH_GENERATION_ENABLED,
            MultilingualFeatureFlags.QUALITY_ENHANCEMENT_ENABLED,
            MultilingualFeatureFlags.ADAPTIVE_CACHING_ENABLED,
            MultilingualFeatureFlags.PERFORMANCE_MONITORING_ENABLED,
            MultilingualFeatureFlags.EMERGENCY_FALLBACK_ENABLED,
            MultilingualFeatureFlags.CIRCUIT_BREAKER_ENABLED,
        ]

        return flag_name in valid_flags

    def get_user(self, username):
        """Get user by username."""
        try:
            return User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f'User not found: {username}')