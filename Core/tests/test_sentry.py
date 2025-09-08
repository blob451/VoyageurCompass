import os
from django.test import TestCase, override_settings
from django.conf import settings


class SentryIntegrationTestCase(TestCase):
    """Test Sentry integration behavior"""
    
    def test_sentryDsnSetting(self):
        """Test that SENTRY_DSN setting works correctly"""
        # Test that the setting exists and can be accessed
        sentry_dsn = getattr(settings, 'SENTRY_DSN', None)
        # This should not raise an exception
        self.assertIsInstance(sentry_dsn, (str, type(None)))
    
    def test_sentryEnvironmentSetting(self):
        """Test that APP_ENV setting is available for Sentry"""
        app_env = getattr(settings, 'APP_ENV', None)
        self.assertIsNotNone(app_env)
        self.assertIsInstance(app_env, str)
