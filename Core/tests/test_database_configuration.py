"""
Database configuration validation tests for CI/CD pipeline.
Ensures proper database setup in different environments using Canadian English conventions.
"""

import os
import unittest
from django.test import TestCase, override_settings
from django.db import connection
from django.core.management import call_command
from django.core.exceptions import ImproperlyConfigured


class DatabaseConfigurationTestCase(TestCase):
    """Database configuration logic validation across environments."""

    def test_database_engine_detection(self):
        """Database engine configuration verification for environment."""
        # In CI environment, should use PostgreSQL
        if os.getenv('CI', '').lower() == 'true':
            self.assertEqual(connection.vendor, 'postgresql')
            self.assertIn('voyageur_test_db', connection.settings_dict['NAME'])
        else:
            # In local environment, should use SQLite
            self.assertEqual(connection.vendor, 'sqlite')
            # Django test framework modifies the database name for isolation
            self.assertIn('memory', connection.settings_dict['NAME'].lower())

    def test_database_connectivity(self):
        """Database connection functionality verification."""
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)

    def test_migration_compatibility(self):
        """Verify migrations work properly in current database engine."""
        try:
            call_command('migrate', verbosity=0, interactive=False)
            # If we get here, migrations completed successfully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Migration failed: {e}")

    def test_content_types_table_exists(self):
        """Verify django_content_type table exists after migrations."""
        with connection.cursor() as cursor:
            # Check if django_content_type table exists
            if connection.vendor == 'postgresql':
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = 'django_content_type'
                """)
            else:  # SQLite
                cursor.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name='django_content_type'
                """)
            
            result = cursor.fetchone()
            self.assertEqual(result[0], 1, "django_content_type table should exist")

    def test_basic_model_operations(self):
        """Test basic database operations work correctly."""
        from django.contrib.auth.models import User
        
        # Test creation
        user = User.objects.create_user(
            username='test_db_config',
            email='test@example.com',
            password='testpass123'
        )
        self.assertIsNotNone(user.id)
        
        # Test retrieval
        retrieved_user = User.objects.get(username='test_db_config')
        self.assertEqual(retrieved_user.email, 'test@example.com')
        
        # Test update
        retrieved_user.email = 'updated@example.com'
        retrieved_user.save()
        
        updated_user = User.objects.get(username='test_db_config')
        self.assertEqual(updated_user.email, 'updated@example.com')
        
        # Test deletion
        updated_user.delete()
        self.assertFalse(User.objects.filter(username='test_db_config').exists())


class EnvironmentDetectionTestCase(unittest.TestCase):
    """Test environment detection logic without Django setup."""

    def test_ci_environment_detection(self):
        """Test CI environment detection logic."""
        # Simulate CI environment
        with unittest.mock.patch.dict(os.environ, {'CI': 'true'}):
            from VoyageurCompass.test_settings import IS_CI_ENVIRONMENT
            # Need to reload the module to pick up environment changes
            import importlib
            import VoyageurCompass.test_settings
            importlib.reload(VoyageurCompass.test_settings)
            self.assertTrue(VoyageurCompass.test_settings.IS_CI_ENVIRONMENT)

        # Simulate local environment
        with unittest.mock.patch.dict(os.environ, {'CI': 'false'}, clear=True):
            importlib.reload(VoyageurCompass.test_settings)
            self.assertFalse(VoyageurCompass.test_settings.IS_CI_ENVIRONMENT)

    def test_database_url_parsing(self):
        """Test DATABASE_URL parsing logic."""
        test_url = "postgresql://test_user:test_password@localhost:5432/voyageur_test_db"
        
        with unittest.mock.patch.dict(os.environ, {
            'CI': 'true',
            'DATABASE_URL': test_url
        }):
            import importlib
            import VoyageurCompass.test_settings
            importlib.reload(VoyageurCompass.test_settings)
            
            # Verify DATABASE_URL is captured
            self.assertEqual(VoyageurCompass.test_settings.DATABASE_URL, test_url)