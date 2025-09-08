"""
Real test data fixtures for Core module testing.
"""
import os
from datetime import datetime, timedelta
from decimal import Decimal
from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework_simplejwt.tokens import RefreshToken
from Core.models import UserSecurityProfile, BlacklistedToken


class CoreTestDataFactory:
    """Factory for creating real test data without mocks."""
    
    @staticmethod
    def create_test_user(username="testuser", email="test@example.com", password="TestPassword123!"):
        """Create real test user with authentication capabilities."""
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name="Test",
            last_name="User"
        )
        
        # Create security profile
        security_profile = UserSecurityProfile.objects.create(
            user=user,
            secret_question="What is your favourite testing framework?",
            secret_answer_hash="hashed_answer_here"  # Will be properly hashed by model method
        )
        security_profile.set_secret_answer("Real Testing")
        
        return user
    
    @staticmethod
    def create_admin_user(username="adminuser", email="admin@example.com"):
        """Create real admin user for permission testing."""
        user = CoreTestDataFactory.create_test_user(username, email)
        user.is_staff = True
        user.is_superuser = True
        user.save()
        return user
    
    @staticmethod
    def generate_jwt_tokens(user):
        """Generate real JWT tokens for authentication testing."""
        refresh = RefreshToken.for_user(user)
        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }
    
    @staticmethod
    def blacklist_token(token, user=None):
        """Create real blacklisted token for testing."""
        if user is None:
            # Get or create a test user for the blacklisted token
            user = User.objects.filter(username__startswith='test').first()
            if not user:
                user = CoreTestDataFactory.create_test_user()
        
        return BlacklistedToken.objects.create(
            token=token,
            user=user,
            expires_at=timezone.now() + timedelta(days=7),
            reason='logout'
        )
    
    @staticmethod
    def create_multiple_users(count=5):
        """Create multiple test users for concurrent testing."""
        users = []
        for i in range(count):
            user = CoreTestDataFactory.create_test_user(
                username=f"testuser{i}",
                email=f"test{i}@example.com"
            )
            users.append(user)
        return users
    
    @staticmethod
    def create_test_request_data():
        """Create real request data for middleware testing."""
        return {
            'path': '/api/v1/test/',
            'method': 'GET',
            'user_agent': 'Mozilla/5.0 (Test Browser)',
            'remote_addr': '127.0.0.1',
            'headers': {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Test Browser)'
            }
        }
    
    @staticmethod
    def cleanup_test_data():
        """Clean up test data after test completion."""
        # Clean up users and related data
        User.objects.filter(username__startswith='testuser').delete()
        User.objects.filter(username__startswith='adminuser').delete()
        User.objects.filter(username__startswith='integrationuser').delete()
        
        # Clean up security profiles
        UserSecurityProfile.objects.filter(
            user__username__startswith='test'
        ).delete()
        
        # Clean up blacklisted tokens
        BlacklistedToken.objects.filter(
            blacklisted_at__gte=timezone.now() - timedelta(hours=1)
        ).delete()


class DatabaseTestUtilities:
    """Utilities for real database testing without mocks."""
    
    @staticmethod
    def verify_database_connection():
        """Verify real database connection availability."""
        from django.db import connection
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except Exception:
            return False
    
    @staticmethod
    def get_database_info():
        """Get real database connection information."""
        from django.db import connection
        return {
            'vendor': connection.vendor,
            'database': connection.settings_dict['NAME'],
            'host': connection.settings_dict['HOST'],
            'port': connection.settings_dict['PORT']
        }
    
    @staticmethod
    def execute_raw_sql(sql, params=None):
        """Execute raw SQL for advanced testing scenarios."""
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute(sql, params or [])
            if sql.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            return cursor.rowcount
    
    @staticmethod
    def reset_sequences():
        """Reset database sequences for consistent testing."""
        from django.core.management.color import no_style
        from django.db import connection
        
        style = no_style()
        sql = connection.ops.sql_flush(style, [User._meta.db_table])
        if sql:
            with connection.cursor() as cursor:
                for query in sql:
                    cursor.execute(query)


class TestEnvironmentManager:
    """Manager for test environment setup and teardown."""
    
    @staticmethod
    def setup_test_environment():
        """Set up complete test environment with real services."""
        # Verify database connectivity
        if not DatabaseTestUtilities.verify_database_connection():
            raise RuntimeError("Test database connection failed")
        
        # Create test directories
        test_dirs = [
            'Temp/test_media',
            'Temp/test_static',
            'Temp/test_logs',
            'Temp/test_uploads'
        ]
        
        for test_dir in test_dirs:
            os.makedirs(test_dir, exist_ok=True)
        
        return {
            'database': DatabaseTestUtilities.get_database_info(),
            'directories_created': test_dirs,
            'timestamp': timezone.now().isoformat()
        }
    
    @staticmethod
    def teardown_test_environment():
        """Clean up test environment after testing."""
        # Clean up test data
        CoreTestDataFactory.cleanup_test_data()
        
        # Clean up test directories
        import shutil
        test_dirs = [
            'Temp/test_media',
            'Temp/test_static', 
            'Temp/test_logs',
            'Temp/test_uploads'
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)
    
    @staticmethod
    def get_environment_status():
        """Get current test environment status."""
        return {
            'database_connected': DatabaseTestUtilities.verify_database_connection(),
            'user_count': User.objects.filter(username__startswith='test').count(),
            'blacklisted_tokens': BlacklistedToken.objects.count(),
            'timestamp': timezone.now().isoformat()
        }
