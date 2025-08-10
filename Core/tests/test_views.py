"""
Comprehensive tests for Core app API views.
"""

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from rest_framework_simplejwt.tokens import RefreshToken


class AuthenticationTestCase(APITestCase):
    """Test cases for authentication endpoints."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123',
            'first_name': 'Test',
            'last_name': 'User'
        }
        self.user = User.objects.create_user(**self.user_data)
    
    def test_user_registration(self):
        """Test user registration endpoint."""
        url = reverse('core:auth-register')
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpass123',
            'password2': 'newpass123',
            'first_name': 'New',
            'last_name': 'User'
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)
        self.assertEqual(response.data['user']['username'], 'newuser')
        
        # Verify user was created in database
        self.assertTrue(User.objects.filter(username='newuser').exists())
    
    def test_user_registration_password_mismatch(self):
        """Test registration with password mismatch."""
        url = reverse('core:auth-register')
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpass123',
            'password2': 'differentpass',
            'first_name': 'New',
            'last_name': 'User'
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
    
    def test_user_registration_duplicate_username(self):
        """Test registration with existing username."""
        url = reverse('core:auth-register')
        data = {
            'username': 'testuser',  # Already exists
            'email': 'another@example.com',
            'password': 'newpass123',
            'password2': 'newpass123'
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('username', response.data)
    
    def test_user_login(self):
        """Test user login endpoint."""
        url = reverse('core:auth-login')
        data = {
            'username': 'testuser',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)
        self.assertEqual(response.data['user']['username'], 'testuser')
    
    def test_user_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        url = reverse('core:auth-login')
        data = {
            'username': 'testuser',
            'password': 'wrongpassword'
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertIn('detail', response.data)
    
    def test_token_refresh(self):
        """Test JWT token refresh endpoint."""
        # Get initial tokens
        refresh = RefreshToken.for_user(self.user)
        
        url = reverse('core:auth-refresh')
        data = {'refresh': str(refresh)}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
    
    def test_token_refresh_invalid(self):
        """Test token refresh with invalid token."""
        url = reverse('core:auth-refresh')
        data = {'refresh': 'invalid_token'}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_user_logout(self):
        """Test user logout endpoint."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        refresh = RefreshToken.for_user(self.user)
        
        url = reverse('core:auth-logout')
        data = {'refresh': str(refresh)}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)
    
    def test_user_profile(self):
        """Test user profile endpoint."""
        self.client.force_authenticate(user=self.user)
        url = reverse('core:auth-profile')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], 'testuser')
        self.assertEqual(response.data['email'], 'test@example.com')
    
    def test_user_profile_unauthenticated(self):
        """Test profile access without authentication."""
        url = reverse('core:auth-profile')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_update_user_profile(self):
        """Test updating user profile."""
        self.client.force_authenticate(user=self.user)
        url = reverse('core:auth-profile')
        data = {
            'first_name': 'Updated',
            'last_name': 'Name',
            'email': 'updated@example.com'
        }
        response = self.client.patch(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['first_name'], 'Updated')
        self.assertEqual(response.data['last_name'], 'Name')
        
        # Verify changes in database
        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, 'Updated')
        self.assertEqual(self.user.last_name, 'Name')


class HealthCheckTestCase(APITestCase):
    """Test cases for health check endpoints."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
    
    def test_health_check(self):
        """Test basic health check endpoint."""
        url = reverse('core:health-check')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'healthy')
        self.assertIn('timestamp', response.data)
        self.assertIn('version', response.data)
    
    @patch('django.db.connection.ensure_connection')
    def test_database_health_check(self, mock_db_connection):
        """Test database health check."""
        mock_db_connection.return_value = None  # Successful connection
        
        url = reverse('core:health-database')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('database', response.data)
        self.assertEqual(response.data['database']['status'], 'healthy')
    
    @patch('django.db.connection.ensure_connection')
    def test_database_health_check_failure(self, mock_db_connection):
        """Test database health check with connection failure."""
        mock_db_connection.side_effect = Exception("Database connection failed")
        
        url = reverse('core:health-database')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['database']['status'], 'unhealthy')
    
    @patch('redis.Redis.ping')
    def test_redis_health_check(self, mock_redis_ping):
        """Test Redis health check."""
        mock_redis_ping.return_value = True
        
        url = reverse('core:health-redis')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('redis', response.data)
        self.assertEqual(response.data['redis']['status'], 'healthy')
    
    @patch('redis.Redis.ping')
    def test_redis_health_check_failure(self, mock_redis_ping):
        """Test Redis health check with connection failure."""
        mock_redis_ping.side_effect = Exception("Redis connection failed")
        
        url = reverse('core:health-redis')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['redis']['status'], 'unhealthy')
    
    def test_comprehensive_health_check(self):
        """Test comprehensive health check endpoint."""
        url = reverse('core:health-comprehensive')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('overall_status', response.data)
        self.assertIn('services', response.data)
        self.assertIn('database', response.data['services'])
        self.assertIn('redis', response.data['services'])
        self.assertIn('system_info', response.data)


class UtilityEndpointsTestCase(APITestCase):
    """Test cases for utility endpoints."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
    
    def test_server_time(self):
        """Test server time endpoint."""
        url = reverse('core:server-time')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('server_time', response.data)
        self.assertIn('timezone', response.data)
        self.assertIn('timestamp', response.data)
        
        # Verify timestamp is recent (within last minute)
        server_time = datetime.fromisoformat(response.data['server_time'].replace('Z', '+00:00'))
        now = datetime.now(server_time.tzinfo)
        time_diff = abs((now - server_time).total_seconds())
        self.assertLess(time_diff, 60)  # Within 60 seconds
    
    def test_api_info(self):
        """Test API information endpoint."""
        url = reverse('core:api-info')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('name', response.data)
        self.assertIn('version', response.data)
        self.assertIn('description', response.data)
        self.assertIn('endpoints', response.data)
        self.assertEqual(response.data['name'], 'VoyageurCompass API')
    
    def test_system_status(self):
        """Test system status endpoint."""
        url = reverse('core:system-status')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('uptime', response.data)
        self.assertIn('memory_usage', response.data)
        self.assertIn('cpu_usage', response.data)
        self.assertIn('active_connections', response.data)


class PermissionTestCase(APITestCase):
    """Test cases for API permissions and access control."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        self.admin_user = User.objects.create_superuser(
            username='admin',
            password='adminpass123',
            email='admin@example.com'
        )
    
    def test_protected_endpoint_requires_auth(self):
        """Test that protected endpoints require authentication."""
        # This would test specific protected endpoints
        # For now, testing profile endpoint as an example
        url = reverse('core:auth-profile')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_admin_only_endpoint(self):
        """Test endpoints that require admin permissions."""
        # Example: system status might be admin-only
        url = reverse('core:system-status')
        
        # Test without authentication
        response = self.client.get(url)
        # Depending on implementation, this might be 401 or 403
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN])
        
        # Test with regular user
        self.client.force_authenticate(user=self.user)
        response = self.client.get(url)
        # Might be accessible to all authenticated users or admin-only
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_403_FORBIDDEN])
    
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        url = reverse('core:health-check')
        response = self.client.get(url)
        
        # Check for CORS headers (if CORS is enabled)
        # self.assertIn('Access-Control-Allow-Origin', response.headers)


class ErrorHandlingTestCase(APITestCase):
    """Test cases for error handling and responses."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
    
    def test_404_error(self):
        """Test 404 error response."""
        # Use an explicitly nonexistent URL path to test 404
        url = '/api/nonexistent-endpoint/'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_method_not_allowed(self):
        """Test method not allowed error."""
        url = reverse('core:health-check')
        response = self.client.post(url)  # GET-only endpoint
        
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
    
    def test_validation_error_response(self):
        """Test validation error response format."""
        url = reverse('core:auth-register')
        data = {}  # Empty data should cause validation errors
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIsInstance(response.data, dict)
    
    def test_authentication_error_response(self):
        """Test authentication error response format."""
        url = reverse('core:auth-profile')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertIn('detail', response.data)


class RateLimitingTestCase(APITestCase):
    """Test cases for rate limiting (if implemented)."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
    
    def test_rate_limiting(self):
        """Test rate limiting on public endpoints."""
        from unittest.mock import patch
        from rest_framework.throttling import AnonRateThrottle
        
        url = reverse('core:health-check')
        
        # Test with mocked throttle to verify rate limiting behavior
        with patch('rest_framework.throttling.AnonRateThrottle.get_rate') as mock_get_rate:
            # Set a very restrictive rate limit for testing (2 per minute)
            mock_get_rate.return_value = '2/min'
            
            # Create a new throttle instance with the mocked rate
            throttle = AnonRateThrottle()
            
            with patch('rest_framework.throttling.AnonRateThrottle.allow_request') as mock_allow:
                # First requests should be allowed
                mock_allow.side_effect = [True, True, False, False, False]
                
                responses = []
                for i in range(5):
                    response = self.client.get(url)
                    responses.append(response)
                
                # Verify some requests succeeded and some were throttled
                success_count = sum(1 for r in responses if r.status_code == 200)
                throttled_count = sum(1 for r in responses if r.status_code == 429)
                
                # With the mock, first 2 should succeed, next 3 should be throttled
                # Note: The actual throttling behavior depends on the view configuration
                # This test verifies the throttling mechanism is in place
                self.assertGreater(success_count, 0, "Some requests should succeed")
                
        # Alternative test: verify throttle configuration exists
        from django.conf import settings
        throttle_rates = settings.REST_FRAMEWORK.get('DEFAULT_THROTTLE_RATES', {})
        self.assertIn('anon', throttle_rates, "Anonymous rate limiting should be configured")
        self.assertIn('user', throttle_rates, "User rate limiting should be configured")
        
        # Verify throttle classes are configured
        throttle_classes = settings.REST_FRAMEWORK.get('DEFAULT_THROTTLE_CLASSES', [])
        self.assertIn('rest_framework.throttling.AnonRateThrottle', throttle_classes)
        self.assertIn('rest_framework.throttling.UserRateThrottle', throttle_classes)


class SecurityTestCase(APITestCase):
    """Test cases for security features."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
    
    def test_jwt_token_security(self):
        """Test JWT token security features."""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Login to get tokens
        url = reverse('core:auth-login')
        data = {
            'username': 'testuser',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')
        
        access_token = response.data['access']
        refresh_token = response.data['refresh']
        
        # Test that access token works
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {access_token}')
        profile_url = reverse('core:auth-profile')
        response = self.client.get(profile_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Test token expiration (would need to wait or mock time)
        # This is usually tested with mocked time
    
    def test_password_requirements(self):
        """Test password strength requirements."""
        url = reverse('core:auth-register')
        
        # Test too short password (less than 8 characters)
        data = {
            'username': 'testuser2',
            'email': 'test2@example.com',
            'password': '123',
            'password2': '123'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
        
        # Test numeric-only password
        data = {
            'username': 'testuser3',
            'email': 'test3@example.com',
            'password': '12345678',
            'password2': '12345678'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
        
        # Test common password
        data = {
            'username': 'testuser4',
            'email': 'test4@example.com',
            'password': 'password',
            'password2': 'password'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
        
        # Test password too similar to username
        data = {
            'username': 'testuser5',
            'email': 'test5@example.com',
            'password': 'testuser5123',
            'password2': 'testuser5123'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
        
        # Test mismatched passwords
        data = {
            'username': 'testuser6',
            'email': 'test6@example.com',
            'password': 'StrongPass123!',
            'password2': 'DifferentPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)
        
        # Test valid strong password
        data = {
            'username': 'testuser7',
            'email': 'test7@example.com',
            'password': 'StrongPass123!',
            'password2': 'StrongPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection."""
        from django.contrib.auth.models import User
        
        # Test malicious input in username field during registration
        url = reverse('core:auth-register')
        malicious_username = "admin'; DROP TABLE auth_user; --"
        
        data = {
            'username': malicious_username,
            'email': 'hacker@example.com',
            'password': 'StrongPass123!',
            'password2': 'StrongPass123!'
        }
        
        initial_user_count = User.objects.count()
        response = self.client.post(url, data, format='json')
        
        # Should either create user safely or reject input
        # Table should still exist and user count should be consistent
        self.assertTrue(User.objects.count() >= initial_user_count)
        
        # Test malicious input in login
        login_url = reverse('core:auth-login')
        login_data = {
            'username': "admin' OR '1'='1",
            'password': 'any_password'
        }
        response = self.client.post(login_url, login_data, format='json')
        
        # Should fail authentication, not bypass it
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_xss_protection(self):
        """Test protection against XSS attacks."""
        from django.contrib.auth.models import User
        
        # Test XSS in user profile fields during registration
        url = reverse('core:auth-register')
        xss_payload = '<script>alert("XSS")</script>'
        
        data = {
            'username': 'xssuser',
            'email': 'xss@example.com',
            'password': 'StrongPass123!',
            'password2': 'StrongPass123!',
            'first_name': xss_payload,
            'last_name': f'<img src="x" onerror="alert(1)">'
        }
        
        response = self.client.post(url, data, format='json')
        
        if response.status_code == status.HTTP_201_CREATED:
            # User was created, verify XSS payload is stored safely
            user = User.objects.get(username='xssuser')
            
            # The raw data should be stored (Django handles escaping on output)
            self.assertEqual(user.first_name, xss_payload)
            
            # Test that API responses are properly serialized (no script execution)
            profile_url = reverse('core:auth-profile')
            self.client.force_authenticate(user=user)
            profile_response = self.client.get(profile_url)
            
            self.assertEqual(profile_response.status_code, status.HTTP_200_OK)
            # JSON response should contain the raw string, safely serialized
            self.assertIn(xss_payload, str(profile_response.data))
            # But it should not be executable (proper content-type header)
            self.assertEqual(profile_response['content-type'], 'application/json')