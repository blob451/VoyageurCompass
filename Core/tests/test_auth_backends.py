"""
Unit tests for Core authentication backends.
Tests EmailOrUsernameModelBackend and BlacklistCheckMiddleware.
"""

from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.test.utils import override_settings

from Core.backends import EmailOrUsernameModelBackend, BlacklistCheckMiddleware
from Core.models import BlacklistedToken


class EmailOrUsernameModelBackendTestCase(TestCase):
    """Test cases for EmailOrUsernameModelBackend."""
    
    def setUp(self):
        """Set up test data."""
        self.backend = EmailOrUsernameModelBackend()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.request = Mock()
    
    def test_authenticate_with_username(self):
        """Test authentication with username."""
        result = self.backend.authenticate(
            self.request, 
            username='testuser', 
            password='testpass123'
        )
        self.assertEqual(result, self.user)
    
    def test_authenticate_with_email(self):
        """Test authentication with email address."""
        result = self.backend.authenticate(
            self.request, 
            username='test@example.com', 
            password='testpass123'
        )
        self.assertEqual(result, self.user)
    
    def test_authenticate_with_uppercase_username(self):
        """Test authentication with case-insensitive username."""
        result = self.backend.authenticate(
            self.request, 
            username='TESTUSER', 
            password='testpass123'
        )
        self.assertEqual(result, self.user)
    
    def test_authenticate_with_uppercase_email(self):
        """Test authentication with case-insensitive email."""
        result = self.backend.authenticate(
            self.request, 
            username='TEST@EXAMPLE.COM', 
            password='testpass123'
        )
        self.assertEqual(result, self.user)
    
    def test_authenticate_with_wrong_password(self):
        """Test authentication failure with wrong password."""
        result = self.backend.authenticate(
            self.request, 
            username='testuser', 
            password='wrongpass'
        )
        self.assertIsNone(result)
    
    def test_authenticate_with_nonexistent_user(self):
        """Test authentication failure with nonexistent user."""
        result = self.backend.authenticate(
            self.request, 
            username='nonexistent', 
            password='testpass123'
        )
        self.assertIsNone(result)
    
    def test_authenticate_with_none_username(self):
        """Test authentication failure with None username."""
        result = self.backend.authenticate(
            self.request, 
            username=None, 
            password='testpass123'
        )
        self.assertIsNone(result)
    
    def test_authenticate_with_none_password(self):
        """Test authentication failure with None password."""
        result = self.backend.authenticate(
            self.request, 
            username='testuser', 
            password=None
        )
        self.assertIsNone(result)
    
    def test_authenticate_with_multiple_users_same_email(self):
        """Test authentication failure when multiple users have same email."""
        # Create another user with same email
        User.objects.create_user(
            username='testuser2',
            email='test@example.com',
            password='testpass123'
        )
        
        result = self.backend.authenticate(
            self.request, 
            username='test@example.com', 
            password='testpass123'
        )
        self.assertIsNone(result)
    
    def test_get_user_existing(self):
        """Test get_user with existing user ID."""
        result = self.backend.get_user(self.user.pk)
        self.assertEqual(result, self.user)
    
    def test_get_user_nonexistent(self):
        """Test get_user with nonexistent user ID."""
        result = self.backend.get_user(99999)
        self.assertIsNone(result)
    
    @patch('Core.backends.logger')
    def test_logging_successful_authentication(self, mock_logger):
        """Test that successful authentication is logged."""
        self.backend.authenticate(
            self.request, 
            username='testuser', 
            password='testpass123'
        )
        mock_logger.info.assert_called_with(f"Successful authentication for user: {self.user.username}")
    
    @patch('Core.backends.logger')
    def test_logging_password_failure(self, mock_logger):
        """Test that password failures are logged."""
        self.backend.authenticate(
            self.request, 
            username='testuser', 
            password='wrongpass'
        )
        mock_logger.warning.assert_called_with("Password check failed for user: testuser")
    
    @patch('Core.backends.logger')
    def test_logging_user_not_found(self, mock_logger):
        """Test that user not found is logged."""
        self.backend.authenticate(
            self.request, 
            username='nonexistent', 
            password='testpass123'
        )
        mock_logger.warning.assert_called_with("User not found: nonexistent")


class BlacklistCheckMiddlewareTestCase(TestCase):
    """Test cases for BlacklistCheckMiddleware."""
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.get_response = Mock(return_value=Mock())
        self.middleware = BlacklistCheckMiddleware(self.get_response)
        self.test_token = 'test_jwt_token_here'
    
    def test_request_without_auth_header(self):
        """Test request without authorization header passes through."""
        request = self.factory.get('/api/some-endpoint/')
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)
    
    def test_request_with_non_bearer_auth(self):
        """Test request with non-Bearer auth passes through."""
        request = self.factory.get(
            '/api/some-endpoint/',
            HTTP_AUTHORIZATION='Basic sometoken'
        )
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)
    
    def test_logout_endpoint_bypasses_blacklist_check(self):
        """Test that logout endpoint bypasses blacklist check."""
        request = self.factory.post(
            '/api/v1/auth/logout/',
            HTTP_AUTHORIZATION=f'Bearer {self.test_token}'
        )
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)
    
    @patch('Core.models.BlacklistedToken.is_token_blacklisted')
    def test_valid_token_passes_through(self, mock_is_blacklisted):
        """Test that valid (non-blacklisted) token passes through."""
        mock_is_blacklisted.return_value = False
        
        request = self.factory.get(
            '/api/some-endpoint/',
            HTTP_AUTHORIZATION=f'Bearer {self.test_token}'
        )
        
        response = self.middleware(request)
        
        mock_is_blacklisted.assert_called_once_with(self.test_token)
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)
    
    @patch('Core.models.BlacklistedToken.is_token_blacklisted')
    def test_blacklisted_token_returns_401(self, mock_is_blacklisted):
        """Test that blacklisted token returns 401 error."""
        mock_is_blacklisted.return_value = True
        
        request = self.factory.get(
            '/api/some-endpoint/',
            HTTP_AUTHORIZATION=f'Bearer {self.test_token}'
        )
        
        response = self.middleware(request)
        
        mock_is_blacklisted.assert_called_once_with(self.test_token)
        self.get_response.assert_not_called()
        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 401)
    
    @patch('Core.models.BlacklistedToken.is_token_blacklisted')
    @patch('Core.backends.logger')
    def test_blacklist_check_exception_graceful_degradation(self, mock_logger, mock_is_blacklisted):
        """Test graceful degradation when blacklist check fails."""
        mock_is_blacklisted.side_effect = Exception("Database error")
        
        request = self.factory.get(
            '/api/some-endpoint/',
            HTTP_AUTHORIZATION=f'Bearer {self.test_token}'
        )
        
        response = self.middleware(request)
        
        # Should continue processing request despite error
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)
        mock_logger.error.assert_called()
    
    def test_malformed_bearer_token_passes_through(self):
        """Test that malformed Bearer token header passes through."""
        request = self.factory.get(
            '/api/some-endpoint/',
            HTTP_AUTHORIZATION='Bearer'  # Missing token
        )
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response, self.get_response.return_value)


class AuthBackendIntegrationTestCase(TestCase):
    """Integration tests for authentication backends."""
    
    def setUp(self):
        """Set up test data."""
        self.backend = EmailOrUsernameModelBackend()
        self.user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com',
            password='integrationpass123'
        )
    
    def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow."""
        request = Mock()
        
        # Test successful authentication
        authenticated_user = self.backend.authenticate(
            request,
            username='integrationuser',
            password='integrationpass123'
        )
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.username, 'integrationuser')
        
        # Test get_user retrieval
        retrieved_user = self.backend.get_user(authenticated_user.pk)
        self.assertEqual(retrieved_user, authenticated_user)
        
        # Test failed authentication
        failed_auth = self.backend.authenticate(
            request,
            username='integrationuser',
            password='wrongpassword'
        )
        self.assertIsNone(failed_auth)
    
    def test_concurrent_authentication_requests(self):
        """Test authentication with concurrent requests."""
        results = []
        
        def authenticate_user():
            request = Mock()
            result = self.backend.authenticate(
                request,
                username='integrationuser',
                password='integrationpass123'
            )
            results.append(result)
        
        # Simulate multiple sequential authentication requests
        for _ in range(5):
            authenticate_user()
        
        # All authentications should succeed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result)
            self.assertEqual(result.username, 'integrationuser')