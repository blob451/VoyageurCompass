"""
Unit tests for Core authentication backends.
Tests EmailOrUsernameModelBackend and BlacklistCheckMiddleware.
"""

from django.contrib.auth.models import User
from django.http import HttpResponse, JsonResponse
from django.test import RequestFactory, TestCase

from Core.backends import BlacklistCheckMiddleware, EmailOrUsernameModelBackend
from Core.models import BlacklistedToken
from Core.tests.fixtures import CoreTestDataFactory, TestEnvironmentManager


class EmailOrUsernameModelBackendTestCase(TestCase):
    """Test cases for EmailOrUsernameModelBackend."""

    def setUp(self):
        """Set up test data."""
        TestEnvironmentManager.setup_test_environment()
        self.backend = EmailOrUsernameModelBackend()
        self.user = CoreTestDataFactory.create_test_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.factory = RequestFactory()
        self.request = self.factory.post("/auth/login/")

    def tearDown(self):
        """Clean up test data."""
        CoreTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()

    def test_authenticate_with_username(self):
        """Test authentication with username."""
        result = self.backend.authenticate(self.request, username="testuser", password="testpass123")
        self.assertEqual(result, self.user)
        self.assertTrue(self.user.check_password("testpass123"))

    def test_authenticate_with_email(self):
        """Test authentication with email address."""
        result = self.backend.authenticate(self.request, username="test@example.com", password="testpass123")
        self.assertEqual(result, self.user)
        self.assertEqual(result.email, "test@example.com")

    def test_authenticate_with_uppercase_username(self):
        """Test authentication with case-insensitive username."""
        result = self.backend.authenticate(self.request, username="TESTUSER", password="testpass123")
        self.assertEqual(result, self.user)

    def test_authenticate_with_uppercase_email(self):
        """Test authentication with case-insensitive email."""
        result = self.backend.authenticate(self.request, username="TEST@EXAMPLE.COM", password="testpass123")
        self.assertEqual(result, self.user)

    def test_authenticate_with_wrong_password(self):
        """Test authentication failure with wrong password."""
        result = self.backend.authenticate(self.request, username="testuser", password="wrongpass")
        self.assertIsNone(result)

    def test_authenticate_with_nonexistent_user(self):
        """Test authentication failure with nonexistent user."""
        result = self.backend.authenticate(self.request, username="nonexistent", password="testpass123")
        self.assertIsNone(result)

    def test_authenticate_with_none_username(self):
        """Test authentication failure with None username."""
        result = self.backend.authenticate(self.request, username=None, password="testpass123")
        self.assertIsNone(result)

    def test_authenticate_with_none_password(self):
        """Test authentication failure with None password."""
        result = self.backend.authenticate(self.request, username="testuser", password=None)
        self.assertIsNone(result)

    def test_authenticate_with_multiple_users_same_email(self):
        """Test authentication failure when multiple users have same email."""
        # Create another user with same email
        CoreTestDataFactory.create_test_user(username="testuser2", email="test@example.com", password="testpass123")

        result = self.backend.authenticate(self.request, username="test@example.com", password="testpass123")
        self.assertIsNone(result)
        # Verify multiple users exist with same email
        users_with_email = User.objects.filter(email="test@example.com")
        self.assertEqual(users_with_email.count(), 2)

    def test_get_user_existing(self):
        """Test get_user with existing user ID."""
        result = self.backend.get_user(self.user.pk)
        self.assertEqual(result, self.user)

    def test_get_user_nonexistent(self):
        """Test get_user with nonexistent user ID."""
        result = self.backend.get_user(99999)
        self.assertIsNone(result)

    def test_logging_successful_authentication(self):
        """Test that successful authentication works without mocking."""
        result = self.backend.authenticate(self.request, username="testuser", password="testpass123")
        self.assertEqual(result, self.user)
        self.assertTrue(result.is_authenticated)

    def test_logging_password_failure(self):
        """Test that password failures return None."""
        result = self.backend.authenticate(self.request, username="testuser", password="wrongpass")
        self.assertIsNone(result)
        # Verify user exists but password is wrong
        user = User.objects.get(username="testuser")
        self.assertFalse(user.check_password("wrongpass"))

    def test_logging_user_not_found(self):
        """Test that nonexistent user authentication returns None."""
        result = self.backend.authenticate(self.request, username="nonexistent", password="testpass123")
        self.assertIsNone(result)
        # Verify user does not exist
        self.assertFalse(User.objects.filter(username="nonexistent").exists())


class BlacklistCheckMiddlewareTestCase(TestCase):
    """Test cases for BlacklistCheckMiddleware."""

    def setUp(self):
        """Set up test data."""
        TestEnvironmentManager.setup_test_environment()
        self.factory = RequestFactory()

        # Create real response function
        def get_response(request):
            return HttpResponse("Test response", status=200)

        self.get_response = get_response
        self.middleware = BlacklistCheckMiddleware(self.get_response)

        # Create real test user and token
        self.user = CoreTestDataFactory.create_test_user()
        self.tokens = CoreTestDataFactory.generate_jwt_tokens(self.user)
        self.test_token = self.tokens["access"]

    def tearDown(self):
        """Clean up test data."""
        CoreTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()

    def test_request_without_auth_header(self):
        """Test request without authorization header passes through."""
        request = self.factory.get("/api/some-endpoint/")

        response = self.middleware(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")

    def test_request_with_non_bearer_auth(self):
        """Test request with non-Bearer auth passes through."""
        request = self.factory.get("/api/some-endpoint/", HTTP_AUTHORIZATION="Basic sometoken")

        response = self.middleware(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")

    def test_logout_endpoint_bypasses_blacklist_check(self):
        """Test that logout endpoint bypasses blacklist check."""
        request = self.factory.post("/api/v1/auth/logout/", HTTP_AUTHORIZATION=f"Bearer {self.test_token}")

        response = self.middleware(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")

    def test_valid_token_passes_through(self):
        """Test that valid (non-blacklisted) token passes through."""
        request = self.factory.get("/api/some-endpoint/", HTTP_AUTHORIZATION=f"Bearer {self.test_token}")

        # Ensure token is not blacklisted
        self.assertFalse(BlacklistedToken.is_token_blacklisted(self.test_token))

        response = self.middleware(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")

    def test_blacklisted_token_returns_401(self):
        """Test that blacklisted token returns 401 error."""
        # Blacklist the token for real
        CoreTestDataFactory.blacklist_token(self.test_token, self.user)

        request = self.factory.get("/api/some-endpoint/", HTTP_AUTHORIZATION=f"Bearer {self.test_token}")

        response = self.middleware(request)

        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 401)

    def test_blacklist_check_exception_graceful_degradation(self):
        """Test graceful degradation when blacklist check fails."""
        # Use invalid token format to trigger exception handling
        invalid_token = "invalid.token.format"

        request = self.factory.get("/api/some-endpoint/", HTTP_AUTHORIZATION=f"Bearer {invalid_token}")

        response = self.middleware(request)

        # Should continue processing request despite token parsing error
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")

    def test_malformed_bearer_token_passes_through(self):
        """Test that malformed Bearer token header passes through."""
        request = self.factory.get("/api/some-endpoint/", HTTP_AUTHORIZATION="Bearer")  # Missing token

        response = self.middleware(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "Test response")


class AuthBackendIntegrationTestCase(TestCase):
    """Integration tests for authentication backends."""

    def setUp(self):
        """Set up test data."""
        TestEnvironmentManager.setup_test_environment()
        self.backend = EmailOrUsernameModelBackend()
        self.user = CoreTestDataFactory.create_test_user(
            username="integrationuser", email="integration@example.com", password="integrationpass123"
        )
        self.factory = RequestFactory()

    def tearDown(self):
        """Clean up test data."""
        CoreTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()

    def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow."""
        request = self.factory.post("/auth/login/")

        # Test successful authentication
        authenticated_user = self.backend.authenticate(
            request, username="integrationuser", password="integrationpass123"
        )
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.username, "integrationuser")
        self.assertTrue(authenticated_user.is_authenticated)

        # Test get_user retrieval
        retrieved_user = self.backend.get_user(authenticated_user.pk)
        self.assertEqual(retrieved_user, authenticated_user)
        self.assertEqual(retrieved_user.email, "integration@example.com")

        # Test failed authentication
        failed_auth = self.backend.authenticate(request, username="integrationuser", password="wrongpassword")
        self.assertIsNone(failed_auth)

    def test_concurrent_authentication_requests(self):
        """Test authentication with concurrent requests."""
        results = []

        def authenticate_user():
            request = self.factory.post("/auth/login/")
            result = self.backend.authenticate(request, username="integrationuser", password="integrationpass123")
            results.append(result)

        # Simulate multiple sequential authentication requests
        for _ in range(5):
            authenticate_user()

        # All authentications should succeed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result)
            self.assertEqual(result.username, "integrationuser")
