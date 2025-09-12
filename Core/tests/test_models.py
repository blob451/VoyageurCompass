"""
Unit tests for Core models.
Tests UserSecurityProfile, PasswordResetRequest, and BlacklistedToken models.
"""

import uuid
from datetime import timedelta

from django.contrib.auth.models import User
from django.db import IntegrityError

# Removed mock imports - using real operations
from django.test import TestCase
from django.utils import timezone

from Core.models import BlacklistedToken, PasswordResetRequest, UserSecurityProfile


class UserSecurityProfileTestCase(TestCase):
    """Test cases for UserSecurityProfile model."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass123")
        self.security_profile = UserSecurityProfile.objects.create(
            user=self.user,
            secret_question="What is your favorite color?",
            secret_answer_hash="",  # Will be set using set_secret_answer
        )

    def test_model_creation(self):
        """Test UserSecurityProfile model creation."""
        self.assertEqual(self.security_profile.user, self.user)
        self.assertEqual(self.security_profile.secret_question, "What is your favorite color?")
        self.assertEqual(self.security_profile.failed_reset_attempts, 0)
        self.assertIsNone(self.security_profile.reset_token)
        self.assertIsNone(self.security_profile.reset_token_created)
        self.assertIsNotNone(self.security_profile.created_at)
        self.assertIsNotNone(self.security_profile.updated_at)

    def test_one_to_one_relationship(self):
        """Test one-to-one relationship with User."""
        # Access via user.security_profile
        self.assertEqual(self.user.security_profile, self.security_profile)

        # Cannot create another security profile for same user
        with self.assertRaises(IntegrityError):
            UserSecurityProfile.objects.create(
                user=self.user, secret_question="Another question?", secret_answer_hash="hash"
            )

    def test_set_secret_answer(self):
        """Test setting and hashing secret answer."""
        answer = "Blue"
        self.security_profile.set_secret_answer(answer)

        # Answer should be hashed
        self.assertNotEqual(self.security_profile.secret_answer_hash, answer)
        self.assertNotEqual(self.security_profile.secret_answer_hash, answer.lower())
        self.assertTrue(len(self.security_profile.secret_answer_hash) > 20)  # Hash is longer

    def test_set_secret_answer_normalization(self):
        """Test that secret answers are normalized before hashing."""
        # Test various formats of same answer
        answers = ["Blue", "blue", " Blue ", "BLUE", " blue "]

        # All answers should validate correctly when checked
        self.security_profile.set_secret_answer("Blue")
        self.security_profile.save()

        for answer in answers:
            # All variations should authenticate correctly
            self.assertTrue(self.security_profile.check_secret_answer(answer))

    def test_check_secret_answer_valid(self):
        """Test checking valid secret answer."""
        self.security_profile.set_secret_answer("Blue")
        self.security_profile.save()

        # Various formats should all work
        self.assertTrue(self.security_profile.check_secret_answer("Blue"))
        self.assertTrue(self.security_profile.check_secret_answer("blue"))
        self.assertTrue(self.security_profile.check_secret_answer(" Blue "))
        self.assertTrue(self.security_profile.check_secret_answer("BLUE"))
        self.assertTrue(self.security_profile.check_secret_answer(" blue "))

    def test_check_secret_answer_invalid(self):
        """Test checking invalid secret answer."""
        self.security_profile.set_secret_answer("Blue")
        self.security_profile.save()

        self.assertFalse(self.security_profile.check_secret_answer("Red"))
        self.assertFalse(self.security_profile.check_secret_answer("Green"))
        self.assertFalse(self.security_profile.check_secret_answer(""))
        self.assertFalse(self.security_profile.check_secret_answer("Blu"))

    def test_generate_reset_token(self):
        """Test generating password reset token."""
        token = self.security_profile.generate_reset_token()

        self.assertIsInstance(token, uuid.UUID)
        self.assertEqual(self.security_profile.reset_token, token)
        self.assertIsNotNone(self.security_profile.reset_token_created)

        # Token should be saved to database
        self.security_profile.refresh_from_db()
        self.assertEqual(self.security_profile.reset_token, token)

    def test_is_reset_token_valid_fresh_token(self):
        """Test valid fresh reset token."""
        self.security_profile.generate_reset_token()

        self.assertTrue(self.security_profile.is_reset_token_valid())
        self.assertTrue(self.security_profile.is_reset_token_valid(hours=1))
        self.assertTrue(self.security_profile.is_reset_token_valid(hours=24))

    def test_is_reset_token_valid_expired_token(self):
        """Test expired reset token."""
        # Create token with past timestamp
        past_time = timezone.now() - timedelta(hours=2)
        self.security_profile.reset_token = uuid.uuid4()
        self.security_profile.reset_token_created = past_time
        self.security_profile.save()

        self.assertFalse(self.security_profile.is_reset_token_valid(hours=1))
        self.assertTrue(self.security_profile.is_reset_token_valid(hours=3))

    def test_is_reset_token_valid_no_token(self):
        """Test validity check without token."""
        # No token set
        self.assertFalse(self.security_profile.is_reset_token_valid())

        # Token but no creation time
        self.security_profile.reset_token = uuid.uuid4()
        self.assertFalse(self.security_profile.is_reset_token_valid())

        # Creation time but no token
        self.security_profile.reset_token = None
        self.security_profile.reset_token_created = timezone.now()
        self.assertFalse(self.security_profile.is_reset_token_valid())

    def test_clear_reset_token(self):
        """Test clearing reset token."""
        # Generate token first
        self.security_profile.generate_reset_token()
        self.assertIsNotNone(self.security_profile.reset_token)
        self.assertIsNotNone(self.security_profile.reset_token_created)

        # Clear token
        self.security_profile.clear_reset_token()
        self.assertIsNone(self.security_profile.reset_token)
        self.assertIsNone(self.security_profile.reset_token_created)

        # Check database
        self.security_profile.refresh_from_db()
        self.assertIsNone(self.security_profile.reset_token)

    def test_str_representation(self):
        """Test string representation."""
        expected = f"Security Profile for {self.user.username}"
        self.assertEqual(str(self.security_profile), expected)


class PasswordResetRequestTestCase(TestCase):
    """Test cases for PasswordResetRequest model."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass123")
        self.reset_request = PasswordResetRequest.objects.create(
            user=self.user, reason="Forgot password and security answer"
        )

    def test_model_creation(self):
        """Test PasswordResetRequest model creation."""
        self.assertEqual(self.reset_request.user, self.user)
        self.assertEqual(self.reset_request.reason, "Forgot password and security answer")
        self.assertEqual(self.reset_request.status, "pending")
        self.assertEqual(self.reset_request.admin_notes, "")
        self.assertIsNotNone(self.reset_request.created_at)
        self.assertIsNotNone(self.reset_request.updated_at)

    def test_status_choices(self):
        """Test status field choices."""
        valid_statuses = ["pending", "approved", "rejected", "expired"]

        for status in valid_statuses:
            self.reset_request.status = status
            self.reset_request.save()
            self.reset_request.refresh_from_db()
            self.assertEqual(self.reset_request.status, status)

    def test_multiple_requests_per_user(self):
        """Test that users can have multiple reset requests."""
        request2 = PasswordResetRequest.objects.create(user=self.user, reason="Account locked", status="approved")

        requests = PasswordResetRequest.objects.filter(user=self.user)
        self.assertEqual(requests.count(), 2)
        self.assertIn(self.reset_request, requests)
        self.assertIn(request2, requests)

    def test_ordering(self):
        """Test that requests are ordered by creation date (newest first)."""
        # Create second request (will be newer)
        request2 = PasswordResetRequest.objects.create(user=self.user, reason="Second request")

        requests = list(PasswordResetRequest.objects.all())
        self.assertEqual(requests[0], request2)  # Newer first
        self.assertEqual(requests[1], self.reset_request)  # Older second

    def test_cascade_delete(self):
        """Test that requests are deleted when user is deleted."""
        request_id = self.reset_request.id
        self.user.delete()

        with self.assertRaises(PasswordResetRequest.DoesNotExist):
            PasswordResetRequest.objects.get(id=request_id)

    def test_str_representation(self):
        """Test string representation."""
        expected = f"Reset request for {self.user.username} - pending"
        self.assertEqual(str(self.reset_request), expected)

        self.reset_request.status = "approved"
        expected = f"Reset request for {self.user.username} - approved"
        self.assertEqual(str(self.reset_request), expected)


class BlacklistedTokenTestCase(TestCase):
    """Test cases for BlacklistedToken model."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass123")
        self.test_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"
        self.expires_at = timezone.now() + timedelta(hours=1)

    def test_model_creation(self):
        """Test BlacklistedToken model creation."""
        blacklisted = BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=self.expires_at)

        self.assertEqual(blacklisted.token, self.test_token)
        self.assertEqual(blacklisted.user, self.user)
        self.assertEqual(blacklisted.expires_at, self.expires_at)
        self.assertEqual(blacklisted.reason, "logout")  # Default
        self.assertIsNotNone(blacklisted.blacklisted_at)

    def test_reason_choices(self):
        """Test reason field choices."""
        valid_reasons = ["logout", "timeout", "password_change", "admin_action"]

        for reason in valid_reasons:
            blacklisted = BlacklistedToken.objects.create(
                token=f"token_{reason}", user=self.user, expires_at=self.expires_at, reason=reason
            )
            self.assertEqual(blacklisted.reason, reason)

    def test_unique_token_constraint(self):
        """Test that tokens must be unique."""
        BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=self.expires_at)

        # Cannot create another with same token
        with self.assertRaises(IntegrityError):
            BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=self.expires_at)

    def test_multiple_tokens_per_user(self):
        """Test that users can have multiple blacklisted tokens."""
        token1 = BlacklistedToken.objects.create(token="token1", user=self.user, expires_at=self.expires_at)
        token2 = BlacklistedToken.objects.create(
            token="token2", user=self.user, expires_at=self.expires_at, reason="timeout"
        )

        user_tokens = BlacklistedToken.objects.filter(user=self.user)
        self.assertEqual(user_tokens.count(), 2)
        self.assertIn(token1, user_tokens)
        self.assertIn(token2, user_tokens)

    def test_is_token_blacklisted_true(self):
        """Test is_token_blacklisted returns True for blacklisted token."""
        BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=self.expires_at)

        self.assertTrue(BlacklistedToken.is_token_blacklisted(self.test_token))

    def test_is_token_blacklisted_false(self):
        """Test is_token_blacklisted returns False for non-blacklisted token."""
        self.assertFalse(BlacklistedToken.is_token_blacklisted(self.test_token))
        self.assertFalse(BlacklistedToken.is_token_blacklisted("nonexistent.token"))

    def test_blacklist_token_success(self):
        """Test successful token blacklisting by directly creating entry."""
        # Create a blacklisted token directly
        expires_at = timezone.now() + timedelta(hours=1)
        BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=expires_at, reason="timeout")

        # Verify it was created
        self.assertTrue(BlacklistedToken.is_token_blacklisted(self.test_token))

        retrieved = BlacklistedToken.objects.get(token=self.test_token)
        self.assertEqual(retrieved.user, self.user)
        self.assertEqual(retrieved.reason, "timeout")
        self.assertEqual(retrieved.expires_at, expires_at)

    def test_blacklist_token_failure(self):
        """Test failed token blacklisting due to invalid token using real JWT operations."""
        # Use a malformed token that will actually fail JWT decode
        invalid_token = "invalid.malformed.token"

        result = BlacklistedToken.blacklist_token(invalid_token, self.user)

        self.assertFalse(result)
        self.assertFalse(BlacklistedToken.is_token_blacklisted(invalid_token))

    def test_cleanup_expired_tokens(self):
        """Test cleanup of expired tokens."""
        # Create expired token
        expired_token = BlacklistedToken.objects.create(
            token="expired.token", user=self.user, expires_at=timezone.now() - timedelta(hours=1)
        )

        # Create valid token
        valid_token = BlacklistedToken.objects.create(
            token="valid.token", user=self.user, expires_at=timezone.now() + timedelta(hours=1)
        )

        # Cleanup expired tokens
        deleted_count = BlacklistedToken.cleanup_expired_tokens()

        self.assertEqual(deleted_count, 1)
        self.assertFalse(BlacklistedToken.objects.filter(id=expired_token.id).exists())
        self.assertTrue(BlacklistedToken.objects.filter(id=valid_token.id).exists())

    def test_str_representation(self):
        """Test string representation."""
        blacklisted = BlacklistedToken.objects.create(
            token=self.test_token, user=self.user, expires_at=self.expires_at, reason="timeout"
        )

        expected = f"Blacklisted token for {self.user.username} (timeout)"
        self.assertEqual(str(blacklisted), expected)

    def test_cascade_delete(self):
        """Test that tokens are deleted when user is deleted."""
        blacklisted = BlacklistedToken.objects.create(token=self.test_token, user=self.user, expires_at=self.expires_at)

        token_id = blacklisted.id
        self.user.delete()

        with self.assertRaises(BlacklistedToken.DoesNotExist):
            BlacklistedToken.objects.get(id=token_id)


class CoreModelsIntegrationTestCase(TestCase):
    """Integration tests for Core models working together."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="integrationuser", email="integration@example.com", password="integrationpass123"
        )

    def test_complete_password_reset_flow(self):
        """Test complete password reset flow using models."""
        # 1. Create security profile
        security_profile = UserSecurityProfile.objects.create(
            user=self.user, secret_question="What is your pet's name?"
        )
        security_profile.set_secret_answer("Fluffy")
        security_profile.save()

        # 2. User forgets password, answers security question correctly
        self.assertTrue(security_profile.check_secret_answer("fluffy"))

        # 3. Generate reset token
        security_profile.generate_reset_token()
        self.assertTrue(security_profile.is_reset_token_valid())

        # 4. User resets password (simulate password change)
        self.user.set_password("newpassword123")
        self.user.save()

        # 5. Clear reset token after successful reset
        security_profile.clear_reset_token()
        self.assertIsNone(security_profile.reset_token)

        # 6. Blacklist any existing tokens due to password change
        test_token = "user.jwt.token"
        BlacklistedToken.blacklist_token(test_token, self.user, reason="password_change")
        # Note: This will fail due to invalid token format, but that's expected

    def test_security_profile_and_reset_request_interaction(self):
        """Test interaction between security profile and reset requests."""
        # Create security profile
        security_profile = UserSecurityProfile.objects.create(user=self.user, secret_question="What is your hometown?")
        security_profile.set_secret_answer("Springfield")

        # User tries wrong answer multiple times
        for i in range(3):
            self.assertFalse(security_profile.check_secret_answer("wrong answer"))
            security_profile.failed_reset_attempts += 1

        security_profile.save()

        # After failed attempts, create manual reset request
        reset_request = PasswordResetRequest.objects.create(
            user=self.user, reason=f"Failed security question {security_profile.failed_reset_attempts} times"
        )

        self.assertEqual(reset_request.status, "pending")
        self.assertEqual(security_profile.failed_reset_attempts, 3)

        # Admin approves request
        reset_request.status = "approved"
        reset_request.admin_notes = "Verified user identity via phone"
        reset_request.save()

        # Reset failed attempts after admin approval
        security_profile.failed_reset_attempts = 0
        security_profile.save()

        self.assertEqual(reset_request.status, "approved")
        self.assertEqual(security_profile.failed_reset_attempts, 0)

    def test_user_deletion_cascades(self):
        """Test that all related objects are deleted when user is deleted."""
        # Create all related objects
        security_profile = UserSecurityProfile.objects.create(user=self.user, secret_question="Test question?")

        reset_request = PasswordResetRequest.objects.create(user=self.user, reason="Test reset")

        blacklisted_token = BlacklistedToken.objects.create(
            token="test.token", user=self.user, expires_at=timezone.now() + timedelta(hours=1)
        )

        # Store IDs
        security_id = security_profile.id
        request_id = reset_request.id
        token_id = blacklisted_token.id

        # Delete user
        self.user.delete()

        # All related objects should be deleted
        with self.assertRaises(UserSecurityProfile.DoesNotExist):
            UserSecurityProfile.objects.get(id=security_id)

        with self.assertRaises(PasswordResetRequest.DoesNotExist):
            PasswordResetRequest.objects.get(id=request_id)

        with self.assertRaises(BlacklistedToken.DoesNotExist):
            BlacklistedToken.objects.get(id=token_id)
