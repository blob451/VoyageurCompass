"""
Core models for user management and security.
"""

from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password
import uuid


class UserSecurityProfile(models.Model):
    """Extended security profile with password recovery functionality."""
    
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='security_profile'
    )
    
    secret_question = models.CharField(
        max_length=255,
        help_text="Security question for password recovery"
    )
    
    secret_answer_hash = models.CharField(
        max_length=255,
        help_text="Hashed answer to the security question"
    )
    
    reset_token = models.UUIDField(
        null=True,
        blank=True,
        help_text="Temporary token for password reset"
    )
    
    reset_token_created = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the reset token was created"
    )
    
    failed_reset_attempts = models.IntegerField(
        default=0,
        help_text="Number of failed password reset attempts"
    )
    
    last_reset_attempt = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last password reset attempt timestamp"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'core_user_security_profile'
        verbose_name = 'User Security Profile'
        verbose_name_plural = 'User Security Profiles'
    
    def set_secret_answer(self, answer):
        """Hash and store security question answer."""
        normalised_answer = answer.lower().strip()
        self.secret_answer_hash = make_password(normalised_answer)
    
    def check_secret_answer(self, answer):
        """Verify security question answer."""
        normalised_answer = answer.lower().strip()
        return check_password(normalised_answer, self.secret_answer_hash)
    
    def generate_reset_token(self):
        """Generate timestamped password reset token."""
        from django.utils import timezone
        self.reset_token = uuid.uuid4()
        self.reset_token_created = timezone.now()
        self.save()
        return self.reset_token
    
    def is_reset_token_valid(self, hours=1):
        """Validate reset token expiration status."""
        if not self.reset_token or not self.reset_token_created:
            return False
        
        from django.utils import timezone
        from datetime import timedelta
        
        expiry_time = self.reset_token_created + timedelta(hours=hours)
        return timezone.now() <= expiry_time
    
    def clear_reset_token(self):
        """Clear reset token after successful password reset."""
        self.reset_token = None
        self.reset_token_created = None
        self.save()
    
    def __str__(self):
        return f"Security Profile for {self.user.username}"


class PasswordResetRequest(models.Model):
    """Administrative password reset requests for forgotten security answers."""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('expired', 'Expired'),
    ]
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='reset_requests'
    )
    
    reason = models.TextField(
        help_text="Reason for password reset request"
    )
    
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    
    admin_notes = models.TextField(
        blank=True,
        help_text="Notes from admin about this request"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'core_password_reset_request'
        ordering = ['-created_at']
        verbose_name = 'Password Reset Request'
        verbose_name_plural = 'Password Reset Requests'
    
    def __str__(self):
        return f"Reset request for {self.user.username} - {self.status}"


class BlacklistedToken(models.Model):
    """Blacklisted JWT tokens for secure logout functionality."""
    token = models.TextField(unique=True, db_index=True)
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='blacklisted_tokens'
    )
    blacklisted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    reason = models.CharField(
        max_length=50,
        choices=[
            ('logout', 'User Logout'),
            ('timeout', 'Session Timeout'),
            ('password_change', 'Password Change'),
            ('admin_action', 'Admin Action'),
        ],
        default='logout'
    )
    
    class Meta:
        db_table = 'core_blacklisted_token'
        ordering = ['-blacklisted_at']
        verbose_name = 'Blacklisted Token'
        verbose_name_plural = 'Blacklisted Tokens'
        indexes = [
            models.Index(fields=['token']),
            models.Index(fields=['expires_at']),
        ]
    
    def __str__(self):
        return f"Blacklisted token for {self.user.username} ({self.reason})"
    
    @classmethod
    def is_token_blacklisted(cls, token):
        """Verify token blacklist status."""
        return cls.objects.filter(token=token).exists()
    
    @classmethod
    def blacklist_token(cls, token, user, reason='logout'):
        """Add JWT token to blacklist with decoded expiration."""
        try:
            from rest_framework_simplejwt.tokens import UntypedToken
            from django.utils import timezone
            import jwt
            from django.conf import settings
            
            decoded_token = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=["HS256"]
            )
            expires_at = timezone.datetime.fromtimestamp(
                decoded_token['exp'], 
                tz=timezone.utc
            )
            
            cls.objects.create(
                token=token,
                user=user,
                expires_at=expires_at,
                reason=reason
            )
            return True
        except Exception:
            return False
    
    @classmethod
    def cleanup_expired_tokens(cls):
        """Purge expired tokens from blacklist."""
        from django.utils import timezone
        expired_count = cls.objects.filter(
            expires_at__lt=timezone.now()
        ).delete()[0]
        return expired_count