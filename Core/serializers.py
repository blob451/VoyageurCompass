"""
Serializers for Core app.
Handles user authentication and registration.
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework.validators import UniqueValidator
from Core.models import UserSecurityProfile, PasswordResetRequest


class UserSerializer(serializers.ModelSerializer):
    """Basic user serializer."""
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration with security question."""
    
    email = serializers.EmailField(
        required=True,
        validators=[UniqueValidator(queryset=User.objects.all())]
    )
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        label='Confirm Password'
    )
    secret_question = serializers.CharField(
        required=True,
        max_length=255,
        help_text="Security question for password recovery"
    )
    secret_answer = serializers.CharField(
        write_only=True,
        required=True,
        min_length=3,
        help_text="Answer to the security question"
    )
    
    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'password2',
            'first_name', 'last_name', 'secret_question', 'secret_answer'
        ]
        extra_kwargs = {
            'first_name': {'required': False},
            'last_name': {'required': False}
        }
    
    def validate(self, attrs):
        """Validate that passwords match."""
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({
                'password': "Password fields didn't match."
            })
        return attrs
    
    def create(self, validated_data):
        """Create user with validated data and security profile."""
        # Extract security data
        secret_question = validated_data.pop('secret_question')
        secret_answer = validated_data.pop('secret_answer')
        validated_data.pop('password2')
        
        # Create user
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )
        
        # Create security profile
        security_profile = UserSecurityProfile(
            user=user,
            secret_question=secret_question
        )
        security_profile.set_secret_answer(secret_answer)
        security_profile.save()
        
        return user


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for password change."""
    
    old_password = serializers.CharField(
        required=True,
        style={'input_type': 'password'}
    )
    new_password = serializers.CharField(
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password2 = serializers.CharField(
        required=True,
        style={'input_type': 'password'},
        label='Confirm New Password'
    )
    
    def validate(self, attrs):
        """Validate that new passwords match."""
        if attrs['new_password'] != attrs['new_password2']:
            raise serializers.ValidationError({
                'new_password': "New password fields didn't match."
            })
        return attrs
    
    def validate_old_password(self, value):
        """Validate old password is correct."""
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError('Old password is incorrect')
        return value


class UserProfileSerializer(serializers.ModelSerializer):
    """Detailed user profile serializer."""
    
    portfolio_count = serializers.SerializerMethodField()
    last_login = serializers.DateTimeField(read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name',
            'date_joined', 'last_login', 'portfolio_count'
        ]
        read_only_fields = ['id', 'username', 'date_joined', 'last_login']
    
    def get_portfolio_count(self, obj):
        """Get number of portfolios for the user."""
        return obj.portfolios.count()


class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for requesting password reset with secret answer."""
    
    username = serializers.CharField(
        required=True,
        help_text="Username or email address"
    )
    secret_answer = serializers.CharField(
        required=True,
        write_only=True,
        help_text="Answer to your security question"
    )
    
    def validate(self, attrs):
        """Validate username exists and secret answer is correct."""
        username = attrs.get('username')
        secret_answer = attrs.get('secret_answer')
        
        # Try to find user by username or email
        user = None
        if '@' in username:
            try:
                user = User.objects.get(email=username)
            except User.DoesNotExist:
                pass
        
        if not user:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                pass
        
        if not user:
            raise serializers.ValidationError("User not found")
        
        # Check if user has security profile
        try:
            security_profile = user.security_profile
        except:
            raise serializers.ValidationError("Security profile not found. Please contact admin.")
        
        # Verify secret answer
        if not security_profile.check_secret_answer(secret_answer):
            # Increment failed attempts
            security_profile.failed_reset_attempts += 1
            from django.utils import timezone
            security_profile.last_reset_attempt = timezone.now()
            security_profile.save()
            
            if security_profile.failed_reset_attempts >= 5:
                raise serializers.ValidationError(
                    "Too many failed attempts. Please contact admin for assistance."
                )
            raise serializers.ValidationError("Incorrect answer to security question")
        
        # Reset failed attempts on success
        security_profile.failed_reset_attempts = 0
        security_profile.save()
        
        attrs['user'] = user
        return attrs


class PasswordResetSerializer(serializers.Serializer):
    """Serializer for resetting password with token."""
    
    token = serializers.UUIDField(
        required=True,
        help_text="Password reset token"
    )
    new_password = serializers.CharField(
        required=True,
        write_only=True,
        validators=[validate_password],
        style={'input_type': 'password'}
    )
    new_password2 = serializers.CharField(
        required=True,
        write_only=True,
        style={'input_type': 'password'},
        label='Confirm New Password'
    )
    
    def validate(self, attrs):
        """Validate token and passwords."""
        token = attrs.get('token')
        new_password = attrs.get('new_password')
        new_password2 = attrs.get('new_password2')
        
        # Check passwords match
        if new_password != new_password2:
            raise serializers.ValidationError("Passwords don't match")
        
        # Find user by token
        try:
            security_profile = UserSecurityProfile.objects.get(reset_token=token)
        except UserSecurityProfile.DoesNotExist:
            raise serializers.ValidationError("Invalid or expired reset token")
        
        # Check if token is valid
        if not security_profile.is_reset_token_valid():
            raise serializers.ValidationError("Reset token has expired")
        
        attrs['user'] = security_profile.user
        attrs['security_profile'] = security_profile
        return attrs


class AdminPasswordResetRequestSerializer(serializers.ModelSerializer):
    """Serializer for requesting admin help with password reset."""
    
    username = serializers.CharField(
        required=True,
        help_text="Your username or email"
    )
    
    class Meta:
        model = PasswordResetRequest
        fields = ['username', 'reason']
    
    def validate_username(self, value):
        """Validate that user exists."""
        user = None
        if '@' in value:
            try:
                user = User.objects.get(email=value)
            except User.DoesNotExist:
                pass
        
        if not user:
            try:
                user = User.objects.get(username=value)
            except User.DoesNotExist:
                pass
        
        if not user:
            raise serializers.ValidationError("User not found")
        
        # Check for existing pending request
        existing = PasswordResetRequest.objects.filter(
            user=user,
            status='pending'
        ).exists()
        
        if existing:
            raise serializers.ValidationError(
                "You already have a pending reset request. Please wait for admin response."
            )
        
        self.context['user'] = user
        return value
    
    def create(self, validated_data):
        """Create the password reset request."""
        validated_data.pop('username')
        user = self.context['user']
        
        return PasswordResetRequest.objects.create(
            user=user,
            **validated_data
        )