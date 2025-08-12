"""
Serializers for Core app.
Handles user authentication and registration.
"""

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from rest_framework.validators import UniqueValidator


class UserSerializer(serializers.ModelSerializer):
    """Basic user serializer."""

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "date_joined"]
        read_only_fields = ["id", "date_joined"]


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""

    email = serializers.EmailField(
        required=True, validators=[UniqueValidator(queryset=User.objects.all())]
    )
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={"input_type": "password"},
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={"input_type": "password"},
        label="Confirm Password",
    )

    class Meta:
        model = User
        fields = [
            "username",
            "email",
            "password",
            "password2",
            "first_name",
            "last_name",
        ]
        extra_kwargs = {
            "first_name": {"required": False},
            "last_name": {"required": False},
        }

    def validate(self, attrs):
        """Validate that passwords match."""
        if attrs["password"] != attrs["password2"]:
            raise serializers.ValidationError(
                {"password": "Password fields didn't match."}
            )
        
        # Additional password validation for common weak passwords
        password = attrs.get("password", "")
        username = attrs.get("username", "")
        
        # Check for very weak passwords that should be rejected
        if len(password) < 8:
            raise serializers.ValidationError(
                {"password": "This password is too short. It must contain at least 8 characters."}
            )
        
        if password.isdigit():
            raise serializers.ValidationError(
                {"password": "This password is entirely numeric."}
            )
            
        if password.lower() in ["password", "123456", "12345678", "qwerty", "abc123"]:
            raise serializers.ValidationError(
                {"password": "This password is too common."}
            )
            
        # Check similarity to username
        if username.lower() in password.lower() or password.lower() in username.lower():
            raise serializers.ValidationError(
                {"password": "The password is too similar to the username."}
            )
        
        return attrs

    def create(self, validated_data):
        """Create user with validated data."""
        validated_data.pop("password2")

        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
        )

        return user


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for password change."""

    old_password = serializers.CharField(
        required=True, style={"input_type": "password"}
    )
    new_password = serializers.CharField(
        required=True, validators=[validate_password], style={"input_type": "password"}
    )
    new_password2 = serializers.CharField(
        required=True, style={"input_type": "password"}, label="Confirm New Password"
    )

    def validate(self, attrs):
        """Validate that new passwords match."""
        if attrs["new_password"] != attrs["new_password2"]:
            raise serializers.ValidationError(
                {"new_password": "New password fields didn't match."}
            )
        return attrs

    def validate_old_password(self, value):
        """Validate old password is correct."""
        user = self.context["request"].user
        if not user.check_password(value):
            raise serializers.ValidationError("Old password is incorrect")
        return value


class UserProfileSerializer(serializers.ModelSerializer):
    """Detailed user profile serializer."""

    portfolio_count = serializers.SerializerMethodField()
    last_login = serializers.DateTimeField(read_only=True)

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "date_joined",
            "last_login",
            "portfolio_count",
        ]
        read_only_fields = ["id", "username", "date_joined", "last_login"]

    def get_portfolio_count(self, obj):
        """Get number of portfolios for the user."""
        return obj.portfolios.count()
