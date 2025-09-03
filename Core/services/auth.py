"""
Authentication service module.
Comprehensive user authentication and authorisation services for VoyageurCompass.
"""

import logging
import re
from typing import Optional, Dict
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.contrib.auth.password_validation import validate_password

logger = logging.getLogger(__name__)


class AuthenticationService:
    """Comprehensive authentication service with input validation and security features."""
    
    # =====================================================================
    # Input validation methods (camelCase)
    # =====================================================================
    
    @staticmethod
    def validate_username(username: str) -> str:
        """Validate username input format and constraints."""
        if not username or not isinstance(username, str):
            raise ValueError("Username must be a non-empty string")
        username = username.strip()
        if len(username) < 3 or len(username) > 150:
            raise ValueError("Username must be between 3 and 150 characters")
        # Allow alphanumeric, dots, underscores, hyphens
        if not re.match(r'^[\w.\-]+$', username):
            raise ValueError("Username contains invalid characters")
        return username
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address format and length constraints."""
        if not email or not isinstance(email, str):
            raise ValueError("Email must be a non-empty string")
        email = email.strip().lower()
        # Basic email validation
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        if len(email) > 254:  # RFC 5321 limit
            raise ValueError("Email address too long")
        return email
    
    @staticmethod
    def validate_password(password: str) -> str:
        """Validate password strength using Django's built-in validators."""
        if not password or not isinstance(password, str):
            raise ValueError("Password must be a non-empty string")
        
        # Use Django's built-in password validation
        try:
            validate_password(password)
        except ValidationError as e:
            raise ValueError(f"Password validation failed: {'; '.join(e.messages)}")
        
        return password
    
    @staticmethod
    def sanitize_input(value: str, max_length: int = 255) -> str:
        """Sanitise and limit input string length."""
        if not value:
            return ""
        # Remove control characters and limit length
        sanitised = ''.join(char for char in value if char.isprintable())
        return sanitised[:max_length].strip()
    
    
    @staticmethod
    def register_user(username: str, email: str, password: str, **extra_fields) -> Dict:
        """Register new user with validation and duplicate checking."""
        try:
            # Input validation
            username = AuthenticationService.validate_username(username)
            email = AuthenticationService.validate_email(email)
            password = AuthenticationService.validate_password(password)
            
            # Sanitise extra fields
            sanitised_extra_fields = {}
            for key, value in extra_fields.items():
                if key in ['first_name', 'last_name'] and isinstance(value, str):
                    sanitised_extra_fields[key] = AuthenticationService.sanitize_input(value, 30)
                else:
                    sanitised_extra_fields[key] = value
            
            # Validate email
            if User.objects.filter(email=email).exists():
                return {
                    'success': False,
                    'error': 'Email already registered'
                }
            
            # Create user with validated data
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                **sanitised_extra_fields
            )
            
            logger.info(f"New user registered: {username}")
            return {
                'success': True,
                'user': user,
                'message': 'User registered successfully'
            }
            
        except IntegrityError:
            return {
                'success': False,
                'error': 'Username already exists'
            }
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    
    @staticmethod
    def authenticate_user(request, username: str, password: str) -> Dict:
        """
        Authenticate a user and log them in.
        
        Args:
            request: Django request object
            username: Username or email
            password: Password
        
        Returns:
            Dictionary with success status and user or error message
        """
        try:
            # Input validation - allow email or username format
            if not username or not isinstance(username, str):
                raise ValueError("Username must be a non-empty string")
            if not password or not isinstance(password, str):
                raise ValueError("Password must be a non-empty string")
            
            username = username.strip()
            
            # Determine if input is email or username
            is_email = '@' in username
            if '@' in username:
                email = AuthenticationService.validate_email(username)
                # Try to get username from email
                try:
                    userObj = User.objects.get(email=email)
                    user = authenticate(request, username=userObj.username, password=password)
                except User.DoesNotExist:
                    # Email not found, authentication will fail
                    user = None
            else:
                # Regular username login
                username = AuthenticationService.validate_username(username)
                user = authenticate(request, username=username, password=password)
            
            # If email login failed, try one more time with the email as username
            # (in case someone registered with email as their username)
            if not user and is_email:
                try:
                    # Try authenticating with email string as username
                    user = authenticate(request, username=email, password=password)
                except User.DoesNotExist:
                    pass
            
            if user is not None:
                if user.is_active:
                    login(request, user)
                    logger.info(f"User logged in: {user.username}")
                    return {
                        'success': True,
                        'user': user,
                        'message': 'Login successful'
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Account is disabled'
                    }
            else:
                return {
                    'success': False,
                    'error': 'Invalid credentials'
                }
                
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    
    @staticmethod
    def logout_user(request) -> Dict:
        """
        Log out the current user.
        
        Args:
            request: Django request object
        
        Returns:
            Dictionary with success status
        """
        try:
            if request.user.is_authenticated:
                username = request.user.username
                logout(request)
                logger.info(f"User logged out: {username}")
                return {
                    'success': True,
                    'message': 'Logout successful'
                }
            else:
                return {
                    'success': False,
                    'error': 'No user logged in'
                }
        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    
    @staticmethod
    def change_password(user: User, old_password: str, new_password: str) -> Dict:
        """
        Change user password.
        
        Args:
            user: User object
            old_password: Current password
            new_password: New password
        
        Returns:
            Dictionary with success status
        """
        try:
            # Input validation
            if not isinstance(old_password, str) or not isinstance(new_password, str):
                raise ValueError("Passwords must be strings")
            
            # Validate new password
            new_password = AuthenticationService.validate_password(new_password)
            
            if not user.check_password(old_password):
                return {
                    'success': False,
                    'error': 'Current password is incorrect'
                }
            
            user.set_password(new_password)
            user.save()
            
            logger.info(f"Password changed for user: {user.username}")
            return {
                'success': True,
                'message': 'Password changed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    
    @staticmethod
    def get_user_profile(user: User) -> Dict:
        """
        Get user profile information.
        
        Args:
            user: User object
        
        Returns:
            Dictionary with user profile data
        """
        # Validate user object
        if not isinstance(user, User):
            raise ValueError("Invalid user object")
        
        return {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'date_joined': user.date_joined.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'is_active': user.is_active,
        }


# Singleton instance
auth_service = AuthenticationService()