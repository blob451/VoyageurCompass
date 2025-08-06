"""
Authentication Service Module
Handles user authentication and authorization for VoyageurCompass.
"""

import logging
from typing import Optional, Dict
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError

logger = logging.getLogger(__name__)


class AuthenticationService:
    """
    Service class for handling authentication operations.
    """
    
    @staticmethod
    def register_user(username: str, email: str, password: str, **extra_fields) -> Dict:
        """
        Register a new user.
        
        Args:
            username: Username for the new user
            email: Email address
            password: Password
            **extra_fields: Additional user fields
        
        Returns:
            Dictionary with success status and user or error message
        """
        try:
            # Validate email
            if User.objects.filter(email=email).exists():
                return {
                    'success': False,
                    'error': 'Email already registered'
                }
            
            # Create user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                **extra_fields
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
            # Try to authenticate with username
            user = authenticate(request, username=username, password=password)
            
            # If failed, try with email
            if not user:
                try:
                    user_obj = User.objects.get(email=username)
                    user = authenticate(request, username=user_obj.username, password=password)
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