"""
Custom authentication backends for VoyageurCompass.
Flexible user authentication supporting username or email credentials.
"""

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
from django.db.models import Q
import logging

logger = logging.getLogger(__name__)


class EmailOrUsernameModelBackend(ModelBackend):
    """Custom authentication backend supporting username or email credentials."""
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        """Authenticate user with username or email credentials."""
        if username is None or password is None:
            return None
        
        try:
            # Try to find user by username or email
            user = User.objects.get(
                Q(username__iexact=username) | Q(email__iexact=username)
            )
            
            # Check password
            if user.check_password(password):
                logger.info(f"Successful authentication for user: {user.username}")
                return user
            else:
                logger.warning(f"Password check failed for user: {username}")
                return None
                
        except User.DoesNotExist:
            logger.warning(f"User not found: {username}")
            return None
        except User.MultipleObjectsReturned:
            logger.error(f"Multiple users found for: {username}")
            return None
    
    def get_user(self, user_id):
        """Retrieve user by primary key identifier."""
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None


class BlacklistCheckMiddleware:
    """JWT token blacklist verification middleware."""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Skip blacklist check for logout endpoint (allow logout even with invalid tokens)
        if request.path == '/api/v1/auth/logout/':
            response = self.get_response(request)
            return response
            
        # Check for blacklisted tokens before processing request
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            
            try:
                from django.core.cache import cache
                import hashlib
                
                # Check Redis cache first (60-second cache)
                cache_key = f"blacklist_check:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    if cached_result:  # Token is blacklisted
                        from django.http import JsonResponse
                        return JsonResponse(
                            {'detail': 'Token has been blacklisted'}, 
                            status=401
                        )
                    # Token is not blacklisted (cached as False), continue
                else:
                    # Check database and cache result
                    from Core.models import BlacklistedToken
                    is_blacklisted = BlacklistedToken.is_token_blacklisted(token)
                    cache.set(cache_key, is_blacklisted, 60)  # Cache for 60 seconds
                    
                    if is_blacklisted:
                        from django.http import JsonResponse
                        return JsonResponse(
                            {'detail': 'Token has been blacklisted'}, 
                            status=401
                        )
                        
            except Exception as e:
                # Log the error but don't block the request if blacklist check fails
                logger.error(f"Error checking blacklisted token: {e}")
                # Allow request to continue if blacklist check fails (graceful degradation)
                pass
        
        response = self.get_response(request)
        return response