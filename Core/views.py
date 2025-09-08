"""
Core application API views.
Authentication, user management, and system utilities.
"""

import logging
from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth.models import User
from django.db import connection
from drf_spectacular.utils import extend_schema

logger = logging.getLogger(__name__)

from Core.serializers import (
    UserSerializer, UserRegistrationSerializer,
    ChangePasswordSerializer, UserProfileSerializer,
    PasswordResetRequestSerializer, PasswordResetSerializer,
    AdminPasswordResetRequestSerializer
)
from Core.models import UserSecurityProfile


class CustomTokenObtainPairView(TokenObtainPairView):
    """JWT authentication endpoint with enhanced user response data."""
    
    @extend_schema(
        summary="Login",
        description="User authentication with JWT token generation",
        responses={
            200: {
                'description': 'Login successful',
                'example': {
                    'access': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                    'refresh': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                    'user': {
                        'id': 1,
                        'username': 'john_doe',
                        'email': 'john@example.com'
                    }
                }
            },
            401: {
                'description': 'Invalid credentials',
                'example': {
                    'detail': 'No active account found with the given credentials'
                }
            }
        }
    )
    def post(self, request, *args, **kwargs):
        """Authenticate user and append profile data to JWT response."""
        response = super().post(request, *args, **kwargs)
        
        # Add user data on successful authentication
        if response.status_code == 200:
            username = request.data.get('username')
            
            try:
                from django.db.models import Q
                user = User.objects.get(
                    Q(username__iexact=username) | Q(email__iexact=username)
                )
                
                user_data = UserSerializer(user).data
                response.data['user'] = user_data
                
                logger.info(f"Successful login for user: {user.username} (ID: {user.id})")
                
            except User.DoesNotExist:
                logger.error(f"JWT authentication succeeded but user not found: {username}")
            except Exception as e:
                logger.error(f"Error adding user data to JWT response: {e}")
        
        return response


class RegisterView(generics.CreateAPIView):
    """User account creation endpoint."""
    
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Register new user",
        description="User account creation with automatic token generation",
        responses={
            201: {
                'description': 'User created successfully',
                'example': {
                    'user': {
                        'id': 1,
                        'username': 'john_doe',
                        'email': 'john@example.com'
                    },
                    'tokens': {
                        'access': 'eyJ0eXAiOiJKV1QiLCJhbGc...',
                        'refresh': 'eyJ0eXAiOiJKV1QiLCJhbGc...'
                    }
                }
            }
        }
    )
    def create(self, request, *args, **kwargs):
        """Create user account and generate authentication tokens."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': UserSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(APIView):
    """User profile retrieval and modification endpoint."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Get user profile",
        description="Current user profile retrieval",
        responses={200: UserProfileSerializer}
    )
    def get(self, request):
        """Retrieve authenticated user's profile data."""
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)
    
    @extend_schema(
        summary="Update user profile",
        description="User profile modification",
        request=UserProfileSerializer,
        responses={200: UserProfileSerializer}
    )
    def put(self, request):
        """Modify authenticated user's profile attributes."""
        serializer = UserProfileSerializer(
            request.user, 
            data=request.data, 
            partial=True
        )
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class ChangePasswordView(APIView):
    """Authenticated password modification endpoint."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Change password",
        description="Authenticated user password modification",
        request=ChangePasswordSerializer,
        responses={
            200: {'description': 'Password changed successfully'},
            400: {'description': 'Invalid password data'}
        }
    )
    def post(self, request):
        """Update user password and regenerate authentication tokens."""
        serializer = ChangePasswordSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            
            refresh = RefreshToken.for_user(user)
            
            return Response({
                'message': 'Password changed successfully',
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                }
            })
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


@extend_schema(
    summary="Health check",
    description="API health status verification",
    responses={
        200: {
            'description': 'API is healthy',
            'example': {
                'status': 'healthy',
                'version': '1.0.0'
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Basic API health status endpoint."""
    return Response({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'VoyageurCompass API'
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def healthCheck(request):
    """Kubernetes liveness probe endpoint."""
    from datetime import datetime, timezone
    return Response({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'requestId': getattr(request, 'correlation_id', None),
    }, headers={
        'Cache-Control': 'no-store'
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def readinessCheck(request):
    """Kubernetes readiness probe with database connectivity verification."""
    logger = logging.getLogger('VoyageurCompass.health')
    
    try:
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        from datetime import datetime, timezone
        return Response({
                'status': 'ready',
                'database': 'connected',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'requestId': getattr(request, 'correlation_id', None),
            },
            headers={'Cache-Control': 'no-store'},
        )
    except Exception as e:
        logger.error('Readiness check failed', exc_info=True)
        from datetime import datetime, timezone
        return Response({
                'status': 'not ready',
                'database': 'disconnected',
                'error': 'database connection failure',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'requestId': getattr(request, 'correlation_id', None),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={'Cache-Control': 'no-store'},
        )


@extend_schema(
    summary="Get API statistics",
    description="Current user usage statistics",
    responses={
        200: {
            'description': 'User statistics',
            'example': {
                'portfolios': 3,
                'total_holdings': 15,
                'analyses_today': 5
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_stats(request):
    """Retrieve authenticated user's usage statistics."""
    user = request.user
    
    stats = {
        'username': user.username,
        'portfolios': user.portfolios.count(),
        'total_holdings': sum(
            p.holdings.count() for p in user.portfolios.all()
        ),
        'member_since': user.date_joined.isoformat(),
        'last_login': user.last_login.isoformat() if user.last_login else None,
    }
    
    return Response(stats)


class PasswordResetRequestView(APIView):
    """Password reset token generation via security question validation."""
    
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Request password reset",
        description="Request a password reset token by answering security question",
        request=PasswordResetRequestSerializer,
        responses={
            200: {
                'description': 'Reset token generated',
                'example': {
                    'message': 'Reset token sent',
                    'token': 'uuid-token-here',
                    'secret_question': 'Your security question'
                }
            }
        }
    )
    def post(self, request):
        """Generate password reset token upon security answer verification."""
        serializer = PasswordResetRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            security_profile = user.security_profile
            
            token = security_profile.generate_reset_token()
            
            return Response({
                'message': 'Password reset token generated',
                'token': str(token),
                'secret_question': security_profile.secret_question
            })
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
    @extend_schema(
        summary="Get security question",
        description="Get the security question for a user",
        responses={
            200: {
                'description': 'Security question retrieved',
                'example': {
                    'secret_question': 'What is your mother\'s maiden name?'
                }
            }
        }
    )
    def get(self, request):
        """Retrieve security question for specified username."""
        username = request.query_params.get('username')
        
        if not username:
            return Response(
                {'error': 'Username parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
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
            return Response(
                {'error': 'User not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        try:
            security_profile = user.security_profile
            return Response({
                'secret_question': security_profile.secret_question
            })
        except:
            return Response(
                {'error': 'Security profile not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class PasswordResetConfirmView(APIView):
    """Password reset execution with validated token."""
    
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Reset password",
        description="Reset password using the token from password reset request",
        request=PasswordResetSerializer,
        responses={
            200: {'description': 'Password reset successfully'}
        }
    )
    def post(self, request):
        """Execute password reset using validated token."""
        serializer = PasswordResetSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            security_profile = serializer.validated_data['security_profile']
            
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            
            security_profile.clear_reset_token()
            
            return Response({
                'message': 'Password reset successfully'
            })
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class AdminPasswordResetRequestView(APIView):
    """Administrative password reset assistance request endpoint."""
    
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Request admin password reset",
        description="Request admin assistance when secret answer is forgotten",
        request=AdminPasswordResetRequestSerializer,
        responses={
            201: {
                'description': 'Request submitted',
                'example': {
                    'message': 'Password reset request submitted to admin',
                    'request_id': 1
                }
            }
        }
    )
    def post(self, request):
        """Submit administrative password reset request."""
        serializer = AdminPasswordResetRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            reset_request = serializer.save()
            
            import logging
            logger = logging.getLogger('VoyageurCompass.admin')
            logger.warning(
                f"Password reset requested for user {reset_request.user.username}: {reset_request.reason}"
            )
            
            return Response({
                'message': 'Password reset request submitted to admin. You will be contacted soon.',
                'request_id': reset_request.id
            }, status=status.HTTP_201_CREATED)
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class LogoutView(APIView):
    """User session termination with JWT token blacklisting."""
    
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Logout user",
        description="Logout user and blacklist JWT tokens",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'refresh_token': {
                        'type': 'string',
                        'description': 'Refresh token to blacklist'
                    }
                }
            }
        },
        responses={
            200: {
                'description': 'Logout successful',
                'example': {
                    'message': 'Logout successful',
                    'tokens_blacklisted': 2
                }
            }
        }
    )
    def post(self, request):
        """Terminate user session and blacklist JWT tokens."""
        try:
            from Core.models import BlacklistedToken
            from django.contrib.auth.models import User
            import jwt
            from django.conf import settings
            
            tokens_blacklisted = 0
            user = None
            
            if request.user.is_authenticated:
                user = request.user
            else:
                auth_header = request.META.get('HTTP_AUTHORIZATION')
                if auth_header and auth_header.startswith('Bearer '):
                    access_token = auth_header.split(' ')[1]
                    try:
                        decoded = jwt.decode(access_token, options={"verify_signature": False})
                        user_id = decoded.get('user_id')
                        if user_id:
                            user = User.objects.get(id=user_id)
                    except:
                        pass
            
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            if auth_header and auth_header.startswith('Bearer '):
                access_token = auth_header.split(' ')[1]
                
                if user and BlacklistedToken.blacklist_token(
                    access_token, 
                    user, 
                    'logout'
                ):
                    tokens_blacklisted += 1
            
            refresh_token = request.data.get('refresh_token')
            if refresh_token and user:
                if BlacklistedToken.blacklist_token(
                    refresh_token, 
                    user, 
                    'logout'
                ):
                    tokens_blacklisted += 1
            
            username = user.username if user else 'unknown'
            logger.info(f"User logged out: {username}, tokens blacklisted: {tokens_blacklisted}")
            
            return Response({
                'message': 'Logout successful',
                'tokens_blacklisted': tokens_blacklisted
            })
            
        except Exception as e:
            username = request.user.username if hasattr(request.user, 'username') else 'unknown'
            logger.error(f"Error during logout for user {username}: {str(e)}")
            return Response(
                {'message': 'Logout completed (with errors)', 'tokens_blacklisted': 0},
                status=status.HTTP_200_OK
            )


class ValidateTokenView(APIView):
    """JWT token validation with authentication status verification."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Validate authentication token",
        description="Check if the current JWT token is valid and return user info",
        responses={
            200: {
                'description': 'Token is valid',
                'example': {
                    'valid': True,
                    'user': {
                        'id': 1,
                        'username': 'john_doe',
                        'email': 'john@example.com'
                    },
                    'token_expires_at': '2025-01-15T12:00:00Z'
                }
            },
            401: {
                'description': 'Token is invalid or expired',
                'example': {
                    'valid': False,
                    'detail': 'Token is invalid or expired'
                }
            }
        }
    )
    def get(self, request):
        """Validate JWT token and return user profile information with Redis caching."""
        try:
            user = request.user
            
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            token_expires_at = None
            
            if auth_header and auth_header.startswith('Bearer '):
                access_token = auth_header.split(' ')[1]
                
                # Check Redis cache first for token validation
                from django.core.cache import cache
                import hashlib
                cache_key = f"token_validation:{hashlib.sha256(access_token.encode()).hexdigest()[:16]}"
                cached_result = cache.get(cache_key)
                
                if cached_result:
                    cached_result['user']['id'] = user.id  # Ensure current user ID
                    return Response(cached_result)
                
                try:
                    import jwt
                    from django.conf import settings
                    decoded = jwt.decode(
                        access_token, 
                        settings.SECRET_KEY, 
                        algorithms=["HS256"]
                    )
                    from django.utils import timezone
                    token_expires_at = timezone.datetime.fromtimestamp(
                        decoded['exp'], 
                        tz=timezone.utc
                    ).isoformat()
                except:
                    pass
            
            response_data = {
                'valid': True,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'date_joined': user.date_joined.isoformat()
                },
                'token_expires_at': token_expires_at
            }
            
            # Cache the validation result for 2 minutes
            if auth_header:
                cache.set(cache_key, response_data, 120)
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return Response(
                {
                    'valid': False,
                    'detail': 'Token validation failed'
                },
                status=status.HTTP_401_UNAUTHORIZED
            )
