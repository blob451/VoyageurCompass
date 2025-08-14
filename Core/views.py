"""
API Views for Core app.
Handles authentication, user management, and system utilities.
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
    """Custom JWT token view with additional user data and proper error handling."""
    
    @extend_schema(
        summary="Login",
        description="Authenticate user and receive JWT tokens",
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
        """Override to add user data to response and improve error handling."""
        # Get the response from parent class
        response = super().post(request, *args, **kwargs)
        
        # Only add user data if authentication was successful
        if response.status_code == 200:
            username = request.data.get('username')
            
            # Find the authenticated user using our custom backend logic
            try:
                from django.db.models import Q
                user = User.objects.get(
                    Q(username__iexact=username) | Q(email__iexact=username)
                )
                
                user_data = UserSerializer(user).data
                response.data['user'] = user_data
                
                logger.info(f"Successful login for user: {user.username} (ID: {user.id})")
                
            except User.DoesNotExist:
                # This should not happen if JWT authentication succeeded
                logger.error(f"JWT authentication succeeded but user not found: {username}")
            except Exception as e:
                logger.error(f"Error adding user data to JWT response: {e}")
        
        return response


class RegisterView(generics.CreateAPIView):
    """User registration view."""
    
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]
    
    @extend_schema(
        summary="Register new user",
        description="Create a new user account",
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
        """Override to return tokens after registration."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate tokens for the new user
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': UserSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(APIView):
    """User profile management."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Get user profile",
        description="Get current user's profile information",
        responses={200: UserProfileSerializer}
    )
    def get(self, request):
        """Get current user profile."""
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)
    
    @extend_schema(
        summary="Update user profile",
        description="Update current user's profile information",
        request=UserProfileSerializer,
        responses={200: UserProfileSerializer}
    )
    def put(self, request):
        """Update user profile."""
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
    """Change password view."""
    
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        summary="Change password",
        description="Change the current user's password",
        request=ChangePasswordSerializer,
        responses={
            200: {'description': 'Password changed successfully'},
            400: {'description': 'Invalid password data'}
        }
    )
    def post(self, request):
        """Change user password."""
        serializer = ChangePasswordSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            
            # Generate new tokens after password change
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
    description="Check if the API is running",
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
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'VoyageurCompass API'
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def healthCheck(request):
    """Liveness probe - simple health check"""
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
    """Readiness probe - checks database connectivity"""
    logger = logging.getLogger('VoyageurCompass.health')
    
    try:
        # Ensure connection and simple DB ping
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
    description="Get usage statistics for the current user",
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
    """Get user statistics."""
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
    """Request password reset using secret answer."""
    
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
        """Generate reset token after verifying secret answer."""
        serializer = PasswordResetRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            security_profile = user.security_profile
            
            # Generate reset token
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
        """Get security question for a username."""
        username = request.query_params.get('username')
        
        if not username:
            return Response(
                {'error': 'Username parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Find user
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
    """Reset password using token."""
    
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
        """Reset password with valid token."""
        serializer = PasswordResetSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            security_profile = serializer.validated_data['security_profile']
            
            # Set new password
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            
            # Clear reset token
            security_profile.clear_reset_token()
            
            return Response({
                'message': 'Password reset successfully'
            })
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class AdminPasswordResetRequestView(APIView):
    """Request admin help for password reset."""
    
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
        """Submit password reset request to admin."""
        serializer = AdminPasswordResetRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            reset_request = serializer.save()
            
            # Log the request for admin notification
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
    """User logout view with JWT token blacklisting."""
    
    permission_classes = [AllowAny]  # Allow logout even with invalid tokens
    
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
        """Logout user and blacklist tokens."""
        try:
            from Core.models import BlacklistedToken
            from django.contrib.auth.models import User
            import jwt
            from django.conf import settings
            
            tokens_blacklisted = 0
            user = None
            
            # Try to get user from token if authenticated, otherwise try to decode token manually
            if request.user.is_authenticated:
                user = request.user
            else:
                # Try to extract user from token manually for invalid/expired tokens
                auth_header = request.META.get('HTTP_AUTHORIZATION')
                if auth_header and auth_header.startswith('Bearer '):
                    access_token = auth_header.split(' ')[1]
                    try:
                        # Decode token without verification to get user_id
                        decoded = jwt.decode(access_token, options={"verify_signature": False})
                        user_id = decoded.get('user_id')
                        if user_id:
                            user = User.objects.get(id=user_id)
                    except:
                        pass  # Continue without user if token can't be decoded
            
            # Get access token from Authorization header
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            if auth_header and auth_header.startswith('Bearer '):
                access_token = auth_header.split(' ')[1]
                
                # Blacklist access token (pass user if available, None otherwise)
                if user and BlacklistedToken.blacklist_token(
                    access_token, 
                    user, 
                    'logout'
                ):
                    tokens_blacklisted += 1
            
            # Get refresh token from request body
            refresh_token = request.data.get('refresh_token')
            if refresh_token and user:
                # Blacklist refresh token
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
                status=status.HTTP_200_OK  # Still return success since logout should always succeed
            )


class ValidateTokenView(APIView):
    """Validate JWT token and return auth status."""
    
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
        """Validate current token and return user info."""
        try:
            # If we reach here, the token is valid (IsAuthenticated passed)
            user = request.user
            
            # Get token expiration from the JWT
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            token_expires_at = None
            
            if auth_header and auth_header.startswith('Bearer '):
                access_token = auth_header.split(' ')[1]
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
                    pass  # Continue without expiration info if decode fails
            
            return Response({
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
            })
            
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return Response(
                {
                    'valid': False,
                    'detail': 'Token validation failed'
                },
                status=status.HTTP_401_UNAUTHORIZED
            )